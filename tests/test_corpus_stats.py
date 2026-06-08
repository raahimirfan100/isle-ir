

def test_compute_corpus_stats(tmp_path, capsys, monkeypatch):
    # Small synthetic corpus (no pandas required)
    docs = [
        "apple banana apple",
        "banana orange",
        "orange apple banana",
    ]

    class FakeDF:
        def __init__(self, texts):
            self._texts = texts

        def __getitem__(self, key):
            if key == "text":
                class FakeSeries:
                    def __init__(self, texts):
                        self._texts = list(texts)

                    def sample(self, n, random_state=None):
                        return FakeSeries(self._texts[:n])

                    def tolist(self):
                        return self._texts

                    def __iter__(self):
                        return iter(self._texts)

                return FakeSeries(self._texts)
            raise KeyError(key)
        
        @property
        def empty(self):
            return len(self._texts) == 0

        def __len__(self):
            return len(self._texts)

    df = FakeDF(docs)

    # Ensure local packages under src/ can be imported (app.*, topics.*)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    # Inject lightweight fakes for heavy external libs before import
    import sys
    import types

    sys.modules.setdefault("pandas", types.SimpleNamespace(read_parquet=lambda p: None))
    sys.modules.setdefault(
        "bertopic",
        types.SimpleNamespace(
            BERTopic=type("BERTopic", (), {"load": staticmethod(lambda p: None)})
        ),
    )

    gensim = types.ModuleType("gensim")
    corpora = types.ModuleType("gensim.corpora")
    models = types.ModuleType("gensim.models")
    corpora.Dictionary = lambda *a, **k: None
    models.coherencemodel = types.SimpleNamespace(CoherenceModel=lambda *a, **k: None)
    sys.modules.setdefault("gensim", gensim)
    sys.modules.setdefault("gensim.corpora", corpora)
    sys.modules.setdefault("gensim.models", models)
    sys.modules.setdefault("gensim.models.coherencemodel", models.coherencemodel)

    # Minimal PIL stubs
    def _fake_image_new(mode, size, color):
        class Img:
            def save(self, path):
                from pathlib import Path

                Path(path).write_bytes(b"")

        return Img()

    Image = types.SimpleNamespace(new=_fake_image_new)
    ImageDraw = types.SimpleNamespace(
        Draw=lambda img: types.SimpleNamespace(
            text=lambda *a, **k: None, line=lambda *a, **k: None
        )
    )
    ImageFont = types.SimpleNamespace(
        truetype=lambda name, size: None, load_default=lambda: None
    )
    sys.modules.setdefault("PIL", types.ModuleType("PIL"))
    sys.modules.setdefault("PIL.Image", Image)
    sys.modules.setdefault("PIL.ImageDraw", ImageDraw)
    sys.modules.setdefault("PIL.ImageFont", ImageFont)

    # Stub topic modules to avoid heavy imports
    sys.modules.setdefault(
        "topics.nmf_model",
        types.SimpleNamespace(run_nmf=lambda docs, n_topics=10: ([], None, None)),
    )
    sys.modules.setdefault(
        "topics.bertopic_model",
        types.SimpleNamespace(_get_topic_info=lambda model: []),
    )

    import evaluation.corpus_stats as cs

    # Return the small dataframe instead of reading parquet
    monkeypatch.setattr(cs, "_load_corpus", lambda: df)

    # Use a temporary directory for outputs
    out_dir = tmp_path / "results"
    monkeypatch.setattr(cs, "EVAL_RESULTS_DIR", out_dir)

    # Mock run_nmf to return simple topic keywords
    nmf_topic_info = [{"keywords": ["apple", "banana"]}, {"keywords": ["orange", "banana"]}]
    monkeypatch.setattr(cs, "run_nmf", lambda docs, n_topics=10: (nmf_topic_info, None, None))

    # Mock BERTopic topic info
    bertopic_info = [{"keywords": ["apple", "orange"]}, {"keywords": ["banana", "orange"]}]
    monkeypatch.setattr(cs, "_get_topic_info", lambda model: bertopic_info)

    # Avoid downloading/loading a real BERTopic model
    monkeypatch.setattr(cs, "_load_bertopic_model", lambda: object())

    # Fake CoherenceModel to avoid heavy gensim computation
    class FakeCoherenceModel:
        def __init__(
            self,
            topics=None,
            texts=None,
            dictionary=None,
            coherence=None,
            processes=None,
        ):
            self._coherence = coherence

        def get_coherence(self):
            if self._coherence == "c_v":
                return 0.42
            if self._coherence == "c_npmi":
                return 0.13
            return 0.0

    monkeypatch.setattr(cs, "CoherenceModel", FakeCoherenceModel)

    # Run the pipeline
    cs.compute_corpus_stats()

    # Capture stdout and check that coherence scores were printed
    captured = capsys.readouterr()
    assert "NMF c_v" in captured.out
    assert "BERTopic c_v" in captured.out

    # Check that plots were created
    assert (out_dir / "heaps_law.png").exists()
    assert (out_dir / "zipf_law.png").exists()
