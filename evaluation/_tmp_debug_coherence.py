import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

from app.config import CORPUS_PATH
from topics.nmf_model import run_nmf


def _tokenize(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]


df = pd.read_parquet(CORPUS_PATH)
docs = df['text'].sample(n=1000, random_state=42).tolist()
tokenized_docs = [_tokenize(doc) for doc in docs]
dictionary = Dictionary(tokenized_docs)

nmf_topic_info, _, _ = run_nmf(docs, n_topics=10)
nmf_keywords = [topic['keywords'] for topic in nmf_topic_info if isinstance(topic.get('keywords'), list) and len(topic['keywords']) >= 2]
print('TOPICS', len(nmf_keywords))
print('FIRST', nmf_keywords[0])
print('FIRST TYPES', [type(word) for word in nmf_keywords[0]])

for coherence in ['c_v', 'c_npmi']:
    try:
        value = CoherenceModel(
            topics=nmf_keywords,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence=coherence,
            processes=1,
        ).get_coherence()
        print(coherence, value)
    except Exception as exc:
        print('FAILED', coherence, type(exc).__name__, exc)
