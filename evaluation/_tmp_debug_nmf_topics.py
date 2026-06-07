import pandas as pd

from app.config import CORPUS_PATH
from topics.nmf_model import run_nmf


df = pd.read_parquet(CORPUS_PATH)
docs = df['text'].sample(n=1000, random_state=42).tolist()
topic_info, _, _ = run_nmf(docs, n_topics=10)

for i, topic in enumerate(topic_info):
    keywords = topic['keywords']
    print('TOPIC', i, type(keywords), len(keywords), keywords[:5])
    print('ALL STR', all(isinstance(word, str) for word in keywords))
    print('ALL NONSPACE', all(' ' not in str(word) for word in keywords))
