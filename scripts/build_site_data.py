"""
Build the site's data/*.json files from the analysis outputs in the
WESA_13_1 working directory.

Sources (produced by the corpus_*.py pipeline):
  corpus_documents.pkl        -> documents.json, topics.json
  corpus_topics.csv           -> documents.json, topics.json
  cross_archive_bridges.csv   -> cross_archive_bridges.json
  corpus_word2vec_coords.csv  -> word_embeddings.json

Run from anywhere; paths are absolute. Re-run after regenerating the
pipeline to refresh the site data (e.g. when a new archive is added).
"""
import os
import json
import pandas as pd

SRC = '/Users/fhsr/Desktop/WESA_13_1/text_data'
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# Match the existing files: 2-space indent, no space after separators, ASCII-escaped.
def dump(obj, name):
    path = os.path.join(DATA, name)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, separators=(',', ':'), ensure_ascii=True)
    print(f"  {name}: {len(obj)} records")

# ------------------------------------------------------------------
# documents.json  (one record per corpus document)
# ------------------------------------------------------------------
df = pd.read_pickle(os.path.join(SRC, 'corpus_documents.pkl')).reset_index(drop=True)
topics = pd.read_csv(os.path.join(SRC, 'corpus_topics.csv'))
assert (df['doc_id'].values == topics['doc_id'].values).all(), "pkl / topics.csv misaligned"

documents = []
for row, trow in zip(df.itertuples(index=False), topics.itertuples(index=False)):
    documents.append({
        'doc_id': row.doc_id,
        'archive_folder': row.archive_folder,
        'source_file': row.source_file,
        'date': row.date,
        'place': row.place,
        'series': row.series,
        'word_count': int(row.word_count),
        'topic': trow.topic,
        'topic_keywords': trow.keywords,
        'text_preview': row.text.replace('\n', ' ')[:200],
    })

# ------------------------------------------------------------------
# topics.json  (one record per non-empty LDA topic)
# ------------------------------------------------------------------
topic_records = []
for topic_num, grp in topics.groupby('topic_num'):
    if len(grp) == 0:
        continue
    sample_archives = grp['archive_folder'].value_counts().head(5).index.tolist()
    topic_records.append({
        'topic_num': int(topic_num),
        'topic_name': grp['topic'].iloc[0],
        'doc_count': int(len(grp)),
        'keywords': grp['keywords'].iloc[0],
        'sample_archives': sample_archives,
    })
topic_records.sort(key=lambda t: t['topic_num'])

# ------------------------------------------------------------------
# cross_archive_bridges.json
# ------------------------------------------------------------------
bridges = pd.read_csv(os.path.join(SRC, 'cross_archive_bridges.csv'))
bridge_records = [{
    'doc_a': r.doc_a,
    'archive_a': r.archive_a,
    'doc_b': r.doc_b,
    'archive_b': r.archive_b,
    'similarity': round(float(r.similarity), 6),
    'shared_terms': r.shared_terms,
} for r in bridges.itertuples(index=False)]

# ------------------------------------------------------------------
# word_embeddings.json
# ------------------------------------------------------------------
coords = pd.read_csv(os.path.join(SRC, 'corpus_word2vec_coords.csv'))
word_records = [{
    'word': r.word,
    'PC1': float(r.PC1),
    'PC2': float(r.PC2),
    'PC3': float(r.PC3),
    'category': r.category,
    'frequency': int(r.frequency),
    'size': float(r.size),
} for r in coords.itertuples(index=False)]

print("Writing site data:")
dump(documents, 'documents.json')
dump(topic_records, 'topics.json')
dump(bridge_records, 'cross_archive_bridges.json')
dump(word_records, 'word_embeddings.json')
print("Done.")
