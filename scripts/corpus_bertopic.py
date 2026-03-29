"""
BERTopic analysis of the Tower Blocks Corpus.
Compares against our LDA topic assignments.
"""
import os
import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import plotly.express as px

OUTPUT_DIR = '/Users/fhsr/Desktop/WESA_13_1/text_data'

# ============================================================
# STEP 1: Load corpus
# ============================================================
print("Step 1: Loading corpus...")
df = pd.read_pickle(os.path.join(OUTPUT_DIR, 'corpus_documents.pkl'))

# Merge LDA topics for comparison
topics_csv = os.path.join(OUTPUT_DIR, 'corpus_topics.csv')
if os.path.exists(topics_csv):
    lda_topics = pd.read_csv(topics_csv)
    lda_map = dict(zip(lda_topics['doc_id'], lda_topics['topic']))
    df['lda_topic'] = df['doc_id'].map(lda_map).fillna('unassigned')

docs = df['text'].tolist()
print(f"  {len(docs)} documents")

# ============================================================
# STEP 2: Generate embeddings
# ============================================================
print("\nStep 2: Generating sentence embeddings...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(docs, show_progress_bar=True, batch_size=32)
print(f"  Embeddings shape: {embeddings.shape}")

# ============================================================
# STEP 3: Fit BERTopic
# ============================================================
print("\nStep 3: Fitting BERTopic...")

from hdbscan import HDBSCAN
from umap import UMAP as UMAP_model

# Use tighter UMAP + more sensitive HDBSCAN to find subtler clusters
umap_model = UMAP_model(
    n_components=10,
    n_neighbors=10,
    min_dist=0.0,
    metric='cosine',
    random_state=42,
)

hdbscan_model = HDBSCAN(
    min_cluster_size=12,
    min_samples=3,
    metric='euclidean',
    prediction_data=True,
)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    nr_topics=18,
    verbose=True,
)

topics, probs = topic_model.fit_transform(docs, embeddings)

df['bert_topic_num'] = topics

# Get topic info
topic_info = topic_model.get_topic_info()
print(f"\n  Topics found: {len(topic_info) - 1} (plus outlier topic -1)")
print(f"  Outlier docs: {(df['bert_topic_num'] == -1).sum()}")
print(f"\n  Topic distribution:")
for _, row in topic_info.head(20).iterrows():
    print(f"    Topic {row['Topic']:3d} ({row['Count']:4d} docs): {row['Name']}")

# Build readable topic names
bert_topic_names = {}
for topic_id in sorted(df['bert_topic_num'].unique()):
    if topic_id == -1:
        bert_topic_names[-1] = 'Outliers / Mixed'
    else:
        words = topic_model.get_topic(topic_id)
        top_words = [w for w, _ in words[:4]]
        bert_topic_names[topic_id] = ' / '.join(top_words)

df['bert_topic'] = df['bert_topic_num'].map(bert_topic_names)

# ============================================================
# STEP 4: Compare BERTopic vs LDA
# ============================================================
print("\n=== BERTopic vs LDA Comparison ===")

# Cross-tabulation
if 'lda_topic' in df.columns:
    ct = pd.crosstab(df['bert_topic'], df['lda_topic'])
    print(f"\n  Cross-tabulation ({ct.shape[0]} BERT topics × {ct.shape[1]} LDA topics)")

    # For each BERT topic, find the most common LDA topic
    print("\n  BERT topic → dominant LDA topic:")
    for bt in sorted(df['bert_topic_num'].unique()):
        bt_docs = df[df['bert_topic_num'] == bt]
        top_lda = bt_docs['lda_topic'].value_counts().iloc[0]
        top_lda_name = bt_docs['lda_topic'].value_counts().index[0]
        pct = top_lda / len(bt_docs) * 100
        print(f"    BERT {bt:3d} ({len(bt_docs):3d} docs) '{bert_topic_names[bt][:40]}' → LDA '{top_lda_name[:40]}' ({pct:.0f}%)")

# ============================================================
# STEP 5: 3D plot using UMAP (BERTopic's native reduction)
# ============================================================
print("\nStep 5: Building 3D plot...")

# Use UMAP on the embeddings for 3D coordinates
from umap import UMAP
reducer = UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
coords_3d = reducer.fit_transform(embeddings)

df['UMAP1'] = coords_3d[:, 0]
df['UMAP2'] = coords_3d[:, 1]
df['UMAP3'] = coords_3d[:, 2]

df['text_preview'] = df['text'].apply(
    lambda t: t[:80].replace('\n', ' ').replace('<', '&lt;').strip() + '...'
)

fig = px.scatter_3d(
    df,
    x='UMAP1', y='UMAP2', z='UMAP3',
    color='bert_topic',
    hover_name='doc_id',
    custom_data=['archive_folder', 'date', 'bert_topic', 'text_preview', 'lda_topic'],
    title='Tower Blocks Corpus — BERTopic Clusters (UMAP 3D)',
    width=1200,
    height=800,
)

fig.update_traces(
    hovertemplate=(
        '<b>%{hovertext}</b> | %{customdata[0]}<br>'
        'BERTopic: %{customdata[2]}<br>'
        'LDA: %{customdata[4]}<br>'
        '%{customdata[3]}'
        '<extra></extra>'
    ),
    hovertext=df['doc_id'],
    marker=dict(size=4),
)

fig.update_layout(
    legend=dict(
        font=dict(size=9),
        itemsizing='constant',
        yanchor='top', y=0.99,
        xanchor='left', x=0.01,
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    annotations=[
        dict(
            text=(
                '<b>How to read this plot</b><br>'
                'Each dot is a document positioned by semantic similarity using a language model (all-MiniLM-L6-v2).<br>'
                'Documents that use similar language and discuss similar subjects appear closer together.<br>'
                'Colours show topics discovered by BERTopic. Hover shows both BERTopic and LDA topic assignments for comparison.'
            ),
            showarrow=False,
            xref='paper', yref='paper',
            x=0.5, y=-0.02,
            xanchor='center', yanchor='top',
            font=dict(size=11, color='#555'),
            align='center',
        ),
    ],
)

html_path = os.path.join(OUTPUT_DIR, 'corpus_bertopic_explorer.html')
fig.write_html(html_path, include_plotlyjs=True)

# Save
df.to_pickle(os.path.join(OUTPUT_DIR, 'corpus_documents_bertopic.pkl'))

print(f"\nDone!")
print(f"  {html_path}")
