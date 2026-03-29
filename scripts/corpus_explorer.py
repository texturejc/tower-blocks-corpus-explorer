"""
Corpus Explorer: TF-IDF + PCA + Topic Modelling
Builds an interactive 3D Plotly scatterplot of the Tower Blocks Corpus.
"""
import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import plotly.express as px

CORPUS_DIR = '/Users/fhsr/Desktop/WESA_13_1/text_data/Tower Blocks Corpus Cleaned'
WESA_13_1_FILE = '/Users/fhsr/Desktop/WESA_13_1/text_data/WESA_13_1_documents.txt'
OUTPUT_DIR = '/Users/fhsr/Desktop/WESA_13_1/text_data'

# ============================================================
# STEP 1: Parse all corpus files into a single DataFrame
# ============================================================
print("Step 1: Extracting documents from corpus files...")

def parse_corpus_file(filepath):
    """Parse a cleaned corpus file into list of document dicts."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    docs = []
    # Split on the delimiter pattern
    blocks = re.split(r'={20,}', content)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Try to extract metadata
        archive_match = re.search(r'^ARCHIVE_FOLDER:\s*(.+)$', block, re.M)
        doc_match = re.search(r'^DOCUMENT(?:_ID)?:\s*(.+)$', block, re.M)
        date_match = re.search(r'^DATE:\s*(.+)$', block, re.M)
        place_match = re.search(r'^PLACE:\s*(.+)$', block, re.M)

        if not doc_match:
            # This block doesn't have metadata — might be a continuation
            # or just content without headers. Skip if no doc_id.
            continue

        # Extract text after the "-----" separator
        parts = re.split(r'-{20,}', block, maxsplit=1)
        if len(parts) > 1:
            text = parts[1].strip()
        else:
            # No separator found — take everything after the last metadata line
            lines = block.split('\n')
            meta_end = 0
            for j, line in enumerate(lines):
                if re.match(r'^(ARCHIVE_FOLDER|DOCUMENT|DATE|PLACE|SOURCE_IMAGES|PAGES)[\s_]*:', line):
                    meta_end = j + 1
            text = '\n'.join(lines[meta_end:]).strip()

        docs.append({
            'doc_id': doc_match.group(1).strip(),
            'archive_folder': archive_match.group(1).strip() if archive_match else '',
            'date': date_match.group(1).strip() if date_match else 'Unknown',
            'place': place_match.group(1).strip() if place_match else 'Not stated',
            'text': text,
            'source_file': os.path.basename(filepath),
        })

    return docs

# Parse all files
all_docs = []
for fname in sorted(os.listdir(CORPUS_DIR)):
    if not fname.endswith('.txt'):
        continue
    # Skip the raw WESA_13_1 file — we'll use our grouped version
    if fname == 'WESA_13_1__cleaned.txt':
        continue
    filepath = os.path.join(CORPUS_DIR, fname)
    docs = parse_corpus_file(filepath)
    all_docs.extend(docs)
    print(f"  {fname}: {len(docs)} docs")

# Add our grouped WESA_13_1 documents
wesa_docs = parse_corpus_file(WESA_13_1_FILE)
all_docs.extend(wesa_docs)
print(f"  WESA_13_1_documents.txt (grouped): {len(wesa_docs)} docs")

df = pd.DataFrame(all_docs)
print(f"\nTotal documents parsed: {len(df)}")

# Derive the archive series (TBUK / WESA / VF_NEW)
def get_series(source_file):
    if source_file.startswith('TBUK'):
        return 'TBUK'
    elif source_file.startswith('VF_NEW') or source_file.startswith('VF:'):
        return 'VF_NEW'
    else:
        return 'WESA'

df['series'] = df['source_file'].apply(get_series)
# Also handle WESA_13_1_documents.txt
df.loc[df['source_file'] == 'WESA_13_1_documents.txt', 'series'] = 'WESA'

# Add word count and filter out very short docs
df['word_count'] = df['text'].apply(lambda t: len(t.split()))
n_before = len(df)
df = df[df['word_count'] >= 20].reset_index(drop=True)
print(f"Filtered {n_before - len(df)} short docs (< 20 words). Remaining: {len(df)}")

# Save the full corpus DataFrame
df.to_pickle(os.path.join(OUTPUT_DIR, 'corpus_documents.pkl'))

# ============================================================
# STEP 2: TF-IDF vectorisation
# ============================================================
print("\nStep 2: Building TF-IDF matrix...")

custom_stop_words = [
    'cont', 'continued', 'page', 'mr', 'mrs', 'dr', 'sir',
    'said', 'would', 'shall', 'may', 'also', 'one', 'two',
    'yes', 'sir', 'think', 'know', 'say', 'right', 'see',
    'come', 'go', 'got', 'get', 'let', 'make', 'take',
    'like', 'just', 'well', 'quite', 'rather', 'much',
    'ee', 'ae', 'oe', 'se', 'ne', 're',  # OCR artifacts
]

tfidf = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.85,
    stop_words='english',
    ngram_range=(1, 2),
    token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b',  # letters only, min 2 chars
)

tfidf_matrix = tfidf.fit_transform(df['text'])
feature_names = tfidf.get_feature_names_out()
print(f"TF-IDF matrix: {tfidf_matrix.shape[0]} docs × {tfidf_matrix.shape[1]} features")

# ============================================================
# STEP 3: Dimensionality reduction — PCA to 3D
# ============================================================
print("\nStep 3: Reducing to 3D via TruncatedSVD + PCA...")

# TruncatedSVD works on sparse matrices (50 components)
n_components_svd = min(50, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
svd = TruncatedSVD(n_components=n_components_svd, random_state=42)
reduced_50 = svd.fit_transform(tfidf_matrix)
print(f"  SVD variance explained ({n_components_svd} components): {svd.explained_variance_ratio_.sum():.1%}")

# PCA from 50 → 3
pca = PCA(n_components=3, random_state=42)
coords_3d = pca.fit_transform(reduced_50)
print(f"  PCA 3D variance explained: {pca.explained_variance_ratio_.sum():.1%}")

df['PC1'] = coords_3d[:, 0]
df['PC2'] = coords_3d[:, 1]
df['PC3'] = coords_3d[:, 2]

# ============================================================
# STEP 4: Topic modelling with Gensim LDA
# ============================================================
print("\nStep 4: Running LDA topic modelling...")

N_TOPICS = 15

# Use bag-of-words from raw text for LDA (works much better than TF-IDF weights)
from gensim.utils import simple_preprocess

# Tokenize documents
stop_set = set(custom_stop_words) | set(TfidfVectorizer(stop_words='english').get_stop_words())
texts_tokenized = []
for text in df['text']:
    tokens = simple_preprocess(text, deacc=True, min_len=3)
    tokens = [t for t in tokens if t not in stop_set]
    texts_tokenized.append(tokens)

# Build gensim dictionary and corpus
dictionary = Dictionary(texts_tokenized)
dictionary.filter_extremes(no_below=3, no_above=0.7)
corpus_bow = [dictionary.doc2bow(doc) for doc in texts_tokenized]

lda = LdaModel(
    corpus=corpus_bow,
    id2word=dictionary,
    num_topics=N_TOPICS,
    passes=20,
    iterations=200,
    random_state=42,
    alpha='symmetric',
    eta='auto',
    chunksize=200,
)

# Assign each document to its dominant topic
topic_assignments = []
topic_distributions = []
for bow in corpus_bow:
    topic_dist = lda.get_document_topics(bow, minimum_probability=0.0)
    topic_dist_dict = {t: p for t, p in topic_dist}
    dominant = max(topic_dist_dict, key=topic_dist_dict.get)
    topic_assignments.append(dominant)
    topic_distributions.append(topic_dist_dict)

df['topic_num'] = topic_assignments

# Name topics from top keywords
topic_names = {}
topic_keywords = {}
for topic_id in range(N_TOPICS):
    top_words = lda.show_topic(topic_id, topn=6)
    keywords = [w for w, _ in top_words]
    # Filter out very short/noise words
    keywords = [w for w in keywords if len(w) > 2 and w not in custom_stop_words][:5]
    name = ' / '.join(keywords[:4])
    topic_names[topic_id] = name
    topic_keywords[topic_id] = keywords

df['topic'] = df['topic_num'].map(topic_names)

# Print topic summary
print(f"\n  {N_TOPICS} topics found:")
for tid in range(N_TOPICS):
    n_docs = (df['topic_num'] == tid).sum()
    print(f"    Topic {tid:2d} ({n_docs:3d} docs): {topic_names[tid]}")

# Save topic assignments
topics_df = df[['doc_id', 'archive_folder', 'series', 'topic_num', 'topic']].copy()
topics_df['keywords'] = topics_df['topic_num'].map(lambda t: ', '.join(topic_keywords.get(t, [])))
topics_df.to_csv(os.path.join(OUTPUT_DIR, 'corpus_topics.csv'), index=False)

# ============================================================
# STEP 5: Interactive 3D Plotly scatterplot
# ============================================================
print("\nStep 5: Building interactive 3D plot...")

# Truncated hover text — just first 80 chars
df['text_preview'] = df['text'].apply(
    lambda t: t[:80].replace('\n', ' ').replace('<', '&lt;').strip() + '...'
)

fig = px.scatter_3d(
    df,
    x='PC1', y='PC2', z='PC3',
    color='topic',
    hover_name='doc_id',
    hover_data={
        'PC1': False, 'PC2': False, 'PC3': False,
    },
    custom_data=['archive_folder', 'date', 'topic', 'text_preview'],
    title='Tower Blocks Corpus — Document Explorer',
    labels={'PC1': 'Component 1', 'PC2': 'Component 2', 'PC3': 'Component 3'},
    width=1200,
    height=800,
)

# Compact hover
fig.update_traces(
    hovertemplate=(
        '<b>%{hovertext}</b> | %{customdata[0]}<br>'
        '%{customdata[2]}<br>'
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
    scene=dict(
        xaxis_title='Ledbury / Southwark  ←→  Ronan Point / Newham',
        yaxis_title='Structural / Engineering  ←→  Tenants / Policy',
        zaxis_title='Inquiry Transcripts  ←→  General Documents',
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    annotations=[
        dict(
            text=(
                '<b>How to read this plot</b><br>'
                'Each dot is a document from the Tower Blocks archive. '
                'Documents with similar language appear closer together.<br>'
                'Colours show automatically detected topics. '
                'Hover over a dot to see its details. '
                'Click a topic in the legend to show/hide it.<br>'
                'The three axes capture the main ways documents differ: '
                'which buildings they discuss, whether they are technical or policy-focused, '
                'and whether they are inquiry transcripts.'
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

html_path = os.path.join(OUTPUT_DIR, 'corpus_explorer.html')
fig.write_html(html_path, include_plotlyjs=True)

# ============================================================
# STEP 6: Variance analysis — are PCA dimensions interpretable?
# ============================================================
print("\n=== PCA Variance Analysis ===")
svd_total = svd.explained_variance_ratio_.sum()
pca_per_axis = pca.explained_variance_ratio_
pca_total = pca_per_axis.sum()
# The 3D PCA operates on the 50-dim SVD space, so effective variance
# relative to the full TF-IDF space is:
effective = [r * svd_total for r in pca_per_axis]
effective_total = sum(effective)

print(f"  TruncatedSVD: 50 components explain {svd_total:.1%} of total TF-IDF variance")
print(f"  PCA 3D within SVD space: {pca_total:.1%}")
print(f"    PC1: {pca_per_axis[0]:.1%}  (effective: {effective[0]:.1%} of full)")
print(f"    PC2: {pca_per_axis[1]:.1%}  (effective: {effective[1]:.1%} of full)")
print(f"    PC3: {pca_per_axis[2]:.1%}  (effective: {effective[2]:.1%} of full)")
print(f"  Effective 3D total: {effective_total:.1%} of full TF-IDF variance")
print()

# What do the PCA axes correlate with? Top TF-IDF features per axis.
# Project the SVD components through PCA to get feature loadings
# svd.components_ is (50, n_features), pca.components_ is (3, 50)
# combined loadings = pca.components_ @ svd.components_ → (3, n_features)
loadings = pca.components_ @ svd.components_
print("  Top features driving each axis:")
for ax in range(3):
    top_pos = np.argsort(loadings[ax])[-6:][::-1]
    top_neg = np.argsort(loadings[ax])[:6]
    pos_words = [feature_names[i] for i in top_pos]
    neg_words = [feature_names[i] for i in top_neg]
    print(f"    PC{ax+1} (+): {', '.join(pos_words)}")
    print(f"    PC{ax+1} (-): {', '.join(neg_words)}")

print(f"\nDone!")
print(f"  Documents: {len(df)}")
print(f"  Topics: {N_TOPICS}")
print(f"  Outputs:")
print(f"    {os.path.join(OUTPUT_DIR, 'corpus_documents.pkl')}")
print(f"    {os.path.join(OUTPUT_DIR, 'corpus_topics.csv')}")
print(f"    {html_path}")
