"""
Semantic network analysis of the Tower Blocks Corpus.
1. Word co-occurrence network with community detection
2. Document similarity network with cross-archive bridges
"""
import os
import re
import math
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.utils import simple_preprocess
import plotly.graph_objects as go
import networkx as nx

OUTPUT_DIR = '/Users/fhsr/Desktop/WESA_13_1/text_data'

# ============================================================
# Load corpus
# ============================================================
print("Loading corpus...")
df = pd.read_pickle(os.path.join(OUTPUT_DIR, 'corpus_documents.pkl'))

# Merge topics
topics_csv = os.path.join(OUTPUT_DIR, 'corpus_topics.csv')
if os.path.exists(topics_csv):
    topics_df = pd.read_csv(topics_csv)
    topic_map = dict(zip(topics_df['doc_id'], topics_df['topic']))
    df['topic'] = df['doc_id'].map(topic_map).fillna('unassigned')

stop_words = set(TfidfVectorizer(stop_words='english').get_stop_words())
stop_words.update([
    'cont', 'continued', 'page', 'mr', 'mrs', 'dr', 'sir',
    'said', 'would', 'shall', 'also', 'yes', 'think',
    'know', 'say', 'right', 'see', 'come', 'got', 'get',
    'let', 'make', 'take', 'like', 'just', 'well', 'quite',
    'ee', 'ae', 'oe', 'se', 'ne', 'wesa', 'img', 'ref',
    'did', 'does', 'don', 'going', 'went', 'told', 'asked',
    'time', 'way', 'thing', 'things', 'fact', 'point',
    'want', 'need', 'use', 'used',
])

print(f"  {len(df)} documents loaded")

# ============================================================
# PART 1: Word Co-occurrence Network
# ============================================================
print("\n" + "=" * 60)
print("PART 1: Word Co-occurrence Network")
print("=" * 60)

# Tokenize
print("\nTokenizing...")
doc_tokens = []
for text in df['text']:
    tokens = simple_preprocess(text, deacc=True, min_len=3)
    tokens = [t for t in tokens if t not in stop_words]
    doc_tokens.append(tokens)

# Word frequency
word_freq = Counter()
for tokens in doc_tokens:
    word_freq.update(tokens)

# Keep top words by frequency (must appear in at least 3 docs)
doc_freq = Counter()
for tokens in doc_tokens:
    doc_freq.update(set(tokens))

MIN_DOC_FREQ = 3
MIN_WORD_FREQ = 8
TOP_WORDS = 400

candidates = {w for w, f in word_freq.items()
              if f >= MIN_WORD_FREQ and doc_freq[w] >= MIN_DOC_FREQ}

# Sort by frequency and take top N
top_words = sorted(candidates, key=lambda w: -word_freq[w])[:TOP_WORDS]
top_word_set = set(top_words)
print(f"  Top {len(top_words)} words selected (freq >= {MIN_WORD_FREQ}, doc_freq >= {MIN_DOC_FREQ})")

# Compute PMI (Pointwise Mutual Information) for word pairs
print("Computing PMI co-occurrences...")
WINDOW = 10
total_windows = 0
pair_count = Counter()
word_window_count = Counter()

for tokens in doc_tokens:
    for i in range(len(tokens)):
        w1 = tokens[i]
        if w1 not in top_word_set:
            continue
        window_end = min(i + WINDOW, len(tokens))
        window_words = set()
        for j in range(i + 1, window_end):
            w2 = tokens[j]
            if w2 in top_word_set and w2 != w1:
                window_words.add(w2)
        for w2 in window_words:
            pair = tuple(sorted([w1, w2]))
            pair_count[pair] += 1
        if window_words:
            word_window_count[w1] += 1
            total_windows += 1

# Compute PMI
total_pairs = sum(pair_count.values())
pmi_scores = {}
for (w1, w2), count in pair_count.items():
    if count < 3:
        continue
    p_pair = count / total_pairs
    p_w1 = word_window_count[w1] / total_pairs
    p_w2 = word_window_count[w2] / total_pairs
    if p_w1 > 0 and p_w2 > 0:
        pmi = math.log2(p_pair / (p_w1 * p_w2))
        if pmi > 0:
            # NPMI normalisation
            npmi = pmi / (-math.log2(p_pair)) if p_pair > 0 else 0
            pmi_scores[(w1, w2)] = npmi

print(f"  {len(pmi_scores):,} word pairs with positive PMI")

# Build network — keep top edges by NPMI
TOP_EDGES = 600
top_pairs = sorted(pmi_scores.items(), key=lambda x: -x[1])[:TOP_EDGES]

G_word = nx.Graph()
for (w1, w2), npmi in top_pairs:
    G_word.add_edge(w1, w2, weight=npmi)

# Add isolated top words that didn't make any edge
for w in top_words[:100]:
    if w not in G_word:
        G_word.add_node(w)

print(f"  Network: {G_word.number_of_nodes()} nodes, {G_word.number_of_edges()} edges")

# Community detection
from networkx.algorithms.community import louvain_communities
communities = louvain_communities(G_word, resolution=1.0, seed=42)
community_map = {}
for i, comm in enumerate(communities):
    for node in comm:
        community_map[node] = i

print(f"  Communities found: {len(communities)}")
for i, comm in enumerate(sorted(communities, key=len, reverse=True)):
    top_in_comm = sorted(comm, key=lambda w: -word_freq.get(w, 0))[:6]
    print(f"    Community {i} ({len(comm)} words): {', '.join(top_in_comm)}")

# Layout — spring layout in 3D
print("  Computing 3D layout...")
pos = nx.spring_layout(G_word, dim=3, seed=42, k=0.8, iterations=100,
                        weight='weight')

# Build Plotly figure
print("  Building word network plot...")

# Assign community colours
n_communities = len(communities)
import plotly.colors as pc
palette = pc.qualitative.Set3 + pc.qualitative.Pastel + pc.qualitative.Dark2
if n_communities > len(palette):
    palette = palette * (n_communities // len(palette) + 1)

community_names = {}
for i, comm in enumerate(sorted(communities, key=len, reverse=True)):
    top_3 = sorted(comm, key=lambda w: -word_freq.get(w, 0))[:3]
    community_names[i] = ' / '.join(top_3)

# Remap community indices to sorted order
sorted_comms = sorted(range(len(communities)),
                       key=lambda i: -len(communities[i]))
comm_remap = {old: new for new, old in enumerate(sorted_comms)}
for node in community_map:
    community_map[node] = comm_remap.get(community_map[node], community_map[node])

# Rebuild community_names with remapped indices
communities_sorted = sorted(communities, key=len, reverse=True)
community_names = {}
for i, comm in enumerate(communities_sorted):
    top_3 = sorted(comm, key=lambda w: -word_freq.get(w, 0))[:3]
    community_names[i] = ' / '.join(top_3)

fig_word = go.Figure()

# Draw edges
edge_x, edge_y, edge_z = [], [], []
for u, v in G_word.edges():
    x0, y0, z0 = pos[u]
    x1, y1, z1 = pos[v]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]
    edge_z += [z0, z1, None]

fig_word.add_trace(go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    mode='lines',
    line=dict(color='rgba(80,80,80,0.4)', width=1),
    hoverinfo='none',
    showlegend=False,
))

# Draw nodes by community
for comm_id in range(min(n_communities, 20)):
    comm_nodes = [n for n in G_word.nodes() if community_map.get(n) == comm_id]
    if not comm_nodes:
        continue

    xs = [pos[n][0] for n in comm_nodes]
    ys = [pos[n][1] for n in comm_nodes]
    zs = [pos[n][2] for n in comm_nodes]
    freqs_c = [word_freq.get(n, 1) for n in comm_nodes]
    log_f = np.log1p(freqs_c)
    sizes = 4 + (log_f - log_f.min()) / (log_f.max() - log_f.min() + 1e-9) * 14

    fig_word.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers+text',
        marker=dict(size=sizes, color=palette[comm_id % len(palette)], opacity=0.8),
        text=comm_nodes,
        textposition='top center',
        textfont=dict(size=[max(7, int(s * 0.6)) for s in sizes],
                      color=palette[comm_id % len(palette)]),
        name=community_names.get(comm_id, f'Community {comm_id}'),
        hovertemplate='<b>%{text}</b><br>Freq: %{customdata[0]}<extra></extra>',
        customdata=[[f] for f in freqs_c],
    ))

fig_word.update_layout(
    title='Word Co-occurrence Network — Concept Clusters',
    width=1400, height=900,
    scene=dict(
        xaxis=dict(showticklabels=False, title=''),
        yaxis=dict(showticklabels=False, title=''),
        zaxis=dict(showticklabels=False, title=''),
    ),
    legend=dict(font=dict(size=10), itemsizing='constant'),
    margin=dict(l=0, r=0, b=0, t=40),
    annotations=[
        dict(
            text=(
                '<b>How to read this plot</b><br>'
                'Each dot is a word. Lines connect words that frequently appear near each other in the archive.<br>'
                'Colours show automatically detected communities — groups of words that form tight conceptual clusters.<br>'
                'Larger dots are more frequent words. Look for bridges between clusters to find cross-cutting themes.'
            ),
            showarrow=False, xref='paper', yref='paper',
            x=0.5, y=-0.02, xanchor='center', yanchor='top',
            font=dict(size=11, color='#555'), align='center',
        ),
    ],
)

html_word = os.path.join(OUTPUT_DIR, 'corpus_word_network.html')
fig_word.write_html(html_word, include_plotlyjs=True)
print(f"  Saved: {html_word}")


# ============================================================
# PART 2: Document Similarity Network
# ============================================================
print("\n" + "=" * 60)
print("PART 2: Document Similarity Network")
print("=" * 60)

# TF-IDF with bigrams
print("\nBuilding TF-IDF vectors...")
tfidf = TfidfVectorizer(
    max_features=5000, min_df=2, max_df=0.85,
    stop_words='english', ngram_range=(1, 2),
    token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b',
)
tfidf_matrix = tfidf.fit_transform(df['text'])

# Compute pairwise cosine similarity
print("Computing document similarities...")
sim_matrix = cosine_similarity(tfidf_matrix)

# Zero out self-similarity
np.fill_diagonal(sim_matrix, 0)

# For each document, find its distinctive terms (top TF-IDF features)
feature_names = tfidf.get_feature_names_out()

# Build network — connect docs above a threshold
# Use adaptive threshold: keep top K neighbours per doc
K_NEIGHBOURS = 5
SIMILARITY_FLOOR = 0.15  # minimum similarity to form an edge

G_doc = nx.Graph()
for i in range(len(df)):
    G_doc.add_node(i)

edge_count = 0
cross_archive_edges = 0
for i in range(len(df)):
    top_k = np.argsort(sim_matrix[i])[-K_NEIGHBOURS:]
    for j in top_k:
        if j != i and sim_matrix[i][j] >= SIMILARITY_FLOOR:
            if not G_doc.has_edge(i, j):
                G_doc.add_edge(i, j, weight=float(sim_matrix[i][j]))
                edge_count += 1
                if df.iloc[i]['source_file'] != df.iloc[j]['source_file']:
                    cross_archive_edges += 1

print(f"  Network: {G_doc.number_of_nodes()} nodes, {G_doc.number_of_edges()} edges")
print(f"  Cross-archive edges: {cross_archive_edges} ({cross_archive_edges/max(edge_count,1)*100:.0f}%)")

# Community detection on document network
doc_communities = louvain_communities(G_doc, resolution=0.8, seed=42)
doc_comm_map = {}
for i, comm in enumerate(doc_communities):
    for node in comm:
        doc_comm_map[node] = i

print(f"  Document communities: {len(doc_communities)}")

# Name each community by its most distinctive terms
doc_comm_names = {}
for i, comm in enumerate(sorted(doc_communities, key=len, reverse=True)):
    comm_indices = list(comm)
    # Average TF-IDF for this community
    comm_tfidf = tfidf_matrix[comm_indices].mean(axis=0).A1
    top_terms = np.argsort(comm_tfidf)[-5:][::-1]
    terms = [feature_names[t] for t in top_terms]
    doc_comm_names[i] = ' / '.join(terms[:4])
    archives = df.iloc[comm_indices]['source_file'].value_counts().head(2)
    archive_str = ', '.join(archives.index)
    print(f"    Comm {i} ({len(comm)} docs): {doc_comm_names[i]}  [{archive_str}]")

# Remap
sorted_doc_comms = sorted(range(len(doc_communities)), key=lambda i: -len(doc_communities[i]))
doc_comm_remap = {old: new for new, old in enumerate(sorted_doc_comms)}
for node in doc_comm_map:
    doc_comm_map[node] = doc_comm_remap.get(doc_comm_map[node], doc_comm_map[node])

doc_communities_sorted = sorted(doc_communities, key=len, reverse=True)
doc_comm_names_sorted = {}
for i, comm in enumerate(doc_communities_sorted):
    comm_indices = list(comm)
    comm_tfidf = tfidf_matrix[comm_indices].mean(axis=0).A1
    top_terms = np.argsort(comm_tfidf)[-4:][::-1]
    terms = [feature_names[t] for t in top_terms]
    doc_comm_names_sorted[i] = ' / '.join(terms)

# 3D layout
print("  Computing 3D layout...")
pos_doc = nx.spring_layout(G_doc, dim=3, seed=42, k=1.5, iterations=80, weight='weight')

# Build Plotly figure
print("  Building document network plot...")

fig_doc = go.Figure()

# Draw edges — highlight cross-archive edges
edge_same_x, edge_same_y, edge_same_z = [], [], []
edge_cross_x, edge_cross_y, edge_cross_z = [], [], []

for u, v in G_doc.edges():
    x0, y0, z0 = pos_doc[u]
    x1, y1, z1 = pos_doc[v]
    if df.iloc[u]['source_file'] != df.iloc[v]['source_file']:
        edge_cross_x += [x0, x1, None]
        edge_cross_y += [y0, y1, None]
        edge_cross_z += [z0, z1, None]
    else:
        edge_same_x += [x0, x1, None]
        edge_same_y += [y0, y1, None]
        edge_same_z += [z0, z1, None]

fig_doc.add_trace(go.Scatter3d(
    x=edge_same_x, y=edge_same_y, z=edge_same_z,
    mode='lines', line=dict(color='rgba(60,60,60,0.25)', width=0.8),
    hoverinfo='none', showlegend=False,
))

fig_doc.add_trace(go.Scatter3d(
    x=edge_cross_x, y=edge_cross_y, z=edge_cross_z,
    mode='lines', line=dict(color='rgba(255,80,30,0.5)', width=1.5),
    hoverinfo='none', name='Cross-archive links',
))

# Draw nodes by community
df['text_preview'] = df['text'].apply(
    lambda t: t[:80].replace('\n', ' ').replace('<', '&lt;').strip() + '...'
)

n_doc_communities = len(doc_communities_sorted)
for comm_id in range(min(n_doc_communities, 25)):
    comm_nodes = [n for n in G_doc.nodes() if doc_comm_map.get(n) == comm_id]
    if not comm_nodes:
        continue

    xs = [pos_doc[n][0] for n in comm_nodes]
    ys = [pos_doc[n][1] for n in comm_nodes]
    zs = [pos_doc[n][2] for n in comm_nodes]

    sub = df.iloc[comm_nodes]

    fig_doc.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers',
        marker=dict(size=4, color=palette[comm_id % len(palette)], opacity=0.8),
        name=doc_comm_names_sorted.get(comm_id, f'Community {comm_id}'),
        hovertemplate=(
            '<b>%{customdata[0]}</b> | %{customdata[1]}<br>'
            '%{customdata[2]}<br>'
            '%{customdata[3]}'
            '<extra></extra>'
        ),
        customdata=sub[['doc_id', 'archive_folder', 'topic', 'text_preview']].values,
    ))

fig_doc.update_layout(
    title='Document Similarity Network — Cross-Archive Connections',
    width=1400, height=900,
    scene=dict(
        xaxis=dict(showticklabels=False, title=''),
        yaxis=dict(showticklabels=False, title=''),
        zaxis=dict(showticklabels=False, title=''),
    ),
    legend=dict(font=dict(size=9), itemsizing='constant'),
    margin=dict(l=0, r=0, b=0, t=40),
    annotations=[
        dict(
            text=(
                '<b>How to read this plot</b><br>'
                'Each dot is a document. Lines connect documents that share similar vocabulary.<br>'
                '<span style="color:rgba(255,100,50,0.9)">Orange lines</span> link documents '
                'from <i>different</i> archive folders — these are cross-archive connections.<br>'
                'Colours show document communities detected by the network structure. '
                'Hover over a dot to see its ID, archive, and topic.'
            ),
            showarrow=False, xref='paper', yref='paper',
            x=0.5, y=-0.02, xanchor='center', yanchor='top',
            font=dict(size=11, color='#555'), align='center',
        ),
    ],
)

html_doc = os.path.join(OUTPUT_DIR, 'corpus_doc_network.html')
fig_doc.write_html(html_doc, include_plotlyjs=True)
print(f"  Saved: {html_doc}")

# ============================================================
# Summary: Cross-archive bridges
# ============================================================
print("\n" + "=" * 60)
print("Cross-Archive Bridges (strongest connections between different folders)")
print("=" * 60)

bridges = []
for u, v, data in G_doc.edges(data=True):
    if df.iloc[u]['source_file'] != df.iloc[v]['source_file']:
        # Find shared distinctive terms
        u_vec = tfidf_matrix.getrow(u).toarray().flatten()
        v_vec = tfidf_matrix.getrow(v).toarray().flatten()
        shared = u_vec * v_vec  # element-wise product
        top_shared = np.argsort(shared)[-5:][::-1]
        shared_terms = [feature_names[t] for t in top_shared if shared[t] > 0]

        bridges.append({
            'doc_a': df.iloc[u]['doc_id'],
            'archive_a': df.iloc[u]['source_file'].replace('_cleaned.txt', ''),
            'doc_b': df.iloc[v]['doc_id'],
            'archive_b': df.iloc[v]['source_file'].replace('_cleaned.txt', ''),
            'similarity': data['weight'],
            'shared_terms': ', '.join(shared_terms[:4]),
        })

bridges_df = pd.DataFrame(bridges).sort_values('similarity', ascending=False)
bridges_df.to_csv(os.path.join(OUTPUT_DIR, 'cross_archive_bridges.csv'), index=False)

print(f"\n  Total cross-archive bridges: {len(bridges_df)}")
print(f"\n  Top 25 strongest:")
for _, row in bridges_df.head(25).iterrows():
    print(f"    {row['similarity']:.3f}  {row['archive_a']:20s} ↔ {row['archive_b']:20s}  [{row['shared_terms']}]")

print(f"\n  Saved: {os.path.join(OUTPUT_DIR, 'cross_archive_bridges.csv')}")

# ============================================================
# PART 3: Combined HTML page
# ============================================================
print("\n" + "=" * 60)
print("PART 3: Building combined page")
print("=" * 60)

# Remove the per-plot annotations (we'll put explanations in HTML instead)
fig_word.update_layout(annotations=[], title='', margin=dict(l=0, r=0, b=0, t=10))
fig_doc.update_layout(annotations=[], title='', margin=dict(l=0, r=0, b=0, t=10))

# Export each plot as an HTML div (no full page wrapper)
word_div = fig_word.to_html(full_html=False, include_plotlyjs=False)
doc_div = fig_doc.to_html(full_html=False, include_plotlyjs=False)

combined_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Tower Blocks Corpus — Semantic Networks</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
    background: #fafafa;
    color: #333;
  }}
  .page-header {{
    background: #2c3e50;
    color: white;
    padding: 28px 40px;
  }}
  .page-header h1 {{
    font-size: 26px;
    font-weight: 600;
    margin-bottom: 6px;
  }}
  .page-header p {{
    font-size: 15px;
    color: #bdc3c7;
    max-width: 900px;
    line-height: 1.5;
  }}
  .section {{
    max-width: 1500px;
    margin: 0 auto;
    padding: 30px 40px;
  }}
  .section h2 {{
    font-size: 22px;
    font-weight: 600;
    color: #2c3e50;
    margin-bottom: 12px;
    border-bottom: 2px solid #3498db;
    padding-bottom: 6px;
  }}
  .explainer {{
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 20px;
    line-height: 1.65;
    font-size: 15px;
    color: #444;
  }}
  .explainer strong {{
    color: #2c3e50;
  }}
  .explainer .highlight {{
    background: #fef9e7;
    border-left: 3px solid #f39c12;
    padding: 10px 14px;
    margin: 12px 0;
    font-size: 14px;
  }}
  .plot-container {{
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 30px;
  }}
  .divider {{
    height: 1px;
    background: #e0e0e0;
    margin: 10px 40px;
  }}
</style>
</head>
<body>

<div class="page-header">
  <h1>Tower Blocks Corpus &mdash; Semantic Networks</h1>
  <p>
    Interactive visualisations of the language and document structure within the
    Tower Blocks archive. These plots reveal how concepts relate to each other
    and which documents across different archive folders share common ground.
  </p>
</div>

<div class="section">
  <h2>1. Concept Network</h2>
  <div class="explainer">
    <p>
      This plot maps the <strong>vocabulary</strong> of the archive. Each dot is a word.
      Words are connected by a line if they regularly appear near each other in the documents &mdash;
      for example, &ldquo;Taylor&rdquo; and &ldquo;Woodrow&rdquo; almost always appear together,
      so they are tightly linked.
    </p>
    <p>
      The colours group words into <strong>communities</strong> &mdash; clusters of words that
      form a conceptual unit. For instance, you will see a cluster of structural engineering terms
      (wall, panel, joints, concrete), another of people involved in the inquiry
      (Griffiths, Goodfellow, Eveleigh), and another of tenant advocacy language
      (residents, association, campaign).
    </p>
    <div class="highlight">
      <strong>What to look for:</strong> Words that sit between two clusters act as
      <em>conceptual bridges</em> &mdash; they connect different themes in the archive.
      The larger a word appears, the more frequently it is used across the corpus.
    </div>
    <p>
      <strong>How to interact:</strong> Click and drag to rotate the plot. Scroll to zoom.
      Hover over a word to see how often it appears. Click a community name in the legend
      to show or hide it.
    </p>
  </div>
  <div class="plot-container">
    {word_div}
  </div>
</div>

<div class="divider"></div>

<div class="section">
  <h2>2. Document Network</h2>
  <div class="explainer">
    <p>
      This plot maps the <strong>documents</strong> in the archive. Each dot is a single document
      (a letter, report, transcript, drawing, etc.). Documents are connected by a line if they
      use similar language &mdash; the more vocabulary they share, the closer they appear.
    </p>
    <p>
      <strong style="color: #e74c3c;">Orange lines</strong> are particularly important: they
      connect documents from <em>different</em> archive folders. These cross-archive links
      reveal where the same people, buildings, events, or issues appear in separate parts of
      the collection. Grey lines connect documents within the same folder.
    </p>
    <div class="highlight">
      <strong>What to look for:</strong> Documents at the edges of a cluster, connected by orange
      lines to another cluster, are the key bridging documents &mdash; they link different parts
      of the archive together. Isolated dots with no connections contain unique material not
      found elsewhere in the collection.
    </div>
    <p>
      <strong>How to interact:</strong> Hover over a dot to see the document ID, which archive
      folder it belongs to, its topic, and a preview of its content. Click and drag to rotate;
      scroll to zoom.
    </p>
  </div>
  <div class="plot-container">
    {doc_div}
  </div>
</div>

</body>
</html>"""

combined_path = os.path.join(OUTPUT_DIR, 'corpus_semantic_networks.html')
with open(combined_path, 'w', encoding='utf-8') as f:
    f.write(combined_html)

print(f"  Saved combined page: {combined_path}")
print(f"\nDone!")
