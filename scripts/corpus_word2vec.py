"""
Word2Vec model trained on the Tower Blocks Corpus.
Extracts word embeddings, reduces to 3D via PCA, and displays
as an interactive Plotly scatterplot to reveal concept co-occurrence.
"""
import os
import re
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.graph_objects as go

OUTPUT_DIR = '/Users/fhsr/Desktop/WESA_13_1/text_data'

# ============================================================
# STEP 1: Load and tokenize corpus
# ============================================================
print("Step 1: Loading and tokenizing corpus...")

df = pd.read_pickle(os.path.join(OUTPUT_DIR, 'corpus_documents.pkl'))

stop_words = set(TfidfVectorizer(stop_words='english').get_stop_words())
stop_words.update([
    'cont', 'continued', 'page', 'mr', 'mrs', 'dr', 'sir',
    'said', 'would', 'shall', 'also', 'yes', 'think',
    'know', 'say', 'right', 'see', 'come', 'got', 'get',
    'let', 'make', 'take', 'like', 'just', 'well', 'quite',
    'ee', 'ae', 'oe', 'se', 'ne', 'wesa', 'img',
])

sentences = []
for text in df['text']:
    # Split into sentences on period/newline boundaries
    raw_sents = re.split(r'[.\n]+', text)
    for sent in raw_sents:
        tokens = simple_preprocess(sent, deacc=True, min_len=3)
        tokens = [t for t in tokens if t not in stop_words]
        if len(tokens) >= 3:
            sentences.append(tokens)

print(f"  {len(df)} documents → {len(sentences):,} sentences")
print(f"  Total tokens: {sum(len(s) for s in sentences):,}")

# ============================================================
# STEP 2: Train Word2Vec
# ============================================================
print("\nStep 2: Training Word2Vec...")

model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=8,
    min_count=5,
    sg=1,           # skip-gram (better for semantic relations)
    epochs=30,
    workers=4,
    seed=42,
)

vocab = model.wv
print(f"  Vocabulary: {len(vocab)} words")
print(f"  Vector size: {vocab.vector_size}")

# Quick sanity check
print("\n  Nearest neighbours:")
for probe in ['ronan', 'fire', 'tenants', 'collapse', 'cladding', 'safety']:
    if probe in vocab:
        neighbours = vocab.most_similar(probe, topn=5)
        nn = ', '.join(f"{w} ({s:.2f})" for w, s in neighbours)
        print(f"    {probe}: {nn}")

# ============================================================
# STEP 3: PCA to 3D
# ============================================================
print("\nStep 3: Reducing to 3D via PCA...")

words = list(vocab.index_to_key)
vectors = np.array([vocab[w] for w in words])

pca = PCA(n_components=3, random_state=42)
coords = pca.fit_transform(vectors)

print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.1%}")
print(f"    PC1: {pca.explained_variance_ratio_[0]:.1%}")
print(f"    PC2: {pca.explained_variance_ratio_[1]:.1%}")
print(f"    PC3: {pca.explained_variance_ratio_[2]:.1%}")

# ============================================================
# STEP 4: Compute word frequency for sizing/filtering
# ============================================================
print("\nStep 4: Computing word frequencies...")

from collections import Counter
word_freq = Counter()
for sent in sentences:
    word_freq.update(sent)

freqs = np.array([word_freq[w] for w in words])
log_freqs = np.log1p(freqs)

# ============================================================
# STEP 5: Categorise words by domain for colouring
# ============================================================
print("\nStep 5: Categorising words...")

# Domain keyword lists — words containing these substrings get tagged
domain_rules = {
    'Buildings & Places': [
        'ronan', 'point', 'ledbury', 'tower', 'block', 'blocks', 'building',
        'buildings', 'flat', 'flats', 'floor', 'storey', 'storeys',
        'estate', 'house', 'bromyard', 'hartopp', 'lannoy', 'grenfell',
        'newham', 'southwark', 'hammersmith', 'fulham', 'wandsworth',
        'lambeth', 'london', 'canning', 'woolwich', 'stratford',
    ],
    'Structure & Engineering': [
        'wall', 'walls', 'panel', 'panels', 'concrete', 'steel',
        'joint', 'joints', 'bolt', 'bolts', 'strap', 'straps',
        'slab', 'slabs', 'load', 'bearing', 'structural', 'structure',
        'collapse', 'cladding', 'insulation', 'reinforcement',
        'construction', 'precast', 'mortar', 'cement', 'crack',
        'flank', 'pressure', 'stress', 'failure', 'strength',
        'design', 'engineer', 'engineering', 'architects', 'architect',
    ],
    'Fire & Safety': [
        'fire', 'safety', 'smoke', 'alarm', 'sprinkler', 'evacuation',
        'combustible', 'flammable', 'risk', 'hazard', 'brigade',
        'compartment', 'compartmentalisation', 'escape', 'emergency',
        'explosion', 'gas', 'cooker', 'heating',
    ],
    'People & Organisations': [
        'tenant', 'tenants', 'resident', 'residents', 'council',
        'councillor', 'minister', 'government', 'authority',
        'committee', 'association', 'campaign', 'borough',
        'griffiths', 'goodfellow', 'eveleigh', 'pugsley', 'watson',
        'webb', 'littlewood', 'fairweather', 'hodge', 'pike',
    ],
    'Legal & Inquiry': [
        'inquiry', 'tribunal', 'evidence', 'witness', 'statement',
        'report', 'regulation', 'regulations', 'compliance',
        'inspection', 'certificate', 'assessment', 'prosecution',
        'solicitor', 'counsel', 'barrister', 'cross', 'examination',
    ],
    'Theatre & Culture': [
        'play', 'theatre', 'song', 'act', 'scene', 'projector',
        'audience', 'stage', 'rehearsal', 'production', 'joan',
        'jerry', 'performers', 'drama', 'comedy', 'music',
    ],
}

word_categories = {}
for w in words:
    assigned = False
    for category, keywords in domain_rules.items():
        if w in keywords:
            word_categories[w] = category
            assigned = True
            break
    if not assigned:
        word_categories[w] = 'Other'

categories = [word_categories[w] for w in words]

# ============================================================
# STEP 6: Build interactive 3D Plotly plot
# ============================================================
print("\nStep 6: Building plot...")

# Filter to top N most frequent words to avoid clutter
TOP_N = 800
top_indices = np.argsort(freqs)[-TOP_N:]

plot_words = [words[i] for i in top_indices]
plot_coords = coords[top_indices]
plot_cats = [categories[i] for i in top_indices]
plot_freqs = freqs[top_indices]
plot_log_freqs = log_freqs[top_indices]

# Normalise sizes
size_min, size_max = 3, 18
norm_sizes = (plot_log_freqs - plot_log_freqs.min()) / (plot_log_freqs.max() - plot_log_freqs.min())
plot_sizes = size_min + norm_sizes * (size_max - size_min)

plot_df = pd.DataFrame({
    'word': plot_words,
    'PC1': plot_coords[:, 0],
    'PC2': plot_coords[:, 1],
    'PC3': plot_coords[:, 2],
    'category': plot_cats,
    'frequency': plot_freqs,
    'size': plot_sizes,
})

# Colour map
colour_map = {
    'Buildings & Places': '#e41a1c',
    'Structure & Engineering': '#377eb8',
    'Fire & Safety': '#ff7f00',
    'People & Organisations': '#4daf4a',
    'Legal & Inquiry': '#984ea3',
    'Theatre & Culture': '#f781bf',
    'Other': '#cccccc',
}

fig = go.Figure()

for cat in ['Other', 'Buildings & Places', 'Structure & Engineering',
            'Fire & Safety', 'People & Organisations', 'Legal & Inquiry',
            'Theatre & Culture']:
    mask = plot_df['category'] == cat
    sub = plot_df[mask]
    if len(sub) == 0:
        continue

    fig.add_trace(go.Scatter3d(
        x=sub['PC1'], y=sub['PC2'], z=sub['PC3'],
        mode='markers+text',
        marker=dict(
            size=sub['size'],
            color=colour_map[cat],
            opacity=0.4 if cat == 'Other' else 0.8,
            line=dict(width=0),
        ),
        text=sub['word'],
        textposition='top center',
        textfont=dict(
            size=sub['size'].apply(lambda s: max(7, int(s * 0.7))),
            color=colour_map[cat],
        ),
        name=cat,
        hovertemplate='<b>%{text}</b><br>Frequency: %{customdata[0]}<br>Category: ' + cat + '<extra></extra>',
        customdata=sub[['frequency']].values,
    ))

fig.update_layout(
    title='Tower Blocks Corpus — Word Embedding Space (Word2Vec)',
    width=1400,
    height=900,
    scene=dict(
        xaxis_title='Dimension 1',
        yaxis_title='Dimension 2',
        zaxis_title='Dimension 3',
    ),
    legend=dict(
        font=dict(size=11),
        itemsizing='constant',
        yanchor='top', y=0.99,
        xanchor='left', x=0.01,
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    annotations=[
        dict(
            text=(
                '<b>How to read this plot</b><br>'
                'Each dot is a word from the corpus. Words that frequently appear in similar contexts are positioned near each other.<br>'
                'Size reflects how often the word appears. Colours indicate domain categories.<br>'
                'Clusters of nearby words reveal concepts that co-occur in the archive — '
                'for example, structural engineering terms cluster together, as do names of people involved in the inquiry.'
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

html_path = os.path.join(OUTPUT_DIR, 'corpus_word2vec_explorer.html')
fig.write_html(html_path, include_plotlyjs=True)

# Save model and data
model.save(os.path.join(OUTPUT_DIR, 'corpus_word2vec.model'))
plot_df.to_csv(os.path.join(OUTPUT_DIR, 'corpus_word2vec_coords.csv'), index=False)

print(f"\nDone!")
print(f"  Vocabulary: {len(vocab)} words, top {TOP_N} plotted")
print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.1%}")
print(f"  {html_path}")
