"""
Corpus VAD Explorer: Score documents on Valence, Arousal, Dominance
and display as an interactive 3D Plotly scatterplot.
"""
import os
import re
import pandas as pd
import numpy as np
import plotly.express as px

OUTPUT_DIR = '/Users/fhsr/Desktop/WESA_13_1/text_data'

# ============================================================
# STEP 1: Load corpus and word norms
# ============================================================
print("Step 1: Loading data...")

df = pd.read_pickle(os.path.join(OUTPUT_DIR, 'corpus_documents.pkl'))
norms = pd.read_pickle('Tower Blocks Corpus Cleaned/all_norm_estimates.pkl')

# Build a fast lookup dict for V, A, D
norms_vad = norms[['valence', 'arousal', 'dominance']]
vad_dict = {
    word: (row['valence'], row['arousal'], row['dominance'])
    for word, row in norms_vad.iterrows()
}

print(f"  Documents: {len(df)}")
print(f"  Norm words: {len(vad_dict):,}")

# ============================================================
# STEP 2: Score each document on V, A, D
# ============================================================
print("\nStep 2: Scoring documents...")

def score_document(text, vad_lookup):
    """Compute mean V, A, D for a document by averaging word-level scores."""
    # Simple tokenisation: lowercase, letters only
    tokens = re.findall(r'\b[a-z]+\b', text.lower())

    v_scores = []
    a_scores = []
    d_scores = []
    matched = 0

    for token in tokens:
        if token in vad_lookup:
            v, a, d = vad_lookup[token]
            v_scores.append(v)
            a_scores.append(a)
            d_scores.append(d)
            matched += 1

    if matched == 0:
        return np.nan, np.nan, np.nan, 0, 0

    return (
        np.mean(v_scores),
        np.mean(a_scores),
        np.mean(d_scores),
        matched,
        matched / len(tokens) if tokens else 0,
    )

results = df['text'].apply(lambda t: score_document(t, vad_dict))
df['valence'] = results.apply(lambda r: r[0])
df['arousal'] = results.apply(lambda r: r[1])
df['dominance'] = results.apply(lambda r: r[2])
df['vad_matched_words'] = results.apply(lambda r: r[3])
df['vad_coverage'] = results.apply(lambda r: r[4])

# Drop docs with no VAD scores
n_before = len(df)
df = df.dropna(subset=['valence']).reset_index(drop=True)
print(f"  Scored {len(df)} documents ({n_before - len(df)} dropped for no matches)")
print(f"  Mean coverage: {df['vad_coverage'].mean():.1%} of words matched")
print(f"\n  VAD ranges:")
print(f"    Valence:   {df['valence'].min():.3f} – {df['valence'].max():.3f} (mean {df['valence'].mean():.3f})")
print(f"    Arousal:   {df['arousal'].min():.3f} – {df['arousal'].max():.3f} (mean {df['arousal'].mean():.3f})")
print(f"    Dominance: {df['dominance'].min():.3f} – {df['dominance'].max():.3f} (mean {df['dominance'].mean():.3f})")

# ============================================================
# STEP 3: Build 3D Plotly scatter on V, A, D axes
# ============================================================
print("\nStep 3: Building VAD 3D plot...")

# Truncated hover
df['text_preview'] = df['text'].apply(
    lambda t: t[:80].replace('\n', ' ').replace('<', '&lt;').strip() + '...'
)

# Merge topic labels from the previous analysis
if 'topic' not in df.columns or df['topic'].isna().all():
    topics_csv = os.path.join(OUTPUT_DIR, 'corpus_topics.csv')
    if os.path.exists(topics_csv):
        topics_df = pd.read_csv(topics_csv)
        topic_map = dict(zip(topics_df['doc_id'], topics_df['topic']))
        df['topic'] = df['doc_id'].map(topic_map).fillna('unassigned')
    else:
        df['topic'] = 'unassigned'

fig = px.scatter_3d(
    df,
    x='valence', y='arousal', z='dominance',
    color='topic',
    hover_name='doc_id',
    custom_data=['archive_folder', 'date', 'topic', 'text_preview'],
    title='Tower Blocks Corpus — Emotional Profile (Valence–Arousal–Dominance)',
    labels={
        'valence': 'Valence (negative ← → positive)',
        'arousal': 'Arousal (calm ← → excited)',
        'dominance': 'Dominance (submissive ← → dominant)',
    },
    width=1200,
    height=800,
)

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
        xaxis_title='Valence (negative ← → positive)',
        yaxis_title='Arousal (calm ← → excited)',
        zaxis_title='Dominance (submissive ← → dominant)',
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    annotations=[
        dict(
            text=(
                '<b>How to read this plot</b><br>'
                'Each dot is a document scored for emotional tone using word-level norms.<br>'
                '<b>Valence</b>: how positive or negative the language is. '
                '<b>Arousal</b>: how emotionally intense or calm. '
                '<b>Dominance</b>: how authoritative or submissive the tone.<br>'
                'Colours show the same topics as the main corpus explorer. '
                'Hover over a dot to see its details.'
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

html_path = os.path.join(OUTPUT_DIR, 'corpus_vad_explorer.html')
fig.write_html(html_path, include_plotlyjs=True)

# ============================================================
# STEP 4: Summary stats by topic
# ============================================================
print("\n=== Emotional profile by topic ===")
topic_stats = df.groupby('topic')[['valence', 'arousal', 'dominance']].agg(['mean', 'std', 'count'])
topic_stats.columns = ['_'.join(c) for c in topic_stats.columns]
topic_stats = topic_stats.sort_values('valence_mean')

for topic, row in topic_stats.iterrows():
    n = int(row['valence_count'])
    v = row['valence_mean']
    a = row['arousal_mean']
    d = row['dominance_mean']
    print(f"  {topic:45s}  V={v:.3f}  A={a:.3f}  D={d:.3f}  (n={n})")

# Save
df.to_pickle(os.path.join(OUTPUT_DIR, 'corpus_documents_vad.pkl'))

print(f"\nDone!")
print(f"  {html_path}")
print(f"  {os.path.join(OUTPUT_DIR, 'corpus_documents_vad.pkl')}")
