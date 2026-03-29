# Tower Blocks Corpus Explorer

Interactive visualisations of the **Tower Blocks UK archive** — a collection of 812 documents spanning the Ronan Point disaster (1968), the Grenfell Tower fire (2017), the Ledbury Estate crisis, and decades of building safety campaigning in between.

**Live site:** [GitHub Pages link — enable in repo settings]

## What this is

The archive contains letters, inquiry transcripts, council reports, engineering assessments, tenant campaign materials, and a play script (*The Projector* by Joan Littlewood / John Wells). This project applies natural language processing to reveal:

1. **Concept clusters** — which words and ideas co-occur across the archive
2. **Cross-archive connections** — which documents in different folders discuss the same people, buildings, or events

The main visualisation (`index.html`) presents two interactive 3D network plots built with [Plotly.js](https://plotly.com/javascript/).

## Repository structure

```
├── index.html                  # Main page (GitHub Pages entry point)
├── data/
│   ├── documents.json          # 812 documents with metadata and topic assignments
│   ├── topics.json             # 12 LDA topics with keywords
│   ├── cross_archive_bridges.json  # 1,081 cross-folder document connections
│   └── word_embeddings.json    # 800 words with 3D PCA coordinates and categories
└── scripts/
    ├── corpus_explorer.py      # TF-IDF + PCA + LDA topic modelling
    ├── corpus_networks.py      # Word co-occurrence + document similarity networks
    ├── corpus_word2vec.py      # Word2Vec training and embedding visualisation
    ├── corpus_vad_explorer.py  # Valence-Arousal-Dominance emotional scoring
    └── corpus_bertopic.py      # BERTopic comparison analysis
```

## Data files

All data is in JSON format (array of objects).

### `documents.json`

Each record is one document from the corpus.

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | string | Unique document identifier (e.g. `TBUK1001`, `WESA_13_1_025`) |
| `archive_folder` | string | Source archive collection name |
| `source_file` | string | Which cleaned corpus file this came from |
| `date` | string | Document date if known, otherwise `"Unknown"` |
| `place` | string | Location if known, otherwise `"Not stated"` |
| `series` | string | Archive series: `TBUK`, `WESA`, or `VF_NEW` |
| `word_count` | int | Number of words in the document |
| `topic` | string | LDA topic label (top 4 keywords separated by ` / `) |
| `topic_keywords` | string | Comma-separated topic keywords |
| `text_preview` | string | First 200 characters of the document text |

### `cross_archive_bridges.json`

Each record is a connection between two documents in **different** archive folders that share similar vocabulary.

| Field | Type | Description |
|-------|------|-------------|
| `doc_a` | string | First document ID |
| `archive_a` | string | First document's archive folder |
| `doc_b` | string | Second document ID |
| `archive_b` | string | Second document's archive folder |
| `similarity` | float | Cosine similarity (0–1, higher = more similar) |
| `shared_terms` | string | Top shared TF-IDF terms connecting these documents |

### `word_embeddings.json`

Top 800 most frequent words with their Word2Vec embeddings projected to 3D.

| Field | Type | Description |
|-------|------|-------------|
| `word` | string | The word |
| `PC1`, `PC2`, `PC3` | float | 3D PCA coordinates from 100-dim Word2Vec space |
| `category` | string | Domain category (see below) |
| `frequency` | int | Corpus-wide word count |

**Categories:** `Buildings & Places`, `Structure & Engineering`, `Fire & Safety`, `People & Organisations`, `Legal & Inquiry`, `Theatre & Culture`, `Other`

### `topics.json`

Summary of LDA topics.

| Field | Type | Description |
|-------|------|-------------|
| `topic_num` | int | Topic number |
| `topic_name` | string | Auto-generated label from top keywords |
| `doc_count` | int | Number of documents assigned to this topic |
| `keywords` | string | Comma-separated topic keywords |
| `sample_archives` | array | Example archive folders containing this topic |

## How the visualisations work

### Concept Network (Plot 1)

- **Nodes** = words from the corpus (top 400 by frequency)
- **Edges** = word pairs with high Pointwise Mutual Information (PMI) — words that co-occur more than chance predicts
- **Colours** = communities detected by the Louvain algorithm on the co-occurrence graph
- **Layout** = 3D spring layout (force-directed) via NetworkX
- **Size** = log word frequency

### Document Network (Plot 2)

- **Nodes** = 812 documents
- **Edges** = top-5 nearest neighbours by TF-IDF cosine similarity (threshold ≥ 0.15)
- **Orange edges** = connections between documents in *different* archive folders
- **Grey edges** = connections within the same folder
- **Colours** = communities detected by Louvain on the similarity graph
- **Layout** = 3D spring layout weighted by similarity

## For the webdev team

The current `index.html` is a self-contained page that loads Plotly.js from CDN. The plot data is embedded directly in the HTML (via Plotly's `to_html()` export). To rebuild the plots from the JSON data files instead:

1. Load `word_embeddings.json` and render a 3D scatter with `Plotly.newPlot()` using `PC1/PC2/PC3` as axes
2. Load `documents.json` for the document scatter — you'll need to recompute the network layout client-side, or pre-compute and add `x/y/z` coordinates to the JSON
3. Load `cross_archive_bridges.json` to draw edges between document nodes
4. `topics.json` provides the legend/filter categories

### Key Plotly.js patterns used

```javascript
// 3D scatter trace
Plotly.newPlot('div', [{
  type: 'scatter3d',
  mode: 'markers+text',
  x: words.map(w => w.PC1),
  y: words.map(w => w.PC2),
  z: words.map(w => w.PC3),
  text: words.map(w => w.word),
  marker: { size: words.map(w => Math.log(w.frequency) * 2) }
}]);

// Edge lines (pairs of points with null separators)
{
  type: 'scatter3d',
  mode: 'lines',
  x: edgeCoords.x,  // [x0, x1, null, x0, x1, null, ...]
  y: edgeCoords.y,
  z: edgeCoords.z,
  line: { color: 'rgba(60,60,60,0.3)', width: 1 }
}
```

## Reproducing the analysis

The Python scripts in `scripts/` reproduce the full pipeline. They expect the source corpus at `Tower Blocks Corpus Cleaned/` (64 text files). Dependencies:

```
pandas numpy scikit-learn gensim plotly networkx
# Optional: bertopic sentence-transformers (for BERTopic comparison)
```

Run order:
1. `corpus_explorer.py` — extracts documents, builds TF-IDF, runs LDA
2. `corpus_networks.py` — builds both network plots + combined HTML
3. `corpus_word2vec.py` — trains Word2Vec, produces embedding plot
4. `corpus_vad_explorer.py` — emotional scoring (requires word norms database)
5. `corpus_bertopic.py` — BERTopic comparison (optional)

## About the corpus

The Tower Blocks archive documents the history of large panel system (LPS) tower blocks in the UK, from the Ronan Point collapse in 1968 through to present-day building safety campaigns. It includes materials from:

- **WESA** — Sam Webb Archive (personal papers of architect Sam Webb)
- **TBUK** — Tower Blocks UK network documents
- **VF_NEW** — Newham local history archive (Ronan Point files, tenant campaigns)

The archive was digitised and OCR-processed as part of ongoing research into UK building safety history.
