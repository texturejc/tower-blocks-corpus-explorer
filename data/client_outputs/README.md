# Client Outputs

Three CSV files extracted from the semantic network analysis of the Tower Blocks Corpus.

## Files

### 1. `concept_network_clusters.csv`

Every word in the concept co-occurrence network, with its cluster membership.

| Column | Description |
|--------|-------------|
| `word` | The word |
| `cluster_id` | Integer cluster ID (0 = largest cluster) |
| `cluster_name` | Human-readable label from the cluster's top 3 words |
| `frequency` | How often this word appears across the full corpus |
| `degree` | Number of co-occurrence links to other words in the network |

268 words across 41 clusters.

---

### 2. `document_network_edges.csv`

Pairs of documents that share similar vocabulary, thresholded at the 95th percentile of pairwise cosine similarity (≥ 0.15).

| Column | Description |
|--------|-------------|
| `doc_a` | First document ID |
| `doc_b` | Second document ID |
| `similarity` | Cosine similarity score (0–1, higher = more similar) |
| `cross_archive` | `True` if the two documents come from different archive folders |
| `shared_terms` | Top TF-IDF terms that both documents share |

16,464 edges total; 9,048 (55%) are cross-archive connections.

---

### 3. `concept_cluster_silhouettes.csv`

Quality scores for each concept cluster, measuring how well its words hold together as a group.

| Column | Description |
|--------|-------------|
| `cluster_id` | Cluster ID (matches `concept_network_clusters.csv`) |
| `cluster_name` | Human-readable label |
| `n_words` | Number of words in the cluster |
| `mean_silhouette` | Average silhouette score across all words in the cluster |
| `min_silhouette` | Silhouette score of the worst-fitting word |
| `max_silhouette` | Silhouette score of the best-fitting word |
| `best_fitting_words` | Top 5 words most tightly bound to this cluster |
| `worst_fitting_words` | 3 words most loosely bound (closest to belonging elsewhere) |

27 clusters with ≥ 2 words (smaller clusters omitted since silhouette requires at least 2 members).

## How to read the silhouette scores

Each word in a cluster gets its own silhouette score, which measures how well it fits its assigned cluster compared to the next closest cluster. Scores range from −1 to +1.

- **Mean silhouette** is the average across all words in the cluster — the overall coherence. A high mean (like 0.94 for "taylor / woodrow / anglian") means most words in that cluster are much closer to each other than to any other cluster's words.

- **Max silhouette** is the best-fitting word — the one most tightly bound to its cluster. For example, in the "goodfellow / griffiths / eveleigh" cluster, the highest-scoring word is the one whose usage pattern is most distinctive to that group and furthest from all other clusters.

- **Min silhouette** is the worst-fitting word — the one that nearly belongs to a different cluster. A negative min means at least one word is actually closer to another cluster's centre than its own. For instance, in "tenants / council / housing", a word like "local" might score negatively because it sits almost equidistant between that cluster and the "local / government / authorities" cluster.

In short: mean tells you cluster quality overall, max tells you the core anchor word, and min tells you the weakest link — the word most ambiguously placed that might reasonably belong elsewhere.

### What the overall score means

The corpus-wide silhouette score is 0.012 — very low. This reflects the archive's thematic homogeneity: nearly every document is about tower blocks, building safety, and public inquiries, so the vocabulary overlaps heavily across clusters. The tight clusters (silhouette > 0.3) are fixed collocations like "Taylor Woodrow Anglian" or "dry pack" that always co-occur. The large general clusters (silhouette < 0) contain words shared across many topics, which is expected for a specialised archive like this.
