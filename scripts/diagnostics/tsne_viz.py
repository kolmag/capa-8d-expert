"""
tsne_viz.py — Embedding space diagnostic for Chroma collections.

Usage (from auditor-expert/ or capa-8d-expert/ root):
    uv run scripts/diagnostics/tsne_viz.py \
        --db_path ./chroma_db \
        --collection capa_8d_expert

Requires:
    uv add scikit-learn plotly numpy chromadb
"""

import argparse
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from chromadb import PersistentClient

def load_embeddings(db_path: str, collection_name: str):
    client = PersistentClient(path=db_path)
    collection = client.get_collection(collection_name)
    result = collection.get(include=["embeddings", "metadatas", "documents"])

    vectors = np.array(result["embeddings"])
    metadatas = result["metadatas"]
    documents = result["documents"]

    # Extract label fields — adjust keys to match your actual metadata schema
    doc_categories = [m.get("doc_category", m.get("source", "unknown")) for m in metadatas]
    sources        = [m.get("source", "unknown") for m in metadatas]
    chunk_ids = [m.get("chunk_id", str(i)) for i, m in enumerate(metadatas)]

    # Short preview of each chunk for hover tooltip
    previews = [d[:120].replace("\n", " ") if d else "" for d in documents]

    return vectors, doc_categories, sources, previews


def run_tsne(vectors: np.ndarray, n_components: int = 2) -> np.ndarray:
    print(f"Running t-SNE on {vectors.shape[0]} vectors × {vectors.shape[1]} dims → {n_components}D ...")
    tsne = TSNE(
        n_components=n_components,
        perplexity=min(30, vectors.shape[0] // 4),  # safe for small collections
        random_state=42,
        max_iter=1000,
        init="pca",        # PCA init is more stable than random
        learning_rate="auto",
    )
    return tsne.fit_transform(vectors)


def plot_2d(reduced, labels, previews, title: str):
    unique_labels = sorted(set(labels))
    colors = px.colors.qualitative.Plotly

    fig = go.Figure()
    for i, label in enumerate(unique_labels):
        mask = [j for j, l in enumerate(labels) if l == label]
        fig.add_trace(go.Scatter(
            x=reduced[mask, 0],
            y=reduced[mask, 1],
            mode="markers",
            name=label,
            marker=dict(size=6, color=colors[i % len(colors)], opacity=0.75),
            text=[previews[j] for j in mask],
            hovertemplate="<b>%{fullData.name}</b><br>%{text}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="t-SNE dim 1",
        yaxis_title="t-SNE dim 2",
        legend_title="doc_category",
        height=700,
        template="plotly_white",
    )
    fig.show()
    fig.write_html(f"tsne_2d_{title.replace(' ', '_')}.html")
    print(f"Saved: tsne_2d_{title.replace(' ', '_')}.html")


def plot_3d(reduced, labels, previews, title: str):
    unique_labels = sorted(set(labels))
    colors = px.colors.qualitative.Plotly

    fig = go.Figure()
    for i, label in enumerate(unique_labels):
        mask = [j for j, l in enumerate(labels) if l == label]
        fig.add_trace(go.Scatter3d(
            x=reduced[mask, 0],
            y=reduced[mask, 1],
            z=reduced[mask, 2],
            mode="markers",
            name=label,
            marker=dict(size=4, color=colors[i % len(colors)], opacity=0.75),
            text=[previews[j] for j in mask],
            hovertemplate="<b>%{fullData.name}</b><br>%{text}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="dim 1", yaxis_title="dim 2", zaxis_title="dim 3"),
        legend_title="doc_category",
        height=800,
        template="plotly_white",
    )
    fig.show()
    fig.write_html(f"tsne_3d_{title.replace(' ', '_')}.html")
    print(f"Saved: tsne_3d_{title.replace(' ', '_')}.html")


def main():
    parser = argparse.ArgumentParser(description="t-SNE diagnostic for a Chroma collection")
    parser.add_argument("--db_path",    required=True, help="Path to chroma_db directory")
    parser.add_argument("--collection", required=True, help="Collection name")
    parser.add_argument("--dims",       type=int, default=2, choices=[2, 3], help="2D or 3D plot")
    args = parser.parse_args()

    vectors, doc_categories, sources, previews = load_embeddings(args.db_path, args.collection)
    print(f"Loaded {len(vectors)} chunks from '{args.collection}'")
    print(f"Unique doc_categories: {sorted(set(doc_categories))}")

    title = f"{args.collection} embeddings"
    reduced = run_tsne(vectors, n_components=args.dims)

    if args.dims == 2:
        plot_2d(reduced, doc_categories, previews, title)
    else:
        plot_3d(reduced, doc_categories, previews, title)


if __name__ == "__main__":
    main()