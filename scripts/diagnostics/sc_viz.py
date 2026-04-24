"""
sc_viz.py — Cosine similarity (Sc) diagnostic for Chroma collections.

Produces three analyses:
  1. Sc distribution per doc_category — shows how semantically tight each
     category is internally (intra-category similarity).
  2. Sc heatmap between all chunks — shows cross-category overlap. High Sc
     between different categories = retrieval competition risk.
  3. Per-query Sc distribution — given a set of test queries, shows the
     distribution of Sc scores for retrieved candidates before and after
     BGE reranking (requires --queries flag).

Usage (from capa-8d-expert/ root):
    # Intra-category + heatmap only:
    uv run scripts/diagnostics/sc_viz.py \\
        --db_path ./chroma_db \\
        --collection capa_8d_expert

    # Include per-query analysis against test questions:
    uv run scripts/diagnostics/sc_viz.py \\
        --db_path ./chroma_db \\
        --collection capa_8d_expert \\
        --queries evaluation/tests_v3.jsonl \\
        --n_queries 20

Outputs:
    sc_intra_{collection}.html     — intra-category Sc distributions (violin plot)
    sc_heatmap_{collection}.html   — cross-category Sc heatmap
    sc_queries_{collection}.html   — per-query Sc distributions (if --queries provided)

Requires:
    uv add scikit-learn plotly numpy chromadb openai
"""

import argparse
import json
import random
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
from chromadb import PersistentClient


# ─── Chroma loader ────────────────────────────────────────────────────────────

def load_embeddings(db_path: str, collection_name: str):
    client = PersistentClient(path=db_path)
    collection = client.get_collection(collection_name)
    result = collection.get(include=["embeddings", "metadatas", "documents"])

    vectors     = np.array(result["embeddings"])
    metadatas   = result["metadatas"]
    documents   = result["documents"]
    categories  = [m.get("doc_category", "unknown") for m in metadatas]
    sources     = [m.get("source_file",  "unknown") for m in metadatas]
    headlines   = [m.get("headline",     "")        for m in metadatas]
    previews    = [d[:100].replace("\n", " ") if d else "" for d in documents]

    return vectors, categories, sources, headlines, previews


# ─── Plot 1: Intra-category Sc distributions ─────────────────────────────────

def plot_intra_category(vectors: np.ndarray, categories: list, title: str):
    """
    For each category, compute pairwise cosine similarity between all chunks
    in that category. Plot as violin to show spread and median.
    Tight violin = semantically coherent category (good).
    Wide violin = scattered category (retrieval competition risk).
    """
    unique_cats = sorted(set(categories))
    colors = px.colors.qualitative.Plotly

    fig = go.Figure()
    stats = []

    for i, cat in enumerate(unique_cats):
        idxs = [j for j, c in enumerate(categories) if c == cat]
        if len(idxs) < 2:
            continue
        cat_vecs = vectors[idxs]
        sim_matrix = cosine_similarity(cat_vecs)
        # Upper triangle only (exclude diagonal self-similarity = 1.0)
        upper = sim_matrix[np.triu_indices(len(idxs), k=1)]

        stats.append({
            'category': cat,
            'mean': upper.mean(),
            'median': np.median(upper),
            'std': upper.std(),
            'n_chunks': len(idxs),
        })

        fig.add_trace(go.Violin(
            y=upper,
            name=f"{cat} (n={len(idxs)})",
            box_visible=True,
            meanline_visible=True,
            fillcolor=colors[i % len(colors)],
            opacity=0.7,
            line_color=colors[i % len(colors)],
        ))

    fig.update_layout(
        title=f"Intra-category cosine similarity — {title}",
        yaxis_title="Cosine similarity (Sc)",
        xaxis_title="doc_category",
        height=600,
        template="plotly_white",
        showlegend=True,
        yaxis=dict(range=[0, 1]),
    )

    print("\n=== Intra-category Sc statistics ===")
    print(f"{'Category':<20} {'N':>5} {'Mean Sc':>8} {'Median':>8} {'Std':>7}")
    print("-" * 55)
    for s in sorted(stats, key=lambda x: x['mean'], reverse=True):
        print(f"{s['category']:<20} {s['n_chunks']:>5} {s['mean']:>8.4f} {s['median']:>8.4f} {s['std']:>7.4f}")

    fname = f"sc_intra_{title.replace(' ', '_')}.html"
    fig.write_html(fname)
    fig.show()
    print(f"\nSaved: {fname}")
    return stats


# ─── Plot 2: Cross-category Sc heatmap ───────────────────────────────────────

def plot_cross_category_heatmap(vectors: np.ndarray, categories: list,
                                 sources: list, title: str):
    """
    Compute mean cosine similarity between every pair of categories.
    High off-diagonal values = categories that compete in retrieval.
    Diagonal = intra-category cohesion (same as mean from Plot 1).
    """
    unique_cats = sorted(set(categories))
    n = len(unique_cats)
    heatmap_matrix = np.zeros((n, n))

    for i, cat_i in enumerate(unique_cats):
        idxs_i = [j for j, c in enumerate(categories) if c == cat_i]
        vecs_i = vectors[idxs_i]
        for j, cat_j in enumerate(unique_cats):
            idxs_j = [k for k, c in enumerate(categories) if c == cat_j]
            vecs_j = vectors[idxs_j]
            sim = cosine_similarity(vecs_i, vecs_j)
            if i == j:
                # Intra: upper triangle only
                upper = sim[np.triu_indices(len(idxs_i), k=1)]
                heatmap_matrix[i][j] = upper.mean() if len(upper) > 0 else 0
            else:
                heatmap_matrix[i][j] = sim.mean()

    # Annotate with values
    annotations = []
    for i in range(n):
        for j in range(n):
            annotations.append(dict(
                x=unique_cats[j],
                y=unique_cats[i],
                text=f"{heatmap_matrix[i][j]:.3f}",
                showarrow=False,
                font=dict(size=11, color="white" if heatmap_matrix[i][j] > 0.6 else "black"),
            ))

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_matrix,
        x=unique_cats,
        y=unique_cats,
        colorscale="RdYlGn",
        zmin=0.3,
        zmax=0.9,
        colorbar=dict(title="Mean Sc"),
    ))

    fig.update_layout(
        title=f"Cross-category cosine similarity heatmap — {title}<br>"
              f"<sub>High off-diagonal = retrieval competition risk</sub>",
        height=600,
        width=700,
        template="plotly_white",
        annotations=annotations,
    )

    print("\n=== Cross-category Sc (top competition risks) ===")
    pairs = []
    for i, cat_i in enumerate(unique_cats):
        for j, cat_j in enumerate(unique_cats):
            if i < j:
                pairs.append((heatmap_matrix[i][j], cat_i, cat_j))
    pairs.sort(reverse=True)
    print(f"{'Category A':<20} {'Category B':<20} {'Mean Sc':>8}")
    print("-" * 52)
    for sc, a, b in pairs[:10]:
        risk = " ⚠️" if sc > 0.65 else ""
        print(f"{a:<20} {b:<20} {sc:>8.4f}{risk}")

    fname = f"sc_heatmap_{title.replace(' ', '_')}.html"
    fig.write_html(fname)
    fig.show()
    print(f"\nSaved: {fname}")


# ─── Plot 3: Per-query Sc distributions ──────────────────────────────────────

def plot_query_sc(vectors: np.ndarray, categories: list, sources: list,
                  queries_file: str, n_queries: int, title: str):
    """
    For a sample of test queries:
    1. Embed each query using text-embedding-3-small
    2. Compute Sc between query and all chunks
    3. Plot distribution of top-K Sc scores per category
    Shows: which categories BGE sees as relevant for different query types.
    """
    import os
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Load queries
    with open(queries_file) as f:
        tests = [json.loads(line) for line in f if line.strip()]

    # Sample n_queries, stratified by category if possible
    cats_in_tests = list(set(t.get("category", "unknown") for t in tests))
    sampled = []
    per_cat = max(1, n_queries // len(cats_in_tests))
    for cat in cats_in_tests:
        cat_tests = [t for t in tests if t.get("category") == cat]
        sampled.extend(random.sample(cat_tests, min(per_cat, len(cat_tests))))
    sampled = sampled[:n_queries]

    print(f"\nEmbedding {len(sampled)} queries...")
    query_texts = [t["question"] for t in sampled]
    query_cats  = [t.get("category", "unknown") for t in sampled]

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query_texts,
    )
    query_vecs = np.array([e.embedding for e in response.data])

    # Compute Sc between each query and all chunks
    sc_matrix = cosine_similarity(query_vecs, vectors)  # (n_queries, n_chunks)

    # For each query, get top-20 chunk scores and their categories
    K = 20
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    unique_chunk_cats = sorted(set(categories))

    all_scores_by_chunk_cat = {cat: [] for cat in unique_chunk_cats}

    for qi in range(len(sampled)):
        scores = sc_matrix[qi]
        top_k_idxs = np.argsort(scores)[::-1][:K]
        for idx in top_k_idxs:
            all_scores_by_chunk_cat[categories[idx]].append(scores[idx])

    for i, cat in enumerate(unique_chunk_cats):
        scores = all_scores_by_chunk_cat[cat]
        if not scores:
            continue
        fig.add_trace(go.Violin(
            y=scores,
            name=f"{cat}",
            box_visible=True,
            meanline_visible=True,
            fillcolor=colors[i % len(colors)],
            opacity=0.7,
            line_color=colors[i % len(colors)],
        ))

    fig.update_layout(
        title=f"Top-{K} chunk Sc scores by category — {len(sampled)} queries<br>"
              f"<sub>Higher = more competitive in retrieval for these queries</sub>",
        yaxis_title="Cosine similarity (Sc)",
        height=600,
        template="plotly_white",
        yaxis=dict(range=[0, 1]),
    )

    print(f"\n=== Top-{K} chunk Sc by category (mean across {len(sampled)} queries) ===")
    means = [(np.mean(v), cat) for cat, v in all_scores_by_chunk_cat.items() if v]
    for mean_sc, cat in sorted(means, reverse=True):
        print(f"  {cat:<20} mean Sc = {mean_sc:.4f}  (n={len(all_scores_by_chunk_cat[cat])} appearances)")

    fname = f"sc_queries_{title.replace(' ', '_')}.html"
    fig.write_html(fname)
    fig.show()
    print(f"\nSaved: {fname}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cosine similarity (Sc) diagnostic for Chroma collections"
    )
    parser.add_argument("--db_path",    required=True,  help="Path to chroma_db directory")
    parser.add_argument("--collection", required=True,  help="Collection name")
    parser.add_argument("--queries",    default=None,   help="Path to .jsonl test questions file")
    parser.add_argument("--n_queries",  type=int, default=20,
                        help="Number of queries to sample for per-query analysis (default: 20)")
    parser.add_argument("--skip_heatmap", action="store_true",
                        help="Skip the cross-category heatmap (slow for large collections)")
    args = parser.parse_args()

    vectors, categories, sources, headlines, previews = load_embeddings(
        args.db_path, args.collection
    )
    print(f"Loaded {len(vectors)} chunks from '{args.collection}'")
    print(f"Categories: {sorted(set(categories))}")

    title = args.collection

    # Plot 1: Intra-category distributions
    plot_intra_category(vectors, categories, title)

    # Plot 2: Cross-category heatmap
    if not args.skip_heatmap:
        plot_cross_category_heatmap(vectors, categories, sources, title)

    # Plot 3: Per-query Sc (optional)
    if args.queries:
        plot_query_sc(vectors, categories, sources,
                      args.queries, args.n_queries, title)


if __name__ == "__main__":
    main()
