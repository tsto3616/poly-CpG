import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import networkx as nx

# this is to build the feature matrix for the construction of the tree in the figure
def build_feature_matrix(M, weights=None):
    """
    M: DataFrame (n_samples, n_cpgs)
    weights: array-like (n_cpgs,) or None
    """
    X = M.values.astype("float32")
    if weights is not None:
        X = X * weights[None, :]
    return X

# embed the dataframe as 20 PC to reduce the complexity to the variances
def run_pca(X, n_components=20):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

# the embedded space allows for the construction of a k-NN graph for the structuring of the tree
def build_knn_graph(X_pca, k=10, metric="euclidean"):
    n = X_pca.shape[0]
    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(X_pca)
    distances, indices = nn.kneighbors(X_pca)

    G_knn = nx.Graph()
    for i in range(n):
        for j_idx, d in zip(indices[i], distances[i]):
            if i == j_idx:
                continue
            G_knn.add_edge(i, j_idx, weight=float(d))
    return G_knn

# minimum spanning tree organises the k-NN clusters into a meaningful latent space in which the 
# connections between the sample denote common ancestory - much like a phylogeny. 
def build_mst(G_knn):
    G_mst = nx.minimum_spanning_tree(G_knn, weight="weight")
    return G_mst

# the diffusion pseudotime was constructed to add a relative timeframe for the molecular differentiation
# of the samples/ animals 
def compute_diffusion_pseudotime(G_mst, meta, root_by="Age_chrono"):
    """
    meta: DataFrame indexed by sample ID, with column root_by
    """
    root_id = meta[root_by].idxmin()
    id_to_idx = {id_: i for i, id_ in enumerate(meta.index)}
    root_node = id_to_idx[root_id]

    dpt = nx.shortest_path_length(G_mst, source=root_node, weight="weight")
    n = len(meta)
    dpt_array = np.array([dpt[i] for i in range(n)], dtype="float32")

    dpt_norm = (dpt_array - dpt_array.min()) / (dpt_array.max() - dpt_array.min())
    return dpt_norm

# this is used to plot the minimum spanning tree along the pseudotime
def plot_mst_embedding(X_emb, G_mst, clusters, out_file, cluster_palette="Set2"):
    plt.figure(figsize=(8, 8))

    for u, v in G_mst.edges():
        plt.plot(
            [X_emb[u, 0], X_emb[v, 0]],
            [X_emb[u, 1], X_emb[v, 1]],
            color="lightgray",
            linewidth=1,
            alpha=0.7,
        )

    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    palette = sns.color_palette(cluster_palette, len(uniq))
    cmap = {c: palette[i] for i, c in enumerate(uniq)}
    colors = [cmap[c] for c in clusters]

    plt.scatter(
        X_emb[:, 0],
        X_emb[:, 1],
        c=colors,
        s=40,
        edgecolor="k",
        alpha=0.9,
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Methylation-based MST (PCA embedding)")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

# the tree is plotted along side of the pseudotimeframe tree
def plot_tree_with_tiles_aligned(
    G_mst,
    dpt_norm,
    meta,
    cluster_col="Cluster",
    tile_vars=None,
    out_file="tree_with_phenotype_tiles.svg",
): # these are the are the variables you want to include!
    if tile_vars is None:
        tile_vars = ["chrono_age", "epi_age", "MaxADC", "ADC_stack3", "ADC_EWAS", "GCSH", "DNMT1", "LMX1A"]

    # this provides the master ordering for the two graphs to line up perfectly
    order = np.argsort(dpt_norm)
    meta_ord = meta.iloc[order]
    dpt_ord = dpt_norm[order]

    # Shared x-axis for EVERYTHING
    x = np.arange(len(order))

    # build the figure with the top panel being the tree.
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.05)

    # the top panel is the tree
    ax_tree = fig.add_subplot(gs[0])

    # Map original MST node indices to ordered positions
    idx_to_ord = {orig: i for i, orig in enumerate(order)}

    # Draw MST edges using ordered positions
    for u, v in G_mst.edges():
        ax_tree.plot(
            [idx_to_ord[u], idx_to_ord[v]],
            [dpt_norm[u], dpt_norm[v]],
            color="lightgray",
            linewidth=1,
            alpha=0.7,
            zorder=1
        )

    # include the HDBSCAN clustering colours 
    cluster_order = [-1, 0, 1, 2]

    clusters = meta_ord[cluster_col].astype(int).values
    uniq = sorted(np.unique(clusters), key=lambda x: (x == -1, x))
    palette = sns.color_palette("tab10", len(cluster_order))
    cmap = {c: palette[i] for i, c in enumerate(cluster_order)}
    colors = [cmap[c] for c in clusters]

    # Draw dots
    ax_tree.scatter(
        x,
        dpt_ord,
        c=colors,
        s=60,
        edgecolor="k",
        zorder=2
    )

    # Legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cmap[c], markersize=10, label=f"Cluster {c}")
        for c in cluster_order
    ]
    ax_tree.legend(handles=handles, title="Clusters", loc="upper left",
                   bbox_to_anchor=(1.02, 1.0))

    ax_tree.set_xlim(-0.5, len(order) - 0.5)
    ax_tree.set_xticks([])
    ax_tree.set_ylabel("Diffusion pseudotime")
    ax_tree.set_title("Aligned Tree + Phenotype Tiles")

    # Bottom panel is the heat map
    ax_heat = fig.add_subplot(gs[1], sharex=ax_tree)

    ax_heat.get_yaxis().set_visible(False)

    # Build heatmap matrix (variables × individuals)
    heat_df = meta_ord[tile_vars]

    def pct_normalize(col):
        # rank 1..n → convert to 0..100%
        ranks = col.rank(method="average")
        return 100 * (ranks - 1) / (len(col) - 1)

    heat_norm = heat_df.apply(pct_normalize, axis=0).T.values

    num_vars, num_samples = heat_norm.shape

    # Have to use imshow instead of seaborne
    ax_heat = fig.add_subplot(gs[1], sharex=ax_tree)

    # Draw heatmap using imshow (perfect alignment)
    img = ax_heat.imshow(
        heat_norm,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        extent=[-0.5, num_samples - 0.5, num_vars, 0]
    )

    # Create an external axis for the colorbar
    # Create a brand‑new axis for the colorbar
    cax = fig.add_axes([0.92, 0.25, 0.02, 0.5])  
    # [left, bottom, width, height] in figure coordinates

    # Create the colorbar
    cb = fig.colorbar(img, cax=cax)

    # Convert numeric ticks to percentages
    ticks = cb.get_ticks()
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{int(t)}%" for t in ticks])

    # Remove y-axis numbers and ticks
    ax_heat.set_yticks(np.arange(num_vars) + 0.5)
    ax_heat.set_yticklabels(tile_vars)

    # No x tick labels
    ax_heat.set_xticks([])

    # Force exact alignment
    ax_heat.set_xlim(-0.5, num_samples - 0.5)

    ax_heat.set_aspect("auto")

    # This connects the dots from the tree to the heatmap tiles
    for i in range(len(order)):
        ax_tree.plot(
            [i, i],
            [dpt_ord[i], -0.02],
            color="gray",
            linewidth=0.5,
            alpha=0.5,
            zorder=0
        )

    plt.subplots_adjust(hspace=0.05)
    plt.savefig(out_file, dpi=300)
    plt.close()


# This connects everything to run it all!
def main():
    # 1. EWAS CpG-level effects
    df_ewas = pd.read_csv("EWAS_ADVI_batched.csv", engine="python")
    gamma_map = df_ewas.set_index("CpG")["gamma_mean"]

    # 2. Residuals + CpGs + ADC_stack3 + Cluster
    df_resid = pd.read_csv("CSL_with_Bayesian_residuals.csv", engine="python").dropna()
    # assume there is an ID column to align individuals
    if "CSL_Samples" not in df_resid.columns:
        raise ValueError("CSL_with_Bayesian_residuals.csv must contain an 'ID' column.")

    df_resid = df_resid.set_index("CSL_Samples")

    cpg_cols = [c for c in df_resid.columns if c.startswith("cg")]
    M_full = df_resid[cpg_cols]

    # 3. Metadata: MaxADC, epi_age, chrono_age
    df_meta = pd.read_csv("CSL_HDBSCAN_with_LatentAge_and_weights.csv", engine="python")
    if "CSL_Samples" not in df_meta.columns:
        raise ValueError("CSL_HDBSCAN_with_LatentAge_and_weights.csv must contain an 'ID' column.")

    df_meta = df_meta.set_index("CSL_Samples")

    # Merge ADC_stack3 and Cluster from residuals into meta
    for col in ["Bayes_mu", "Cluster"]:
        if col not in df_resid.columns:
            raise ValueError(f"Column '{col}' not found in CSL_with_Bayesian_residuals.csv")
        df_meta[col] = df_resid[col]
        
    df_genes = pd.read_csv("Gene_sample_effect_scores.csv")

    df_genes = df_genes.set_index("CSL_Samples")
    for col in ["GCSH", "DNMT1", "LMX1A"]:
        if col not in df_genes.columns:
            raise ValueError(f"Column '{col}' not found in the Genes.csv")
        df_meta[col] = df_genes[col]
        
    # Check required meta columns
    required_meta = ["MaxADC", "Age_epi", "Age_chrono", "Bayes_mu", "Cluster", "GCSH", "DNMT1", "LMX1A"]
    missing = [c for c in required_meta if c not in df_meta.columns]
    if missing:
        raise ValueError(f"Missing columns in CSL_HDBSCAN_with_LatentAge_and_weights.csv: {missing}")

    # Align M_full to meta index
    common_ids = df_meta.index.intersection(M_full.index)
    df_meta = df_meta.loc[common_ids]
    M_full = M_full.loc[common_ids]

    # Reindex gamma to CpGs in M_full
    gamma_vec = gamma_map.reindex(M_full.columns).fillna(0.0).values.astype("float32")

    # Compute EWAS polyCpG score and ADC_EWAS
    polyCpG_EWAS = M_full.values.astype("float32") @ gamma_vec
    df_meta["ADC_EWAS"] = polyCpG_EWAS

    # Build feature matrix for trajectory (you can choose raw M or EWAS-weighted)
    X = build_feature_matrix(M_full, weights=None)

    # PCA
    X_pca, pca = run_pca(X, n_components=20)
    X_emb = X_pca[:, :2]

    # k-NN graph and MST
    G_knn = build_knn_graph(X_pca, k=10, metric="euclidean")
    G_mst = build_mst(G_knn)

    # Diffusion pseudotime (root by chrono_age)
    dpt_norm = compute_diffusion_pseudotime(G_mst, df_meta, root_by="Age_chrono")

    # Plot MST in PCA space with cluster coloring
    clusters = df_meta["Cluster"].values
    plot_mst_embedding(
        X_emb,
        G_mst,
        clusters,
        out_file="mst_pca_embedding_clusters.svg",
        cluster_palette="Set2",
    )

    # Plot tree + phenotype tiles - replace with wanted vars
    tile_vars = ["Age_chrono", "Age_epi", "MaxADC", "Bayes_mu", "ADC_EWAS", "GCSH", "DNMT1", "LMX1A"]
    plot_tree_with_tiles_aligned(
        G_mst,
        dpt_norm,
        df_meta,
        cluster_col="Cluster",
        tile_vars=tile_vars,
        out_file="tree_with_phenotype_tiles.svg",
    )


if __name__ == "__main__":
    main()
