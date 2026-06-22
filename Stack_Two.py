# Stack two is for the calculation of the latent age and reliability cluster weights
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load the csv file from Stack One (note must use the original beta values not the M scores)
beta_df = pd.read_csv("CSL_master_table.csv")

# using the blood general pinniped clock for epigenomic age estimation - supplementary materials
# from the original methylation study (not a coauthor)
blood_clock = {
    "intercept": 28.61220287,
    "cg00764339": 30.88980172,
    "cg00930873": 6.08373985,
    "cg11728741": 25.01531776,
    "cg12017700": 123.80526190,
    "cg12841266": 4.68739715,
    "cg14870509": -0.33840083,
    "cg16702105": 5.82507567,
    "cg18992104": -31.90455582,
    "cg21801378": 17.84724510,
    "cg21852818": 0.33912659,
    "cg26051365": -1.85334983,
    "cg26053530": -6.91527501
}

# select all values not including the model intercept
clock_cpgs = [cg for cg in blood_clock.keys() if cg != "intercept"]

# Now for the identification of the CpGs shared by the molecular dating profile (blood_clock) and 
# the CpGs of each animal 
available_cpgs = [cg for cg in clock_cpgs if cg in beta_df.columns]
print("Clock CpGs found:", available_cpgs)

if len(available_cpgs) == 0:
    raise ValueError("No clock CpGs found in CSL_master_table.csv.")

# this is to extract the beta scores
beta_vals = beta_df[available_cpgs].apply(pd.to_numeric, errors="coerce")

# now compute the methylation-age comparisons using the methylation formula defined by the original
# pinniped study by: Robeck et al. (2023)
epi_age = np.full(len(beta_df), blood_clock["intercept"])

# the formula for linear regression of the epigenetic age
for cg in available_cpgs:
    epi_age += beta_vals[cg] * blood_clock[cg]

beta_df["EpigeneticAge"] = epi_age

# load the HDBSCAN results
cluster_df = pd.read_csv("CSL_clusters_PCA20_HDBSCAN.csv")

# now load the merged files for the HDBSCAN and the epigenetic age of each individual. 
merged = cluster_df.merge(
    beta_df[["CSL_Samples", "EpigeneticAge"]],
    on="CSL_Samples",
    how="left"
)

# now save the results to the csv file
merged.to_csv("CSL_HDBSCAN_with_EpigeneticAge.csv", index=False)

df = merged

# Rename for clarity (adjust if your column names differ)
df = df.rename(columns={
    "Age": "Age_chrono",
    "EpigeneticAge": "Age_epi",
    "Cluster": "Cluster",
    "ClusterProb": "ClusterProb"
})

# CpG columns used for clustering (fill this with your actual CpG list)
cpg_cols = [c for c in df.columns if c.startswith("cg")]  # example heuristic

# Latent age definition
# LatentAge_i(lambda) = lambda * Age_chrono_i + (1 - lambda) * Age_epi_i
def latent_age(df, lam):
    return lam * df["Age_chrono"].values + (1.0 - lam) * df["Age_epi"].values

# 2.4.3 Hyperparameter tuning (1D Fisher problem)
# J(lambda) = S_B(lambda)/S_W(lambda), lambda* = argmax_{lambda∈[0,1]} J(lambda)
def fisher_J(df, lam):
    LA = latent_age(df, lam)
    df_tmp = df.copy()
    df_tmp["LatentAge"] = LA

    # cluster means
    cluster_means = df_tmp.groupby("Cluster")["LatentAge"].mean()
    # global mean
    global_mean = df_tmp["LatentAge"].mean()
    # cluster sizes
    cluster_sizes = df_tmp["Cluster"].value_counts().sort_index()

    # S_W: within-cluster variance
    SW = 0.0
    for k, mean_k in cluster_means.items():
        mask = df_tmp["Cluster"] == k
        SW += np.sum((df_tmp.loc[mask, "LatentAge"].values - mean_k) ** 2)

    # S_B: between-cluster variance
    SB = 0.0
    for k, mean_k in cluster_means.items():
        n_k = cluster_sizes[k]
        SB += n_k * (mean_k - global_mean) ** 2

    if SW == 0:
        return 0.0
    return SB / SW

# grid search over lamdba ∈ [0,1]
lams = np.linspace(0.0, 1.0, 1001)
J_vals = np.array([fisher_J(df, lam) for lam in lams])
lam_star = lams[np.argmax(J_vals)]
print("λ* =", lam_star)

# compute latent age at lambda*
df["LatentAge"] = latent_age(df, lam_star)

# Reliability coherence score r_k^coh
# r_k^coh = 1 / (1 + Var(Age_chrono_i : i∈k) + Var(Age_epi_i : i∈k))
r_coh = {}
for k, group in df.groupby("Cluster"):
    var_chrono = np.var(group["Age_chrono"].values, ddof=1) if len(group) > 1 else 0.0
    var_epi = np.var(group["Age_epi"].values, ddof=1) if len(group) > 1 else 0.0
    r_coh[k] = 1.0 / (1.0 + var_chrono + var_epi)

# Ordering reliability r_k^ord
# \bar{Age}_k = mean(Age_epi_i : i∈k)
# r_k^ord = I( \bar{Age}_k < \bar{Age}_{k+1} )
cluster_order = (
    df.groupby("Cluster")["Age_epi"]
      .mean()
      .sort_values()
)
ordered_clusters = cluster_order.index.tolist()

r_ord = {k: 0.0 for k in ordered_clusters}
for idx in range(len(ordered_clusters) - 1):
    k = ordered_clusters[idx]
    k_next = ordered_clusters[idx + 1]
    mean_k = cluster_order.loc[k]
    mean_next = cluster_order.loc[k_next]
    r_ord[k] = 1.0 if mean_k < mean_next else 0.0
# last cluster has no k+1; keep r_ord[last] = 0.0

# For the Tightness r_k^tight
# r_k^tight = 1 / (1 + Var(M_ij : i∈k, j∈C))
r_tight = {}
for k, group in df.groupby("Cluster"):
    if len(group) > 1 and len(cpg_cols) > 0:
        M_vals = group[cpg_cols].values.reshape(-1)
        var_M = np.var(M_vals, ddof=1)
    else:
        var_M = 0.0
    r_tight[k] = 1.0 / (1.0 + var_M)

# 2.4.7 Final cluster reliability r_k and individual weights w_i
# r_k = (r_k^coh + r_k^ord + r_k^tight) / 3
# w_i = p_i^cluster * r_{c_i}
r_k = {}
for k in ordered_clusters:
    r_k[k] = (r_coh.get(k, 0.0) + r_ord.get(k, 0.0) + r_tight.get(k, 0.0)) / 3.0

df["r_coh"] = df["Cluster"].map(r_coh)
df["r_ord"] = df["Cluster"].map(r_ord)
df["r_tight"] = df["Cluster"].map(r_tight)
df["r_k"] = df["Cluster"].map(r_k)

df["w_i"] = df["ClusterProb"] * df["r_k"]

# save the results if needed
df.to_csv("CSL_HDBSCAN_with_LatentAge_and_weights.csv", index=False)
print("Saved CSL_HDBSCAN_with_LatentAge_and_weights.csv")

# define the cluster shapes for mapping
cluster_ids = sorted(df["Cluster"].unique())
markers = ['o', 's', 'D', '^', 'v', 'P', 'X', '*']  # extend if needed
cluster_marker_map = {cid: markers[i % len(markers)] for i, cid in enumerate(cluster_ids)}

# plot the clusters
plt.figure(figsize=(10, 8))

for cid in cluster_ids:
    subset = df[df["Cluster"] == cid]
    plt.scatter(
        subset["Age_chrono"],
        subset["Age_epi"],
        s=80,
        c=subset["ClusterProb"],
        cmap="viridis",
        marker=cluster_marker_map[cid],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9,
        label=f"Cluster {cid}"
    )
    
# include the label aesthetics 
plt.xlabel("Chronological Age (years)", fontsize=14)
plt.ylabel("Epigenetic Age (years)", fontsize=14)
plt.title("Chronological Age vs Epigenetic Age\nShapes = Clusters, Colour = Cluster Probability", fontsize=16)

cbar = plt.colorbar()
cbar.set_label("Cluster Membership Probability", fontsize=12)

plt.legend(title="Cluster", fontsize=10)
plt.tight_layout()

# can save in the interactive viewer
plt.show()
