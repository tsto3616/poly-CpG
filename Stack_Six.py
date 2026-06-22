import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# horseshoe
# laptop version input the csv
df_hs = pd.read_csv("horseshoe_effect_sizes.csv")

df_hs = df_hs.sort_values(["chr", "pos"])
df_hs["chr"] = df_hs["chr"].astype(str)

# cumulative genomic position
df_hs["posi_cumu"] = df_hs.groupby("chr")["pos"].transform(
    lambda x: x + x.min()
)

# note the above step was performeed in stack four and so was not neeeded for the ADVI-EWAS. 
# Create cumulative genomic position
df = pd.read_csv("EWAS_ADVI_effect_sizes.csv")
# compare the horseshoe manhattan to the ewas layer
# Bayesian significance threshold
df["significant"] = (
    (df["p_gamma_gt_0"] > 0.975) |
    (df["p_gamma_gt_0"] < 0.025)
)

df["direction"] = np.where(
    df["p_gamma_gt_0"] > 0.975, "positive",
    np.where(df["p_gamma_gt_0"] < 0.025, "negative", "none")
)

# start of the volcano plot
df["volcano_y"] = np.where(
    df["gamma_mean"] >= 0,
    -np.log10(1 - df["p_gamma_gt_0"]),
    -np.log10(df["p_gamma_gt_0"])
)

# --- EWAS layer ---
plt.scatter(
    df["posi_cumu"],
    df["volcano_y"],
    c="royalblue",
    s=8,
    alpha=0.5,
    label="EWAS"
)

# now repeat for the Horseshoe layer (first sort according to locations): 
df_hs = df_hs.sort_values(["chr", "pos"])
df_hs["chr"] = df_hs["chr"].astype(str)

# cumulative genomic position
df_hs["posi_cumu"] = df_hs.groupby("chr")["pos"].transform(
    lambda x: x + x.min()
)

# beta_mean is from the df_hs dataframe already
# --- Horseshoe layer ---
plt.scatter(
    df_hs["posi_cumu"],
    df_hs["beta_mean"],
    c="black",
    s=12,
    alpha=0.8,
    label="Horseshoe"
)

# Highlight top 1% horseshoe CpGs
hs_sig = df_hs[df_hs["beta_abs"] > np.percentile(df_hs["beta_abs"], 99)] # beta_abs from the df
plt.scatter(
    hs_sig["posi_cumu"],
    hs_sig["beta_mean"],
    c="red",
    s=30,
    label="Top Horseshoe CpGs"
)

# now convert into a format suitable for comparisons to the ADVI-EWAS
df_hs["p_beta_gt_0"] = 1 - norm.cdf(
    0,
    loc=df_hs["beta_mean"],
    scale=df_hs["beta_sd"]
)

df_hs["volcano_beta"] = np.where(
    df_hs["beta_mean"] >= 0,
    -np.log10(1 - df_hs["p_beta_gt_0"]),
    -np.log10(df_hs["p_beta_gt_0"])
)

# same scale volcano
plt.figure(figsize=(10, 7))

# EWAS volcano
plt.scatter(
    df["gamma_mean"],
    df["volcano_y"],
    c="blue",
    s=10,
    alpha=0.5,
    label="EWAS"
)

# Horseshoe volcano
plt.scatter(
    df_hs["beta_mean"],
    df_hs["volcano_beta"],
    c="black",
    s=10,
    alpha=0.5,
    label="Horseshoe"
)

# Highlight top 1% horseshoe CpGs
hs_sig = df_hs[df_hs["beta_abs"] > np.percentile(df_hs["beta_abs"], 99)]
plt.scatter(
    hs_sig["beta_mean"],
    hs_sig["volcano_beta"],
    c="red",
    s=15,
    label="Top 1% Horseshoe CpGs"
)

plt.xlabel("Effect size")
plt.ylabel("-log10 posterior tail probability")
plt.title("Unified Volcano: EWAS vs Horseshoe")
plt.legend()
plt.tight_layout()
plt.savefig("overlay_(no_line)_EWAS_horseshoe_volcano_unified.svg", dpi=300)
plt.close()

# now for unified manhattan
plt.figure(figsize=(14,7))

# EWAS Manhattan
plt.scatter(
    df["posi_cumu"],
    df["volcano_y"],
    c="blue",
    s=8,
    alpha=0.5,
    label="EWAS"
)

# Horseshoe Manhattan
plt.scatter(
    df_hs["posi_cumu"],
    df_hs["volcano_beta"],
    c="black",
    s=8,
    alpha=0.5,
    label="Horseshoe"
)

# Highlight top 1% horseshoe CpGs
plt.scatter(
    hs_sig["posi_cumu"],
    hs_sig["volcano_beta"],
    c="red",
    s=12,
    label="Top 1% Horseshoe CpGs"
)

plt.xlabel("Genomic position")
plt.ylabel("-log10 posterior tail probability")
plt.title("Unified Manhattan: EWAS vs Horseshoe")
plt.legend()
plt.tight_layout()
plt.savefig("overlay_(no_line)_EWAS_horseshoe_manhattan_unified.svg", dpi=300)
plt.close()
