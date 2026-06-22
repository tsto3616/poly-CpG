import pandas as pd
import numpy as np
import pymc as pm

# data loading
df_base = pd.read_csv("CSL_with_Bayesian_residuals.csv")
df_base = df_base.dropna()

cpg_cols = [c for c in df_base.columns if c.startswith("cg")]

M = df_base[cpg_cols]

annot = pd.read_csv("CpG_annotations.csv")
annot = annot.set_index("CpG")

# keep only CpGs that exist in M
annot = annot.loc[annot.index.intersection(M.columns)]

# reorder to match M
annot = annot.loc[M.columns]

# horseshoe
M_std = (M.values - M.values.mean(0)) / M.values.std(0)
y = df_base["MaxADC"].values.astype("float32")

with pm.Model() as hs_model:
    tau = pm.HalfCauchy("tau", 1.0)
    lam = pm.HalfCauchy("lam", 1.0, shape=M_std.shape[1])
    sigma = pm.HalfNormal("sigma", 5.0)

    beta = pm.Normal("beta", 0, tau * lam, shape=M_std.shape[1])
    alpha = pm.Normal("alpha", 0, 10)

    mu = alpha + pm.math.dot(M_std, beta)
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    approx = pm.fit(30000, method="advi")
    idata_hs = approx.sample(2000)


# work out the summary statistics
beta_post = idata_hs.posterior["beta"]  # dims: chain, draw, CpG
beta_mean = beta_post.mean(dim=("chain","draw")).values
beta_sd   = beta_post.std(dim=("chain","draw")).values
beta_abs  = np.abs(beta_mean)

annot = annot.rename_axis("CpG").reset_index()

# create a dtaafrma for it
df_hs = pd.DataFrame({
    "CpG": M.columns,
    "beta_mean": beta_mean,
    "beta_sd": beta_sd,
    "beta_abs": beta_abs,
}).merge(annot, on="CpG") # note renamed CGid to CpG

# rename to fit conversion with other data
df_hs = df_hs.rename(columns={"seqnames": "chr", "probeStart": "pos"})

df_hs = df_hs.sort_values(["chr", "pos"])
df_hs["chr"] = df_hs["chr"].astype(str)

# cumulative genomic position - like oter data
df_hs["posi_cumu"] = df_hs.groupby("chr")["pos"].transform(
    lambda x: x + x.min()
)

# export to csv
df_hs.to_csv("horseshoe_effect_sizes.csv", index=False)
print("Saved horseshoe effect sizes to horseshoe_effect_sizes.csv")
