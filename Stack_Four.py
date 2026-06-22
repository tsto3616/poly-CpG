import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import norm

# load the data
df = pd.read_csv("CSL_with_Bayesian_residuals.csv")
df = df.dropna()

# sort the data
residuals   = df["Bayes_residual"].values.astype("float32")
LatentAge2  = df["LatentAge"].values.astype("float32")

df["Cluster"] = df["Cluster"].astype("category")
cluster_idx = df["Cluster"].cat.codes.values
K = df["Cluster"].nunique()

# clip the weights to prevent comoputational errors
w = np.clip(df["w_i"].values.astype("float32"), 1e-6, None)

# prepare the indivdual CpGs (in this case "cg")
cpg_cols = [c for c in df.columns if c.startswith("cg")]
M_full   = df[cpg_cols].values.astype("float32")

# develop the full matrix shape
N, J = M_full.shape

# now develop the batch size for which the analysis will be performed on (vital otherwise talking
# about 10,000 CpGs x 20,000 iterations and 2,000 posterior draws.) The ADVI method was used for 
# computational speeds. 
BATCH_SIZE = 1000
batches = [(i, min(i+BATCH_SIZE, J)) for i in range(0, J, BATCH_SIZE)]

results = []

# ---------------------------------------------------------
# Loop over CpG batches
# ---------------------------------------------------------
for start, end in batches:
    print(f"Processing CpGs {start}-{end}")

    M = M_full[:, start:end]
    cols = cpg_cols[start:end]
    J_batch = M.shape[1]

    with pm.Model() as ewas:

        # Priors
        mu_gamma  = pm.Normal("mu_gamma", 0, 10)
        tau_gamma = pm.HalfNormal("tau_gamma", 5)
        tau_b     = pm.HalfNormal("tau_b", 5)

        alpha    = pm.Normal("alpha", 0, 10, shape=J_batch)
        beta_age = pm.Normal("beta_age", 0, 10, shape=J_batch)

        b_cluster = pm.Normal("b_cluster", 0, tau_b, shape=(K, J_batch))
        gamma     = pm.Normal("gamma", mu_gamma, tau_gamma, shape=J_batch)
        sigma     = pm.HalfNormal("sigma", 5, shape=J_batch)

        # Linear predictor
        mu_ij = (
            alpha[None, :] +
            beta_age * LatentAge2[:, None] +
            b_cluster[cluster_idx, :] +
            gamma * M
        )

        sd_ij = sigma[None, :] / pm.math.sqrt(w)[:, None]

        Y_obs = pm.Normal(
            "Y_obs",
            mu=mu_ij,
            sigma=sd_ij,
            observed=np.tile(residuals[:, None], (1, J_batch))
        )

        # ADVI
        approx = pm.fit(
            n=20_000,                     # faster
            method="advi",
            progressbar=True
        )

        idata = approx.sample(2000)

        gamma_mean = idata.posterior["gamma"].mean(dim=("chain", "draw")).values
        gamma_sd   = idata.posterior["gamma"].std(dim=("chain", "draw")).values

        # Store results
        for cpg, gm, gs in zip(cols, gamma_mean, gamma_sd):
            results.append((cpg, gm, gs))

# ---------------------------------------------------------
# Save final results
# ---------------------------------------------------------
df = pd.DataFrame(results, columns=["CpG", "gamma_mean", "gamma_sd"])
df.to_csv("EWAS_ADVI_batched.csv", index=False)

print("Finished batched ADVI EWAS")

# now work out the 95% CIs 
df["gamma_lower"] = df["gamma_mean"] - 1.96 * df["gamma_sd"]
df["gamma_upper"] = df["gamma_mean"] + 1.96 * df["gamma_sd"]

# Computer the posterior using the assumption of a Gaussian variational posterior

df["p_gamma_gt_0"] = 1 - norm.cdf(0, loc=df["gamma_mean"], scale=df["gamma_sd"])

# save the final effect
df.to_csv("EWAS_ADVI_effect_sizes.csv", index=False)

print("Finished generating EWAS effect sizes without rerunning the model.")
