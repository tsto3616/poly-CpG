import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.multivariate.manova import MANOVA

# load the data
df = pd.read_csv("CSL_HDBSCAN_with_LatentAge_and_weights.csv")

Y = df["MaxADC"].values                  # aerobic diving capacity
LA = df["LatentAge"].values         # latent age
clusters = df["Cluster"].astype("category")
cluster_idx = clusters.cat.codes.values  # 0..K-1
w_raw = df["w_i"].values            # weights from stack 2

n_clusters = clusters.cat.categories.size

# Clip weights to avoid zeros (and negative values if any)
eps = 1e-6
w = np.clip(w_raw, eps, None)

# Bayesian model
# Y_i | mu_i, sigmoid^2, w_i ~ N(mu_i, sigmoid^2 / w_i)
# mu_i = alpha + beta_age * LatentAge_i + b_{c_i}

if __name__ == "__main__":
    with pm.Model() as model:

        # Priors
        alpha = pm.Normal("alpha", mu=0.0, sigma=10.0)          # α ~ N(0, 10^2)
        beta_age = pm.Normal("beta_age", mu=0.0, sigma=10.0)    # β_age ~ N(0, 10^2)

        tau = pm.HalfNormal("tau", sigma=5.0)                   # τ ~ HalfNormal(5)
        b_cluster = pm.Normal("b_cluster", mu=0.0, sigma=tau, shape=n_clusters)

        sigma = pm.HalfNormal("sigma", sigma=5.0)               # σ ~ HalfNormal(5)

        # Linear predictor
        mu = alpha + beta_age * LA + b_cluster[cluster_idx] # LA = the latent age

        # Weighted likelihood: variance sigma^2 / w_i → sd = sigma / sqrt(w_i)
        sd_i = sigma / pm.math.sqrt(w)
        Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sd_i, observed=Y)

        trace = pm.sample(
            draws=4000,
            tune=2000,
            target_accept=0.9,
            chains=4,
            random_seed=42
        )

    pm.summary(trace)
    # extract the posterior (Bayes_mu)
    posterior = pm.to_inference_data(trace)

    # Posterior mean of alpha, beta_age, b_cluster
    alpha_mean = posterior.posterior["alpha"].mean().values
    beta_mean = posterior.posterior["beta_age"].mean().values
    b_cluster_mean = posterior.posterior["b_cluster"].mean(dim=("chain","draw")).values

    # Compute posterior mean μ_i for each animal
    mu_mean = alpha_mean + beta_mean * LA + b_cluster_mean[cluster_idx]

    # Compute residuals
    residuals = Y - mu_mean # where the Y is the maximum aerobic diving capacity

    # ---------------------------------------------------------
    # Add residuals to dataframe and export
    # ---------------------------------------------------------
    df["Bayes_mu"] = mu_mean
    df["Bayes_residual"] = residuals

    df.to_csv("CSL_with_Bayesian_residuals.csv", index=False)

    print("Residuals saved to CSL_with_Bayesian_residuals.csv")
    print(df[["MaxADC", "Bayes_mu", "Bayes_residual"]].head())

# additional statistics performed (graph/ MANOVA): 
# Ensure cluster is categorical for colouring
df["Cluster"] = df["Cluster"].astype("category")

markers = ['o', 's', 'D', '^', 'v', 'P', 'X', '*']  # extend if needed

cluster_marker_map = {
    cid: markers[i % len(markers)]
    for i, cid in enumerate(df["Cluster"].cat.categories)
} # still to ensure the colouring

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=df,
    x="Bayes_mu",
    y="MaxADC",
    hue="Cluster",
    style="Cluster",
    palette="tab10",
    markers=cluster_marker_map,
    s=80,
    edgecolor="black"
)

# 1:1 line (perfect prediction)
plt.plot(
    [df["Bayes_mu"].min(), df["Bayes_mu"].max()],
    [df["Bayes_mu"].min(), df["Bayes_mu"].max()],
    linestyle="--",
    color="grey",
    linewidth=1
) # this is not linear regression just a perfect intercept and slope

plt.xlabel("Posterior Mean Prediction (Bayes_mu)")
plt.ylabel("Observed MaxADC")
plt.title("Bayesian Hierarchical Model: Predicted vs Observed MaxADC")

plt.tight_layout()
plt.show()

# And now for the MANOVA model - this is for stack 2/ the statistics on the HDBSCAN clusters being 
# produced by the Chronological and Epigenetic age
ma = MANOVA.from_formula(
    "Age_chrono + Age_epi ~ C(Cluster)",
    data=df
)

print(ma.mv_test())

# now to do a MANOVA on the predicted v actual aerobic dive capacity using the cluster:
# MANOVA model
ma = MANOVA.from_formula(
    "Bayes_mu + MaxADC ~ C(Cluster)",
    data=df
)

print(ma.mv_test())
