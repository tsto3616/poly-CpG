import numpy as np
import pandas as pd

def generate_synthetic_dataset(
    n_animals=40,
    n_cpgs=120,
    n_clusters=4,
    cluster_size=10,
    noise=0.2,
    seed=42,
    out_dir="poly-cpg/poly_cpg/data"
):
    rng = np.random.default_rng(seed)

    # 1. CpG positions (sorted)
    pos = np.sort(rng.integers(1_000, 500_000, size=n_cpgs))

    # 2. Clustered gamma/beta effects (guarantees windows)
    gamma = np.zeros(n_cpgs)
    beta  = np.zeros(n_cpgs)

    for c in range(n_clusters):
        start = c * cluster_size
        end   = start + cluster_size

        effect = rng.normal(1.0, 0.1)     # strong consistent effect
        gamma[start:end] = effect + rng.normal(0, 0.05, size=cluster_size)
        beta[start:end]  = effect + rng.normal(0, 0.05, size=cluster_size)

    # 3. Methylation matrix with correlated CpGs inside clusters
    M = rng.normal(0, 1, size=(n_animals, n_cpgs))
    for c in range(n_clusters):
        start = c * cluster_size
        end   = start + cluster_size
        cluster_signal = rng.normal(0, 1, size=n_animals)
        M[:, start:end] += cluster_signal[:, None] * 0.8

    # 4. Phenotype with strong signal
    y = M @ gamma + rng.normal(0, noise, size=n_animals)

    # 5. Save files with correct column names
    pd.DataFrame(M, columns=[f"CpG_{i}" for i in range(n_cpgs)]).to_csv(
        f"{out_dir}/synthetic_methylation.csv", index=False
    )

    pd.DataFrame({"CpG": [f"CpG_{i}" for i in range(n_cpgs)],
                  "pos": pos}).to_csv(
        f"{out_dir}/synthetic_positions.csv", index=False
    )

    pd.DataFrame({"CpG": [f"CpG_{i}" for i in range(n_cpgs)],
                  "gamma": gamma}).to_csv(
        f"{out_dir}/synthetic_gamma.csv", index=False
    )

    pd.DataFrame({"CpG": [f"CpG_{i}" for i in range(n_cpgs)],
                  "beta": beta}).to_csv(
        f"{out_dir}/synthetic_beta.csv", index=False
    )

    pd.DataFrame({"phenotype": y}).to_csv(
        f"{out_dir}/synthetic_phenotype.csv", index=False
    )

    print("Synthetic dataset generated successfully.")

generate_synthetic_dataset(n_animals=40,
    n_cpgs=120,
    n_clusters=4,
    cluster_size=10,
    noise=0.2,
    seed=42,
    out_dir="poly-cpg/poly_cpg/data")
