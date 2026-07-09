import numpy as np
import pandas as pd

def generate_synthetic_dataset(
    n_animals=20,
    n_cpgs=50,
    noise=0.5,
    sparsity=0.4,
    seed=42,
    out_dir="."
):
    rng = np.random.default_rng(seed)

    # 1. CpG positions (single chromosome)
    positions = rng.integers(1_000, 200_000, size=n_cpgs)
    positions.sort()

    # 2. True gamma/beta effects
    n_signal = int(sparsity * n_cpgs)
    signal_idx = rng.choice(n_cpgs, n_signal, replace=False)

    gamma_true = np.zeros(n_cpgs)
    beta_true  = np.zeros(n_cpgs)

    gamma_true[signal_idx] = rng.normal(0.8, 0.2, size=n_signal)
    beta_true[signal_idx]  = rng.normal(0.8, 0.2, size=n_signal)

    # 3. Noisy estimates (what your sliding window uses)
    gamma_est = gamma_true + rng.normal(0, noise, size=n_cpgs)
    beta_est  = beta_true  + rng.normal(0, noise, size=n_cpgs)

    # 4. Methylation matrix
    M = rng.normal(0, 1, size=(n_animals, n_cpgs))

    # 5. Phenotype (linear combination + noise)
    y = M @ gamma_true + rng.normal(0, noise, size=n_animals)

    # 6. Save files
    pd.DataFrame(M, columns=[f"CpG_{i}" for i in range(n_cpgs)]).to_csv(
        f"{out_dir}/synthetic_methylation.csv", index=False
    )

    pd.DataFrame({"CpG": [f"CpG_{i}" for i in range(n_cpgs)],
                  "Position": positions}).to_csv(
        f"{out_dir}/synthetic_positions.csv", index=False
    )

    pd.DataFrame({"CpG": [f"CpG_{i}" for i in range(n_cpgs)],
                  "gamma": gamma_est}).to_csv(
        f"{out_dir}/synthetic_gamma.csv", index=False
    )

    pd.DataFrame({"CpG": [f"CpG_{i}" for i in range(n_cpgs)],
                  "beta": beta_est}).to_csv(
        f"{out_dir}/synthetic_beta.csv", index=False
    )

    pd.DataFrame({"Phenotype": y}).to_csv(
        f"{out_dir}/synthetic_phenotype.csv", index=False
    )

    print("Synthetic dataset generated successfully.")

generate_synthetic_dataset(    n_animals=20,
    n_cpgs=50,
    noise=0.5,
    sparsity=0.4,
    seed=42,
    out_dir="data")
