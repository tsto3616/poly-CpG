import pandas as pd
import numpy as np

def load_example(data_dir="poly_cpg/data"):
    """
    Load the synthetic example dataset packaged with poly-CpG.

    Returns:
        M (np.ndarray): methylation matrix (animals × CpGs)
        pos (np.ndarray): genomic positions of CpGs
        gamma (np.ndarray): ADVI-like effect sizes
        beta (np.ndarray): Horseshoe-like effect sizes
        y (np.ndarray): phenotype vector
        cpg_names (list): CpG IDs in correct order
    """

    df_meth = pd.read_csv(f"{data_dir}/synthetic_methylation.csv")
    df_pos  = pd.read_csv(f"{data_dir}/synthetic_positions.csv")
    df_gam  = pd.read_csv(f"{data_dir}/synthetic_gamma.csv")
    df_bet  = pd.read_csv(f"{data_dir}/synthetic_beta.csv")
    df_y    = pd.read_csv(f"{data_dir}/synthetic_phenotype.csv")

    # Ensure correct ordering
    df_pos = df_pos.sort_values("Position")
    cpg_names = df_pos["CpG"].tolist()

    # Reorder methylation and effects
    M = df_meth[cpg_names].values
    pos = df_pos["Position"].values
    gamma = df_gam.set_index("CpG").loc[cpg_names, "gamma"].values
    beta  = df_bet.set_index("CpG").loc[cpg_names, "beta"].values
    y = df_y["Phenotype"].values

    return M, pos, gamma, beta, y, cpg_names