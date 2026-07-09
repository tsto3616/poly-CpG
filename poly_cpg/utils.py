import pandas as pd
import importlib.resources as pkg_resources

def load_example():
    import poly_cpg.data as data_pkg

    def load_csv(name):
        with pkg_resources.open_text(data_pkg, name) as f:
            return pd.read_csv(f)

    df_meth = load_csv("synthetic_methylation.csv")
    df_pos  = load_csv("synthetic_positions.csv")
    df_gam  = load_csv("synthetic_gamma.csv")
    df_bet  = load_csv("synthetic_beta.csv")
    df_y    = load_csv("synthetic_phenotype.csv")

    M = df_meth.values
    pos = df_pos["pos"].values
    gamma = df_gam["gamma"].values
    beta = df_bet["beta"].values
    y = df_y["phenotype"].values
    cpg_names = df_meth.columns.tolist()

    return M, pos, gamma, beta, y, cpg_names
