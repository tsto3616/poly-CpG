import numpy as np
import pandas as pd

def window_scores(M, gamma, beta, idx_window):
    """
    Compute gamma- and beta-weighted poly-CpG scores for a window.

    Parameters
    ----------
    M : np.ndarray
        Methylation matrix (animals x CpGs).
    gamma : np.ndarray
        ADVI-like effect sizes for each CpG.
    beta : np.ndarray
        Horseshoe-like effect sizes for each CpG.
    idx_window : list[int]
        Indices of CpGs included in the window.

    Returns
    -------
    S_gamma : np.ndarray
        Gamma-weighted poly-CpG score per animal.
    S_beta : np.ndarray
        Beta-weighted poly-CpG score per animal.
    """
    S_gamma = M[:, idx_window] @ gamma[idx_window]
    S_beta  = M[:, idx_window] @ beta[idx_window]
    return S_gamma, S_beta


def r2_and_sign(S, y):
    """
    Compute R² and direction of association between window score and phenotype.

    Parameters
    ----------
    S : np.ndarray
        Window score per animal.
    y : np.ndarray
        Phenotype vector.

    Returns
    -------
    r2 : float
        Coefficient of determination.
    sign : int
        Direction of association (+1, -1, or 0).
    """
    S_c = S - S.mean()
    y_c = y - y.mean()

    num = np.dot(S_c, y_c)
    den = np.sqrt(np.dot(S_c, S_c) * np.dot(y_c, y_c))

    if den == 0:
        return 0.0, 0

    r = num / den
    return r**2, np.sign(r)


def self_tuning_windows_dual(
    M, gamma, beta, pos, y,
    max_window_size=50,
    min_r2_increase=0.005,
    lambda_penalty=0.003,
    min_window_size=4
):
    """
    Self-tuning sliding window algorithm for poly-CpG detection.

    Parameters
    ----------
    M : np.ndarray
        Methylation matrix (animals x CpGs).
    gamma : np.ndarray
        ADVI-like effect sizes.
    beta : np.ndarray
        Horseshoe-like effect sizes.
    pos : np.ndarray
        Genomic positions of CpGs (sorted).
    y : np.ndarray
        Phenotype vector.
    max_window_size : int
        Maximum number of CpGs allowed in a window.
    min_r2_increase : float
        Minimum penalised R² improvement required to expand the window.
    lambda_penalty : float
        Penalisation factor for window size.
    min_window_size : int
        Minimum CpGs required for a valid window.

    Returns
    -------
    pd.DataFrame
        DataFrame containing window boundaries, R² values, and directions.
    """
    n_cpg = M.shape[1]
    results = []

    for start in range(n_cpg):
        idx_window = [start]

        Sg, Sb = window_scores(M, gamma, beta, idx_window)
        r2_g, sign_g = r2_and_sign(Sg, y)
        r2_b, sign_b = r2_and_sign(Sb, y)

        best_r2_g, best_r2_b = r2_g, r2_b
        best_sign_g, best_sign_b = sign_g, sign_b
        best_end = start

        best_score_g = best_r2_g - lambda_penalty * len(idx_window)
        best_score_b = best_r2_b - lambda_penalty * len(idx_window)

        # Expand window to the right
        for end in range(start + 1, min(start + max_window_size, n_cpg)):
            idx_window.append(end)

            Sg, Sb = window_scores(M, gamma, beta, idx_window)
            r2_g, sign_g = r2_and_sign(Sg, y)
            r2_b, sign_b = r2_and_sign(Sb, y)

            score_g = r2_g - lambda_penalty * len(idx_window)
            score_b = r2_b - lambda_penalty * len(idx_window)

            improve_g = (score_g - best_score_g) > min_r2_increase
            improve_b = (score_b - best_score_b) > min_r2_increase
            same_direction = (sign_g == sign_b) and (sign_g != 0)

            if improve_g and improve_b and same_direction:
                best_r2_g, best_r2_b = r2_g, r2_b
                best_sign_g, best_sign_b = sign_g, sign_b
                best_end = end
                best_score_g, best_score_b = score_g, score_b
            else:
                break

        n_cpgs = best_end - start + 1
        if n_cpgs < min_window_size:
            continue

        results.append({
            "start_idx": start,
            "end_idx": best_end,
            "start_pos": pos[start],
            "end_pos": pos[best_end],
            "n_cpgs": n_cpgs,
            "r2_gamma": best_r2_g,
            "r2_beta": best_r2_b,
            "sign_gamma": best_sign_g,
            "sign_beta": best_sign_b,
        })

    return pd.DataFrame(results)


def classify_windows(df_windows, r2_thresh=0.5):
    """
    Classify windows into trusted / semi / weak categories.

    Parameters
    ----------
    df_windows : pd.DataFrame
        Output from self_tuning_windows_dual().
    r2_thresh : float
        Minimum R^2 required for trust classification.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'status' column.
    """
    def classify(row):
        g_ok = row["r2_gamma"] > r2_thresh
        b_ok = row["r2_beta"] > r2_thresh
        same_sign = (row["sign_gamma"] == row["sign_beta"]) and (row["sign_gamma"] != 0)

        if g_ok and b_ok and same_sign:
            return "trusted"
        elif g_ok or b_ok:
            return "semi"
        else:
            return "weak"

    df = df_windows.copy()
    df["status"] = df.apply(classify, axis=1)
    return df
