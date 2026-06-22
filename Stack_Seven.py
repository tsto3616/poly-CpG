# this stack is for the main focus of the paper - the sliding window method
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import linregress
from scipy.stats import norm

# load the gene-gff3 file from NCBI (GCA_009762295.2)
gff = pd.read_csv(
    "genes.gff3.gz",
    sep="\t",
    comment="#",
    header=None,
    names=["chr","src","type","start","end","score","strand","phase","attr"]
)

# bwlow is to parse the gff3 file through python - this line extracts only gene level data
genes_annot = gff[gff["type"] == "gene"].copy()

# these lines accomodate for multiple differences in the gff3 structure that can cause issues
genes_annot["gene_symbol"] = genes_annot["attr"].str.extract("gene_name=([^;]+)")
genes_annot["gene_symbol"] = genes_annot["attr"].str.extract("gene_symbol=([^;]+)")
genes_annot["gene_symbol"] = genes_annot["attr"].str.extract("Name=([^;]+)")

# replace "chr" with the appropriate term)
# if at any point need to rename a variable just do:
# df["variable_new"] = df["variable_old"]

# this is also for renaming of the chromosome to prevent any structural naming issues
genes_annot["chr"] = (
    genes_annot["chr"]
    .astype(str)
    .str.replace('"', '', regex=False)
    .str.replace("chr", "", regex=False)
    .str.strip()
)

# just to check that the result is ok
print(genes_annot["gene_symbol"].head(20))

# TSS - TSS is the transcription start site
genes_annot["tss"] = np.where(
    genes_annot["strand"] == "+",
    genes_annot["start"],
    genes_annot["end"]
)

# basal regulatory domain - important in the GREAT method
genes_annot["reg_start"] = genes_annot["tss"] - 5000
genes_annot["reg_end"]   = genes_annot["tss"] + 1000

# sort by chr, tss
genes_annot = genes_annot.sort_values(["chr", "tss"])

# extend to midpoints between adjacent genes - so as to prevent the uneven assignment of CpGs to 
# particular genes 
for chr_name, df_chr in genes_annot.groupby("chr"):
    idxs = df_chr.index.to_list()
    tss = df_chr["tss"].values
    if len(tss) > 1:
        mids = (tss[:-1] + tss[1:]) / 2
        genes_annot.loc[idxs[0], "reg_end"] = mids[0]
        for i in range(1, len(df_chr)-1):
            genes_annot.loc[idxs[i], "reg_start"] = mids[i-1]
            genes_annot.loc[idxs[i], "reg_end"]   = mids[i]
        genes_annot.loc[idxs[-1], "reg_start"] = mids[-1]

# --- phenotype + methylation matrix ---
df_base = pd.read_csv("CSL_HDBSCAN_with_LatentAge_and_weights.csv", index_col=False)

# FIX THESE COLUMN NAMES:
sample_col  = "CSL_Samples"
adc_col     = "MaxADC"
cluster_col = "Cluster"  

# extract the MaxADC
y = df_base[adc_col].values

# Extract methylation matrix (all CpG columns)
# Assumes CpG columns start with "cg"
cpg_cols = [c for c in df_base.columns if c.startswith("cg")]
# --- EWAS γ ---
df_gamma = pd.read_csv("EWAS_ADVI_effect_sizes.csv") # Stack_Four.py

# --- Horseshoe β ---
df_hs = pd.read_csv("horseshoe_effect_sizes.csv") # Stack_Five.py

# --- CpG annotation ---
annot = pd.read_csv("CpG_annotations.csv") # retreived from the GitHub: 
# https://github.com/shorvath/MammalianMethylationConsortium

# CpG ID column is actually "CpG" inside the file
annot = annot.rename(columns={"CpG": "CpG_ID"})

# Extract gene symbol from GFF
genes_annot["gene_symbol"] = genes_annot["attr"].str.extract("Name=([^;]+)")

# Clean gene symbol
genes_annot["gene_symbol"] = genes_annot["gene_symbol"].str.upper().str.strip()

# sort the genes according to the Chromosome and the TSS
genes_annot = genes_annot.sort_values(["chr", "tss"])

# this is to ensure no computational errors, also to ensure that the apporproaite order and mid-
# point is selected.
genes_annot["mid"] = genes_annot["tss"].rolling(2).mean()
genes_annot["reg_start"] = genes_annot["reg_start"].astype(int)
genes_annot["reg_end"]   = genes_annot["reg_end"].astype(int)
genes_annot["reg_start"], genes_annot["reg_end"] = (
    genes_annot[["reg_start", "reg_end"]].min(axis=1),
    genes_annot[["reg_start", "reg_end"]].max(axis=1),
)
genes_annot.loc[idxs[0], "reg_end"] = mids[0]
genes_annot.loc[idxs[i], "reg_start"] = mids[i-1]

# Clean the CpG SYMBOL column 
annot["SYMBOL"] = annot["SYMBOL"].astype(str).str.split(";").str[0].str.upper().str.strip()

# Merge the genes, CpGs, gene locations, etc
annot_merged = annot.merge(
    genes_annot[["gene_symbol","chr","start","end","strand"]],
    left_on="SYMBOL",
    right_on="gene_symbol",
    how="inner"
)

# Normalize CpG chromosome names to match GFF (eg. 1, 2, 3, ..., X, Y)
annot_merged["chr"] = (
    annot_merged["chr"]
    .astype(str)
    .str.replace('"', '', regex=False)
    .str.replace("chr", "", regex=False)
    .str.strip()
)

# just making sure it accomodates the variables
annot_merged = annot_merged[annot_merged["chr"].str.match(r"^\d+$|^[XY]$")]

# Drop CpGs with no chromosome assignment
annot_merged = annot_merged.dropna(subset=["chr"])

# Normalize chromosome names
annot_merged["chr"] = annot_merged["chr"].astype(str).str.replace("chr","")

# affirm the direction of the annotations 
annot_merged["strand"] = np.where(
    annot_merged["start"] < annot_merged["end"],
    "+",
    "-"
)

# Remove CpGs with no chromosome or no genomic position
annot_merged = annot_merged.dropna(subset=["chr", "start", "end"])

# Compute CpG genomic position (including the direction)
annot_merged["cpg_pos"] = np.where(
    annot_merged["strand"] == "+",
    annot_merged["start"],
    annot_merged["end"]
)

# enables the structuring of the columns etc for the data - eg remove the CpG3647.10 to be CpG3647
annot_merged["CpG_ID"] = annot_merged["CpG_ID"].str.replace(r"\.\d+$", "", regex=True)

annot_sorted = annot_merged.sort_values(["chr", "cpg_pos"])
annot_cpgs = annot_sorted["CpG_ID"].tolist()
cpg_cols = [c for c in annot_cpgs if c in df_base.columns]

# 5. Build M (CpGs as rows)
M = df_base[cpg_cols].values
pos = (
    annot_sorted
    .set_index("CpG_ID")
    .loc[cpg_cols, "cpg_pos"]
    .astype(int)
    .values
)

# calculate the mean gamma and beta values from the results of ADVI-EWAS and horseshoe respectively
# the gamma and beta values are averages for the CpGs across the genes
gamma = df_gamma.set_index("CpG").loc[cpg_cols, "gamma_mean"].values
beta  = df_hs.set_index("CpG").loc[cpg_cols, "beta_mean"].values


# sliding window functions are applied to the analysis of the beta and gamma scores. Under the 
# first set of code the S_gamma/beta is predictive of the linear combination of the scores to form 
# the methylation-matrix for the gamma and or beta transformed matrix. The gamma[idx_window] score 
# is the individual effect sizes for the same CpGs - these are individualised to the CpGs not averaged 
# for the animals. The formula presented below is the matrix multiplication solution for which the 
# window gamma and beta are derived from. This is calculated for every animal. Each S_ score is a 
# vector for all the animals - defined by the length of n (animals sampled). It is used to calculate 
# the R^2 value later downstream.  
def window_scores(M, gamma, beta, idx_window):
    S_gamma = M[:, idx_window] @ gamma[idx_window]
    S_beta  = M[:, idx_window] @ beta[idx_window]
    return S_gamma, S_beta

# This function determines the significance as mediated by the R^2 value of the window, with each
# of the y values associated with the phenotype. The generated R^2 value was a function of the 
# phenotype and the window. In addition it calculates the direction of influence - enabling the 
# estimation of the relative importance AND the direction of importance. 
def r2_and_sign(S, y):
    S_c = S - S.mean()
    y_c = y - y.mean()
    num = np.dot(S_c, y_c)
    den = np.sqrt(np.dot(S_c, S_c) * np.dot(y_c, y_c))
    if den == 0:
        return 0.0, 0.0
    r = num / den
    return r**2, np.sign(r)

# this foruma caculates the self tuning window which dictates if the sliding window will continue 
# to expand or if it will be terminated - and retained or kept as a significant window. 

# the code creates a window for every CpG, it grows to the right adding CpGs in the order in which
# they arise in the genome. For each step the gamma/ beta weighted window, R^2 and the direction of
# of the influence of the window upon the phenotype is defined. 

# The code also penalises the window size as a function of the lambda_penalty - a future option for 
# tuning of the model for future directions. 

# it requires both the gamma and beta to improve upon the encoded R^2 value by a minimum of 0.005 
# (the min_r2_increase) and a minimum window size of four CpGs for the retainment of the window 
# following cestation. The process continues to grow until the model terminates by reaching the 
# maximum window size (49 CpGs upstream of the original CpG). 
def self_tuning_windows_dual(
    M, gamma, beta, pos, y,
    max_window_size=50,
    min_r2_increase=0.005,
    lambda_penalty=0.003,
    min_window_size=4
):
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

        # grow window to the right
        for end in range(start + 1, min(start + max_window_size, n_cpg)):
            # stop if chromosome changes
            if annot_sorted.iloc[end]["chr"] != annot_sorted.iloc[start]["chr"]:
                break

            idx_window.append(end)
            Sg, Sb = window_scores(M, gamma, beta, idx_window)
            r2_g, sign_g = r2_and_sign(Sg, y)
            r2_b, sign_b = r2_and_sign(Sb, y)

            score_g = r2_g - lambda_penalty * len(idx_window)
            score_b = r2_b - lambda_penalty * len(idx_window)

            improve_g = (score_g - best_score_g) > min_r2_increase
            improve_b = (score_b - best_score_b) > min_r2_increase
            same_direction = (sign_g == sign_b) and (sign_g != 0)

            # stricter criterion: both improve and agree in direction
            if improve_g and improve_b and same_direction:
                best_r2_g, best_r2_b = r2_g, r2_b
                best_sign_g, best_sign_b = sign_g, sign_b
                best_end = end
                best_score_g, best_score_b = score_g, score_b
            else:
                break

        # enforce minimum window size
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

# the windows are then classified for the R^2 penalised threshold for the significance of the CpG's
# gamma or beta predictor for the influence of the CpG upon the phentotype - windows must have a 
# collective score of 0.5 collective posterior probability for determination of if the window is 
# trustworthy or not!
def classify_windows(df_windows, r2_thresh=0.5):
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
    
    df_windows["status"] = None
    df_windows["status"] = df_windows.apply(classify, axis=1)
    df = df_windows.copy()

    
    return df

# this executes with window function as described earlier 
df_windows = self_tuning_windows_dual(
    M, gamma, beta, pos, y,
    max_window_size=50,
    min_r2_increase=0.005,
    lambda_penalty=0.003,
    min_window_size=4
)

print(df_windows.head(5))

# this provided the classification of the windows as trustworthy or not using a tunable threshold.
df_windows = classify_windows(df_windows, r2_thresh=0.5)
print(df_windows.head(5))

# now the code is used to annotate the genes
df_windows["chr"] = None
df_windows["chr"] = df_windows["start_idx"].apply(
    lambda i: annot_sorted.iloc[i]["chr"]
)

# Add CpG IDs for each window
df_windows["CpGs"] = None
df_windows["CpGs"] = df_windows.apply(
    lambda row: cpg_cols[row["start_idx"]: row["end_idx"] + 1],
    axis=1
)

df_windows["CpGs_str"] = None
df_windows["CpGs_str"] = df_windows["CpGs"].apply(lambda xs: ",".join(xs))

# sort for trustworthy windows
df_windows = df_windows.sort_values(
    ["status", "r2_gamma", "r2_beta"],
    ascending=[True, False, False]
)

# Now create trusted windows AFTER chr is correct
windows = df_windows[df_windows["status"] == "trusted"].drop_duplicates("CpGs_str")

# now for the assignment of genes to trustworthy windows only
# Recompute CpGs_str to be safe
df_windows["CpG_ID"] = df_windows.apply(
    lambda row: cpg_cols[row["start_idx"]: row["end_idx"] + 1],
    axis=1
)
df_windows["CpGs_str"] = df_windows["CpG_ID"].apply(lambda xs: ",".join(xs))

# Trusted + unique windows
windows = df_windows[df_windows["status"] == "trusted"].drop_duplicates("CpGs_str")

# Compute TSS
genes_annot["tss"] = np.where(
    genes_annot["strand"] == "+",
    genes_annot["start"],
    genes_annot["end"]
)

# Sort by chromosome and TSS
genes_annot = genes_annot.sort_values(["chr", "tss"]).reset_index(drop=True)

# Compute midpoints between adjacent genes
genes_annot["mid_prev"] = genes_annot["tss"].rolling(2).mean()
genes_annot["mid_next"] = genes_annot["tss"].shift(-1).rolling(2).mean()

# Regulatory domain boundaries
genes_annot["reg_start"] = genes_annot["tss"] - 5000
genes_annot["reg_end"]   = genes_annot["tss"] + 1000

# Extend to midpoints
genes_annot["reg_start"] = genes_annot[["reg_start", "mid_prev"]].max(axis=1)
genes_annot["reg_end"]   = genes_annot[["reg_end", "mid_next"]].min(axis=1)

# Fix types
genes_annot["reg_start"] = genes_annot["reg_start"].fillna(genes_annot["reg_start"]).astype(int)
genes_annot["reg_end"]   = genes_annot["reg_end"].fillna(genes_annot["reg_end"]).astype(int)

# 1. Build genomic positions vector
pos = (
    annot_sorted
    .set_index("CpG_ID")
    .loc[cpg_cols, "cpg_pos"]
    .astype(int)
    .values
)

# 2. Add genomic start/end to windows
df_windows["start_pos"] = df_windows["start_idx"].apply(lambda i: pos[i])
df_windows["end_pos"]   = df_windows["end_idx"].apply(lambda i: pos[i])

# 3. Recompute CpG lists (unchanged)
df_windows["CpG_ID"] = df_windows.apply(
    lambda row: cpg_cols[row["start_idx"]: row["end_idx"] + 1],
    axis=1
)
df_windows["CpGs_str"] = df_windows["CpG_ID"].apply(lambda xs: ",".join(xs))

df_windows["start_pos"] = df_windows["start_idx"].apply(lambda i: int(pos[i]))
df_windows["end_pos"]   = df_windows["end_idx"].apply(lambda i: int(pos[i]))

df_windows["start_pos"] = df_windows["start_pos"].astype(int)
df_windows["end_pos"]   = df_windows["end_pos"].astype(int)

# there was an abundance of repetitive windows due to the structure of the programming and so 
# a quick solution was to remove all non-unique 'trusted' windows. 
windows = (
    df_windows[df_windows["status"] == "trusted"]
    .drop_duplicates("CpGs_str")
    .reset_index(drop=True)
)

# this plots the genes in the context of the chromosomes
valid_chrs = set(genes_annot["chr"].unique())
annot_sorted = annot_sorted[annot_sorted["chr"].isin(valid_chrs)]

print(windows[["chr","start_pos","end_pos"]].head(20))

gene_to_windows = {}

# this produces the gene translation onto the CpG windows. 
for w_idx, row in windows.iterrows():
    chr_w = row["chr"]
    start_w = row["start_pos"]
    end_w = row["end_pos"]

    hits = genes_annot[
        (genes_annot["chr"] == chr_w) &
        (genes_annot["reg_start"] <= end_w) &
        (genes_annot["reg_end"]   >= start_w)
    ]

    for g in hits["gene_symbol"].dropna():
        gene_to_windows.setdefault(g, []).append(w_idx)

# this is to confirm the results
print("WINDOW:", chr_w, start_w, end_w)
print("GENE CHR EXAMPLE:", genes_annot["chr"].iloc[0])
print("UNIQUE GENE CHR:", genes_annot["chr"].unique()[:10])
print("UNIQUE WINDOW CHR:", windows["chr"].unique()[:10])

# list of genes that have at least one window - using the GREAT method of annotation
window_genes = sorted(gene_to_windows.keys())
print("Number of GREAT-assigned genes:", len(window_genes))
rows = []

for gene, win_list in gene_to_windows.items():
    all_cpgs = []
    status_list = []

    for w in win_list:
        wrow = windows.iloc[w]

        # collect CpGs
        s = int(wrow["start_idx"])
        e = int(wrow["end_idx"])
        all_cpgs.extend(cpg_cols[s:e+1])

        # collect status
        if "status" in wrow:
            status_list.append(wrow["status"])
        else:
            status_list.append(None)

    rows.append({
        "gene": gene,
        "windows": win_list,
        "status": status_list,
        "CpGs": sorted(set(all_cpgs))
    })

gene_window_cpg_df = pd.DataFrame(rows)

# this is to reformat it to be suitable for exporting
gene_window_cpg_df["windows"] = gene_window_cpg_df["windows"].apply(lambda xs: ",".join(map(str, xs)))
gene_window_cpg_df["status"] = gene_window_cpg_df["status"].apply(lambda xs: ",".join(map(str, xs)))
gene_window_cpg_df["CpGs"] = gene_window_cpg_df["CpGs"].apply(lambda xs: ",".join(xs))

# Export
gene_window_cpg_df.to_csv("gene_windows_cpgs.csv", index=False)

# now for the construction of the gene-heatmatrix plot and per gene regression:
go_df = gene_window_cpg_df

# this cleans the dataframe for the application of downstream steps by splitting the
# CpG strings into somethign manageable
def clean_cpg_string(x):
    if not isinstance(x, str):
        return []
    return [
        c.strip().replace("'", "").replace('"', "").replace("[", "").replace("]", "")
        for c in x.split(",") if c.strip() != ""
    ]

# apply the command
go_df["CpGs_clean"] = go_df["CpGs"].apply(clean_cpg_string)

# this removed duplicated windows - as before just in case it was neglected - could inflate the 
# results
win_sets = [frozenset(cpgs) for cpgs in go_df["CpGs_clean"]]
seen = set()
unique_indices = []
for i, s in enumerate(win_sets):
    if s not in seen:
        seen.add(s)
        unique_indices.append(i)

go_df = go_df.iloc[unique_indices].reset_index(drop=True)

# Collect all the cleaned CpGs 
cpg_cols = sorted({cpg for lst in go_df["CpGs_clean"] for cpg in lst})

# Determine the effect size from the earlier stacks for the CpGs 
df_gamma = pd.read_csv("EWAS_ADVI_effect_sizes.csv").set_index("CpG")
gamma = df_gamma.loc[cpg_cols, "gamma_mean"].values

df_hs = pd.read_csv("horseshoe_effect_sizes.csv").set_index("CpG")
beta = df_hs.loc[cpg_cols, "beta_mean"].values

# obtain the clusters for the analysis too
df_base = pd.read_csv("CSL_HDBSCAN_with_LatentAge_and_weights.csv")
clusters = df_base["Cluster"].values
cluster_ids = np.unique(clusters)
M = df_base[cpg_cols].values

# Compute the poly-CpG scores for a measurable - unifying - outcome
poly = expit(((gamma + beta) / 2) * M)
poly_df = pd.DataFrame(poly, index=df_base.index, columns=cpg_cols)

# This scores the CpGs for which there were poly-CpGs
def score_from_cpgs(cpg_list):
    if len(cpg_list) == 0:
        return pd.Series(np.zeros(len(poly_df)), index=poly_df.index)
    return poly_df[cpg_list].mean(axis=1)

# Buil the gene scores from the CpG scores (per animal)
go_scores = pd.DataFrame({
    row["gene"]: score_from_cpgs(row["CpGs_clean"])
    for _, row in go_df.iterrows()
})

# build the window scores for the cleaned CpGs too (per animal)
window_scores = pd.DataFrame({
    f"win_{i}": score_from_cpgs(cpgs)
    for i, cpgs in enumerate(go_df["CpGs_clean"])
})

# Esimtate the cluster-gene relationship
cluster_go = pd.DataFrame({
    gene: [go_scores[gene][clusters == c].mean() for c in cluster_ids]
    for gene in go_scores.columns
}, index=cluster_ids)

# estimate the cluster-window relationship
cluster_windows = pd.DataFrame({
    win: [window_scores[win][clusters == c].mean() for c in cluster_ids]
    for win in window_scores.columns
}, index=cluster_ids)

# Construct the final matrices - this is the data for the construction of the aesthetics not the 
# statistics
mat_anim_go  = go_scores.values          # Animals × Genes
mat_anim_win = window_scores.values      # Animals × Windows
mat_clu_go   = cluster_go.values         # Clusters × Genes
mat_clu_win  = cluster_windows.values    # Clusters × Windows

n_anim, n_go  = mat_anim_go.shape
_,      n_win = mat_anim_win.shape
n_clu,  _     = mat_clu_go.shape

tile_w = 1.0
tile_h = 1.0

w_go  = n_go  * tile_w
w_win = n_win * tile_w
h_anim = n_anim * tile_h
h_clu  = n_clu  * tile_h

W = w_go + w_win
H = h_anim + h_clu

# This is for the figure and axis construction
fig, ax = plt.subplots(figsize=(28, 20))
fig.subplots_adjust(left=0.05, right=0.85, top=0.98, bottom=0.05)

ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.invert_yaxis()
ax.set_aspect("auto")

yellow_map = LinearSegmentedColormap.from_list(
    "yellow_map", ["#ffffe0", "#ffd700", "#ff8c00"]
)

# TOP‑LEFT: Animals × Genes
ax.imshow(
    mat_anim_go,
    cmap="Reds",
    aspect="auto",
    extent=[0, w_go, 0, h_anim]
)

# TOP‑RIGHT: Animals × Windows
ax.imshow(
    mat_anim_win,
    cmap="Blues",
    aspect="auto",
    extent=[w_go, W, 0, h_anim]
)

# BOTTOM‑LEFT: Clusters × Genes
ax.imshow(
    mat_clu_go,
    cmap="Greens",
    aspect="auto",
    extent=[0, w_go, h_anim, H]
)

# BOTTOM‑RIGHT: Clusters × Windows
ax.imshow(
    mat_clu_win,
    cmap=yellow_map,
    aspect="auto",
    extent=[w_go, W, h_anim, H]
)

# this aligns the axis labels to be cosnsistent despite the our separate heatmaps. 
# X‑axis: Genes (left) + Windows (right)
x_go_centers  = np.arange(n_go)  * tile_w + tile_w/2
x_win_centers = np.arange(n_win) * tile_w + tile_w/2 + w_go
ax.set_xticks(np.concatenate([x_go_centers, x_win_centers]))

go_labels     = list(go_scores.columns)
window_labels = list(window_scores.columns)
ax.set_xticklabels(go_labels + window_labels, rotation=90)

# Y‑axis: Animals (top) + Clusters (bottom)
y_anim_centers = np.arange(n_anim) * tile_h + tile_h/2
y_clu_centers  = np.arange(n_clu)  * tile_h + tile_h/2 + h_anim
ax.set_yticks(np.concatenate([y_anim_centers, y_clu_centers]))

animal_labels = df_base.iloc[:, 0].astype(str).tolist()
cluster_labels = list(cluster_go.index.astype(str))
ax.set_yticklabels(animal_labels + cluster_labels)

# THis is for the colour bars outside the heatmaps
# Reserve a vertical strip on the right for colourbars
fig.subplots_adjust(right=0.78)   # heatmap uses 0.05 → 0.78 of width - tune

# Reds (Animals × Genes)
cax1 = fig.add_axes([0.80, 0.10, 0.02, 0.18])
plt.colorbar(
    plt.cm.ScalarMappable(cmap="Reds"),
    cax=cax1
)

# Blues (Animals × Windows)
cax2 = fig.add_axes([0.83, 0.10, 0.02, 0.18])
plt.colorbar(
    plt.cm.ScalarMappable(cmap="Blues"),
    cax=cax2
)

# Greens (Clusters × Genes)
cax3 = fig.add_axes([0.80, 0.40, 0.02, 0.18])
plt.colorbar(
    plt.cm.ScalarMappable(cmap="Greens"),
    cax=cax3
)

# Yellow (Clusters × Windows)
cax4 = fig.add_axes([0.83, 0.40, 0.02, 0.18])
plt.colorbar(
    plt.cm.ScalarMappable(cmap=yellow_map),
    cax=cax4
)

# now save it
plt.savefig("Unified_4Quadrant_FINAL_axes.svg", dpi=300, bbox_inches="tight")
plt.close()

# now to work out the direction of effects - linear regression:
combined_effect = gamma + beta

gene_direction = {
    row["gene"]: np.mean([combined_effect[cpg_cols.index(c)] 
                          for c in row["CpGs_clean"]])
    for _, row in go_df.iterrows()
}

# Combined effect per CpG
combined_effect = gamma + beta

# Compute direction per gene
gene_direction = {}
for _, row in go_df.iterrows():
    cpgs = row["CpGs_clean"]
    if len(cpgs) == 0:
        gene_direction[row["gene"]] = 0
        continue
    vals = [combined_effect[cpg_cols.index(c)] for c in cpgs]
    gene_direction[row["gene"]] = np.mean(vals)

# Adaptive threshold for the determination fo direction 
effects = np.array(list(gene_direction.values()))
mad = np.median(np.abs(effects - np.median(effects)))
threshold = 0.5 * mad

def direction_label(x, threshold):
    if x > threshold:
        return "Up"
    elif x < -threshold:
        return "Down"
    else:
        return "Neutral"

# applying the gene direciton label
gene_direction_label = {
    g: direction_label(v, threshold)
    for g, v in gene_direction.items()
}

go_direction_df = pd.DataFrame({
    "Gene": list(gene_direction.keys()),
    "Effect": list(gene_direction.values()),
    "Direction": list(gene_direction_label.values())
})

# now this provides the dataframe of the outputs 
print(go_direction_df)

markers = ['o', 's', 'D', '^', 'v', 'P', 'X', '*']  # extend if needed
# Combined effect per CpG (same order as cpg_cols)
combined_effect = gamma + beta

# Build a per-animal effect matrix for the three genes
gene_list = ["GCSH", "DNMT1", "LMX1A"]
animal_gene_effects = {}

for gene in gene_list:
    cpgs = go_df.loc[go_df["gene"] == gene, "CpGs_clean"].values[0]
    idxs = [cpg_cols.index(c) for c in cpgs]
    # mean effect across CpGs for each animal
    animal_gene_effects[gene] = M[:, idxs].mean(axis=1)

plot_df = pd.DataFrame({
    "MaxADC": df_base["MaxADC"],
    "GCSH": animal_gene_effects["GCSH"],
    "DNMT1": animal_gene_effects["DNMT1"],
    "LMX1A": animal_gene_effects["LMX1A"]
})

# Melt into long format for seaborn
plot_long = plot_df.melt(
    id_vars="MaxADC",
    value_vars=["GCSH", "DNMT1", "LMX1A"],
    var_name="Gene",
    value_name="Effect"
)

# now to plot it
plt.figure(figsize=(10, 7))

# now the regression 
sns.lmplot(
    data=plot_long,
    x="MaxADC",
    y="Effect",
    hue="Gene",
    height=7,
    aspect=1.2,
    scatter_kws={"s": 70, "edgecolor": "black"},
    line_kws={"linewidth": 2},
)

plt.xlabel("Max ADC (experimental)")
plt.ylabel("Per-Animal Gene Effect")
plt.title("MaxADC vs Gene Effect for GCSH, DNMT1, LMX1A")

plt.tight_layout()
plt.show()

# this is for the linear regression statistics accompanying the study of the genes and the maximum 
# aerobic diving capacity
# Compute stats per gene
for gene in plot_long["Gene"].unique():
    df_sub = plot_long[plot_long["Gene"] == gene]

    res = linregress(df_sub["MaxADC"], df_sub["Effect"])

    print(f"\n=== {gene} ===")
    print(f"Slope:       {res.slope:.4f}")
    print(f"Intercept:   {res.intercept:.4f}")
    print(f"R-squared:   {res.rvalue**2:.4f}")
    print(f"P-value:     {res.pvalue:.4e}")
    print(f"Std Error:   {res.stderr:.4f}")

# above prints a nice table!

# now for the simulation study!
# configuration
n_cpg = 200 # number of CpGs
region_start = 0 # chromosome
region_end = 100_000 # chromosme end

noise_levels = [0.5, 1.0, 2.0] 
sparsity_levels = [0.8, 0.4, 0.2]
conds = [(n, s) for n in noise_levels for s in sparsity_levels]

rng = np.random.default_rng(42)

# posterior probaility function/ derivative
def posterior_prob(x, sigma=0.05):
    return norm.cdf(x / sigma)

def significant_cpgs(gamma_est, beta_est, threshold=0.5):
    post_g = posterior_prob(gamma_est)
    post_b = posterior_prob(beta_est)
    return (post_g > threshold) & (post_b > threshold) & (np.sign(gamma_est) == np.sign(beta_est))

# Now for repretition of the sliding window analysis but for chromosome number of one 
# to prevent the computer code from crashing and allowing a simplier result/ interpretation
def window_scores(M, gamma, beta, idx_window):
    S_gamma = M[:, idx_window] @ gamma[idx_window]
    S_beta  = M[:, idx_window] @ beta[idx_window]
    return S_gamma, S_beta

def r2_and_sign(S, y):
    S_c = S - S.mean()
    y_c = y - y.mean()
    num = np.dot(S_c, y_c)
    den = np.sqrt(np.dot(S_c, S_c) * np.dot(y_c, y_c))
    if den == 0:
        return 0.0, 0.0
    r = num / den
    return r**2, np.sign(r)

def self_tuning_windows_dual(
    M, gamma, beta, pos, y,
    max_window_size=50,
    min_r2_increase=0.005,
    lambda_penalty=0.003,
    min_window_size=4
):
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
            "start_pos": int(pos[start]),
            "end_pos": int(pos[best_end]),
            "n_cpgs": n_cpgs,
            "r2_gamma": best_r2_g,
            "r2_beta": best_r2_b,
            "sign_gamma": best_sign_g,
            "sign_beta": best_sign_b,
        })

    return pd.DataFrame(results)

# Classify the windows through posterior probability as performed before
def classify_windows(df_windows, gamma_est, beta_est, threshold=0.5):
    post_g = posterior_prob(gamma_est)
    post_b = posterior_prob(beta_est)

    def classify(row):
        # If indices are missing or NaN → cannot classify → weak
        if pd.isna(row["start_idx"]) or pd.isna(row["end_idx"]):
            return "weak"

        # Convert to integers safely
        start = int(row["start_idx"])
        end   = int(row["end_idx"])

        # If conversion fails or window is invalid
        if start < 0 or end < start or end >= len(gamma_est):
            return "weak"

        idxs = range(start, end + 1)

        pg = post_g[list(idxs)].mean()
        pb = post_b[list(idxs)].mean()

        sg = np.sign(gamma_est[list(idxs)]).mean()
        sb = np.sign(beta_est[list(idxs)]).mean()
        same_sign = np.sign(sg) == np.sign(sb) and np.sign(sg) != 0

        if (pg > threshold) and (pb > threshold) and same_sign:
            return "trusted"
        elif (pg > threshold) or (pb > threshold):
            return "semi"
        else:
            return "weak"

    df = df_windows.copy()
    df["status"] = df.apply(classify, axis=1)
    return df


# This was peformed for the simulation of the slidign window model for a similated study. 
def run_simulation(noise_sd, sparsity):

    positions = np.sort(rng.integers(region_start, region_end, size=n_cpg))

    n_effect = int(n_cpg * sparsity)
    chosen = rng.choice(np.arange(n_cpg), size=n_effect, replace=False)

    gamma_true = np.zeros(n_cpg)
    beta_true  = np.zeros(n_cpg)
    gamma_true[chosen] = 0.5
    beta_true[chosen]  = 0.5

    n_samples = 120
    M = rng.normal(0, 1, size=(n_samples, n_cpg))

    gamma = gamma_true + rng.normal(0, 0.05, size=n_cpg)
    beta  = beta_true  + rng.normal(0, 0.05, size=n_cpg)

    signal = M @ gamma_true
    noise  = rng.normal(0, noise_sd, size=n_samples)
    y = signal + noise

    df_windows = self_tuning_windows_dual(
        M, gamma, beta, positions, y,
        max_window_size=50,
        min_r2_increase=0.005,
        lambda_penalty=0.003,
        min_window_size=4
    )

    if df_windows.empty:
        df_windows = pd.DataFrame(columns=[
            "start_pos", "end_pos", "n_cpgs",
            "r2_gamma", "r2_beta",
            "sign_gamma", "sign_beta",
            "status", "mid"
        ])

        return {
            "recovered": 0,
            "df_windows": df_windows,
            "positions": positions,
            "gamma_true": gamma_true,
            "beta_true": beta_true,
            "gamma_est": gamma,
            "beta_est": beta
        }

    df_windows["start_idx"] = df_windows["start_idx"].astype("Int64")
    df_windows["end_idx"]   = df_windows["end_idx"].astype("Int64")

    df_windows = classify_windows(df_windows, gamma, beta, threshold=0.5)

    sig_mask = significant_cpgs(gamma, beta)
    recovered_count = 0

    for _, row in df_windows.iterrows():
        if row["status"] != "trusted":
            continue
        window_cpgs = np.where((positions >= row["start_pos"]) &
                               (positions <= row["end_pos"]))[0]
        if sig_mask[window_cpgs].any():
            recovered_count += 1

    required_cols = [
        "start_pos", "end_pos", "n_cpgs",
        "r2_gamma", "r2_beta",
        "sign_gamma", "sign_beta",
        "status", "mid"
    ]

    for col in required_cols:
        if col not in df_windows.columns:
            df_windows[col] = np.nan

    return {
        "recovered": recovered_count,
        "df_windows": df_windows[required_cols],
        "positions": positions,
        "gamma_true": gamma_true,
        "beta_true": beta_true,
        "gamma_est": gamma,
        "beta_est": beta
    }

# Now use the results to generate a sparsity-noise grid.
results = {}
for noise, sparsity in conds:
    results[(noise, sparsity)] = run_simulation(noise, sparsity)

# heatmap was constructed for the recovered genes - genes for which there was a match of at 
# one significant CpG in a window
heatmap = np.zeros((3, 3))
for i, noise in enumerate(noise_levels):
    for j, sparsity in enumerate(sparsity_levels):
        heatmap[i, j] = results[(noise, sparsity)]["recovered"]

# now for the basic heat map
plt.figure(figsize=(7, 5))
plt.imshow(heatmap, cmap="viridis")
for i in range(3):
    for j in range(3):
        plt.text(j, i, int(heatmap[i, j]), ha="center", va="center", color="white")
plt.xticks(range(3), sparsity_levels)
plt.yticks(range(3), noise_levels)
plt.xlabel("Sparsity")
plt.ylabel("Noise")
plt.title("Recovered Trusted Windows (Posterior-Based)")
plt.colorbar()
plt.tight_layout()
plt.show()

# The circos plot was constructed for further investigation. This code didnt involve any math just
# the aethetics/ graphical appearances.
def plot_unified_circular_genome(results):

    positions = results[(0.5, 0.8)]["positions"]
    angles = 2 * np.pi * (positions - positions.min()) / (positions.max() - positions.min())

    r_cpg1 = 1.00
    r_win1 = 1.15
    r_cpg2 = 1.30
    r_win2 = 1.45
    r_cpg3 = 1.60
    r_win3 = 1.75

    cpg_radii = [r_cpg1, r_cpg2, r_cpg3]
    win_radii = [r_win1, r_win2, r_win3]

    ordered_conds = [(0.5,0.8), (1.0,0.4), (2.0,0.2)]

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    ax.set_title("Posterior-Based CpG/Window Tracks", fontsize=16, weight="bold", pad=25)

    for (noise, sparsity), r_cpg, r_win in zip(ordered_conds, cpg_radii, win_radii):

        res = results[(noise, sparsity)]
        sig_mask = significant_cpgs(res["gamma_est"], res["beta_est"])

        colors = ["red" if s else "black" for s in sig_mask]
        ax.scatter(angles, np.full_like(angles, r_cpg), c=colors, s=14, alpha=0.9)

        dfw = res["df_windows"]

        theta = np.linspace(0, 2*np.pi, 2000)
        ax.plot(theta, np.full_like(theta, r_win), color="lightgrey", linewidth=2, alpha=0.6)

        for _, row in dfw.iterrows():
            if row["status"] != "trusted":
                continue

            start_angle = 2 * np.pi * (row["start_pos"] - positions.min()) / (positions.max() - positions.min())
            end_angle   = 2 * np.pi * (row["end_pos"]   - positions.min()) / (positions.max() - positions.min())

            window_cpgs = np.where((positions >= row["start_pos"]) &
                                   (positions <= row["end_pos"]))[0]
            is_tp = sig_mask[window_cpgs].any()

            color = "red" if is_tp else "black"
            ax.plot([start_angle, end_angle], [r_win, r_win], color=color, linewidth=3, alpha=0.95)

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(False)
    plt.tight_layout()
    plt.show()

plot_unified_circular_genome(results)

# The R^2 values were determined for the graphing of the values using the following code. 
def plot_r2_distributions(results):
    data = []
    for noise, sparsity in conds:
        dfw = results[(noise, sparsity)]["df_windows"]
        for _, row in dfw.iterrows():
            data.append({
                "r2_gamma": row["r2_gamma"],
                "r2_beta": row["r2_beta"],
                "noise": noise,
                "sparsity": sparsity
            })

    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.violinplot(data=df, x="sparsity", y="r2_gamma", hue="noise")
    plt.title("R² (EWAS) Distributions")

    plt.subplot(1, 2, 2)
    sns.violinplot(data=df, x="sparsity", y="r2_beta", hue="noise")
    plt.title("R² (Horseshoe) Distributions")

    plt.tight_layout()
    plt.show()

plot_r2_distributions(results)

# The gamma plot was used to model the recovery of the gamma values for the comparisons
# between the estimated and observed gamma values. 
def plot_gamma_recovery(results):
    data = []
    for noise, sparsity in conds:
        res = results[(noise, sparsity)]
        for g_true, g_est in zip(res["gamma_true"], res["gamma_est"]):
            data.append({
                "gamma_true": g_true,
                "gamma_est": g_est,
                "noise": noise,
                "sparsity": sparsity
            })

    df = pd.DataFrame(data)

    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=df, x="gamma_true", y="gamma_est", hue="noise", style="sparsity")
    plt.title("γ true vs γ estimated")
    plt.xlabel("True γ")
    plt.ylabel("Estimated γ")
    plt.tight_layout()
    plt.show()

plot_gamma_recovery(results)

# The same was repeated as before but with the beta values (Horseshoe)
def plot_beta_recovery(results):
    data = []
    for noise, sparsity in conds:
        res = results[(noise, sparsity)]
        for b_true, b_est in zip(res["beta_true"], res["beta_est"]):
            data.append({
                "beta_true": b_true,
                "beta_est": b_est,
                "noise": noise,
                "sparsity": sparsity
            })

    df = pd.DataFrame(data)

    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=df, x="beta_true", y="beta_est", hue="noise", style="sparsity")
    plt.title("β true vs β estimated")
    plt.xlabel("True β")
    plt.ylabel("Estimated β")
    plt.tight_layout()
    plt.show()

plot_beta_recovery(results)

# This code generates the summary code for Table 2 within the Manuscipt
def compute_summary_table(results):
    rows = []

    key_conditions = [
        (0.5, 0.8),
        (1.0, 0.4),
        (2.0, 0.2)
    ]

    for noise, sparsity in key_conditions:
        res = results[(noise, sparsity)]
        dfw = res["df_windows"]

        sig_mask = significant_cpgs(res["gamma_est"], res["beta_est"])
        n_sig = sig_mask.sum()

        trusted = dfw[dfw["status"] == "trusted"]
        n_trusted = len(trusted)

        recovered = 0
        for _, row in trusted.iterrows():
            window_cpgs = np.where(
                (res["positions"] >= row["start_pos"]) &
                (res["positions"] <= row["end_pos"])
            )[0]
            if sig_mask[window_cpgs].any():
                recovered += 1

        precision = recovered / n_trusted if n_trusted > 0 else 0

        median_size = trusted["n_cpgs"].median() if n_trusted > 0 else 0
        median_r2 = trusted["r2_gamma"].median() if n_trusted > 0 else 0

        rows.append({
            "noise": noise,
            "sparsity": sparsity,
            "sig_cpgs": n_sig,
            "trusted_windows": n_trusted,
            "recovered_windows": recovered,
            "precision": precision,
            "median_window_size": median_size,
            "median_r2": median_r2
        })

    return pd.DataFrame(rows)

summary_df = compute_summary_table(results)
print(summary_df)

# below is for the mapping of the supplementary figure 1!
# this annotated the window's genomic locations and positioning 
def reconstruct_window_trace_genomic(start_idx, end_idx, M, gamma, beta, y, pos):
    idxs = list(range(start_idx, end_idx + 1))
    rows = []

    for k in range(1, len(idxs) + 1):
        sub = idxs[:k]

        Sg, Sb = window_scores(M, gamma, beta, sub)
        r2_g, sign_g = r2_and_sign(Sg, y)
        r2_b, sign_b = r2_and_sign(Sb, y)

        rows.append({
            "pos_bp": pos[idxs[k-1]],   # genomic coordinate of the k-th CpG
            "r2_advi": r2_g,
            "r2_hs": r2_b,
            "agree": (sign_g == sign_b) and (sign_g != 0)
        })

    return pd.DataFrame(rows)

# now to encode the gene hits for the window and its genomic dataframe from advanced plotting and
# simulation. 
def get_gene_window_df_genomic(gene_name):
    if gene_name not in gene_to_windows:
        raise ValueError(f"No window found for gene {gene_name}")

    w_idx = gene_to_windows[gene_name][0]
    wrow = windows.iloc[w_idx]

    return reconstruct_window_trace_genomic(
        int(wrow["start_idx"]),
        int(wrow["end_idx"]),
        M, gamma, beta, y, pos
    )

# repeat for each gene to move towards plotting
df_gcsh  = get_gene_window_df_genomic("GCSH")
df_dnmt1 = get_gene_window_df_genomic("DNMT1")
df_lmx1a = get_gene_window_df_genomic("LMX1A")

# now for the (plotting of the window of growth)
def plot_window_growth_genomic(df, gene_name, ax):

    # ADVI and Horseshoe curves
    sns.lineplot(data=df, x='pos_bp', y='r2_advi',
                 marker='o', color='blue', label='ADVI', ax=ax)
    sns.lineplot(data=df, x='pos_bp', y='r2_hs',
                 marker='o', color='red', label='Horseshoe', ax=ax)

    # Agreement shading (between CpGs)
    for i in range(len(df)-1):
        if df.loc[i, "agree"]:
            x0 = df.loc[i, "pos_bp"]
            x1 = df.loc[i+1, "pos_bp"]
            ax.axvspan(x0, x1, color='grey', alpha=0.15)

    # Stopping point (max ADVI R²)
    stop_idx = df["r2_advi"].idxmax()
    stop_bp = df.loc[stop_idx, "pos_bp"]
    ax.axvline(stop_bp, color='black', linestyle='--', linewidth=1.5)

    ax.set_title(gene_name)
    ax.set_xlabel("Genomic coordinate (bp)")
    ax.set_ylabel("Penalised R²")

# this forms the start of the Supplementary Figure 1 plot
fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=False)

plot_window_growth_genomic(df_gcsh,  "GCSH",  axes[0])
plot_window_growth_genomic(df_dnmt1, "DNMT1", axes[1])
plot_window_growth_genomic(df_lmx1a, "LMX1A", axes[2])

plt.tight_layout()
plt.show() # this forms the end of the Supplementary Figure 1 plot
