# Stack One - data frame construction and HDBSCAN 
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import hdbscan

# first have to download the pinniped-samples - hand curated - spreadsheet for the GEO samples 
# of only the california sea lion blood samples
samples_df = pd.read_excel("pinniped-samples.xlsx")
samples_df["CSL_Samples"] = samples_df["CSL_Samples"].astype(str)

sample_ids = set(samples_df["CSL_Samples"])

# the norm_methylation_pinniped.csv file was obtained from the GEO website for the accession: 
# GSE227319
meth_df = pd.read_csv("norm_methylation_pinnipeds.csv")

# Identify CpG rows (CGid column)
if "CGid" not in meth_df.columns:
    raise ValueError("Expected a column named 'CGid' in the methylation file.")

# Identify sample columns that match the Excel sample IDs
sample_cols = [col for col in meth_df.columns if col in sample_ids]

# Subset to CpG ID + matching sample columns
subset_df = meth_df[["CGid"] + sample_cols].copy()

# remove all CpGs with no variance or all 0 values 
numeric_part = subset_df[sample_cols].apply(pd.to_numeric, errors="coerce")

# Drop rows where all sample values are zero or NA
nonzero_mask = (numeric_part.sum(axis=1) != 0) & (~numeric_part.isna().all(axis=1))
subset_df = subset_df.loc[nonzero_mask].reset_index(drop=True)

# calculate the variance across the samples 
numeric_part = subset_df[sample_cols].apply(pd.to_numeric, errors="coerce")
subset_df["variance"] = numeric_part.var(axis=1)

# select only the first 10,000 most important/ variant CpGs 
top_10000 = subset_df.sort_values("variance", ascending=False).head(10000)

# Remove variance column
top_10000 = top_10000.drop(columns=["variance"])

# restructure the data so that it is earlier to manage
transposed = top_10000.set_index("CGid").T
transposed.index.name = "CSL_Samples"
transposed.reset_index(inplace=True)

# readd the age column to the samples and beta scores (note the file from GEO was beta scores not
# M scores which were used in downstream analyses).
merged = pd.merge(
    samples_df[["CSL_Samples", "Age"]],
    transposed,
    on="CSL_Samples",
    how="inner"
)

# the maximum aerobic diving capacity was calculated through the following formula (as published)
merged["MaxADC"] = 0.2357 * merged["Age"] + 1.3571

# and now save the csv file
merged.to_csv("CSL_master_table.csv", index=False)

# now load the df for the HDBSCAN clustering
df = merged 

# Identify CpG columns (everything except metadata)
meta_cols = ["CSL_Samples", "Age", "MaxADC"]
cpg_cols = [c for c in df.columns if c not in meta_cols]

# Extract beta values
beta = df[cpg_cols].apply(pd.to_numeric, errors="coerce")

# convert the beta values into M scores
beta = beta.clip(1e-6, 1 - 1e-6)
Mvals = np.log2(beta / (1 - beta))

# Rebuild dataframe: metadata + M-values
M = pd.concat([df[meta_cols], Mvals], axis=1)

# now reduce the dimensions through principal component analysis for the M scores (embedding process)
X = M[cpg_cols].values

pca = PCA(n_components=20)
PCs = pca.fit_transform(X)

for i in range(20):
    M[f"PC{i+1}"] = PCs[:, i]

# now for the HDBSCAN clustering
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=6,
    min_samples=3,
    metric="euclidean"
)

labels = clusterer.fit_predict(PCs)
probs = clusterer.probabilities_

M["Cluster"] = labels
M["ClusterProb"] = probs

# now for the saving of the csv for downstream analysis 
M.to_csv("CSL_clusters_PCA20_HDBSCAN.csv", index=False)