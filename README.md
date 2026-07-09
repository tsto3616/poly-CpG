# poly-CpG: Sliding window analysis of methylomes
This is a lightweight package to accompany the sliding window method for the analysis of methylomes. 

## Additional scripts and data can be found in the "Supplementary" branch. Herein the data and code used within the study are provided.

The package call be installed as:

```
pip install poly_cpg
```

The program can quickly be performed using the commands: 

```
import poly_cpg as pc

# Load synthetic example data
M, pos, gamma, beta, y, cpg_names = pc.load_example()

# Run sliding-window algorithm
df_windows = pc.self_tuning_windows_dual(M, gamma, beta, pos, y)

# Classify windows
df_windows = pc.classify_windows(df_windows)

print(df_windows.head())

```

This produces a table of windows with the genomic coordinates, number of CpGs, R^2 values, directional agreement and trust classificaiton 

The script folder contains the necessary script for the synthetic data generation. 
