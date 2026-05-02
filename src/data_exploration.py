import pandas as pd
import matplotlib.pyplot as plt

genes_df = pd.read_csv("data/processed_brca_data.csv")

print("Expression Matrix Shape:", genes_df.shape)           # (821, 20,533) - 821 samples and ~ 20K genes
print("Subtype Distribution:\n", genes_df["subtype"].value_counts(), genes_df["subtype"].value_counts(normalize=True))
print("Missing values per column:\n", genes_df.isnull().sum().sum())    # 0
print("Basic Expression Stats:\n", genes_df.describe())

#
gene_variances = genes_df.drop(columns=["sampleID", "subtype"]).var()
print("Variance Across Genes:\n", gene_variances.describe())        # found many genes have 0 variance across samples (same expressiona across all samples)
num_zero_var = (gene_variances == 0).sum()
print("Number of zero-variance genes:", num_zero_var)       # 293 genes with 0 variance

# Subtype Class Imbalance visualization
genes_df["subtype"].value_counts().plot(kind="bar")
plt.title("Subtype Distribution")
plt.xlabel("Subtype")
plt.ylabel("Count")
plt.show()

