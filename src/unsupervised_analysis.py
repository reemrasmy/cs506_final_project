import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# This function is splitting the features and labels of the dataset:
    #   1. Allows us to operate on the gene expression counts mathematically (expression_matric)
    #   2. Extracts the labels (subtypes)
def split_features_labels(merged_df):
    expression_matrix = merged_df.drop(columns=["sampleID", "subtype"])
    subtypes = merged_df["subtype"]

    return expression_matrix, subtypes


def variance_filtering(exp_data, n_genes=5000):
    # Computing the variance column-wise: the variance for each gene across all samples
    gene_variances = exp_data.var(axis=0)
    # sorting the genes by variance and taking the genes with the top variance as features
    top_genes = gene_variances.sort_values(ascending=False).head(n_genes).index

    # subset and return the expression data matrix for genes with top variance
    filtered_expression = exp_data[top_genes]

    return filtered_expression

merged_expr = pd.read_csv("data/processed_brca_data.csv", index_col=0)
expression_data, subtypes = split_features_labels(merged_expr)
# n_genes = 10000 --- created more noise, clusters were less accurate
filtered_genes = variance_filtering(expression_data, n_genes=5000)


############################# Apply PCA KMeans #############################

# Applying PCA to reduce high dimensional gene expression data
# Important: PCA is performed on gene expression data only (no label information), and subtype labels are used only for coloring the plot.
classes = ["Basal", "Her2", "LumA", "LumB"]

color_map = {
    "Basal": "red",
    "Her2": "blue",
    "LumA": "green",
    "LumB": "orange"
}


pca = PCA(n_components=50)
X_pca = pca.fit_transform(filtered_genes)


############################# Testing KMeans #############################

# Plotting k-means, k=4 to see overall separability and similarity
kmeans = KMeans(n_clusters=4, random_state=0)
clusters = kmeans.fit_predict(X_pca)

plt.figure(figsize=(12,5))

# True labels (same colors as above)
plt.subplot(1,2,1)
for cls in classes:
    mask = (subtypes == cls)
    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=cls,
        color=color_map[cls],
        alpha=0.7
    )

plt.title("True Breast Cancer Subtypes (PC1 vs. PC2)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()


# KMeans clusters (separate color scheme)
cluster_colors = ["orange", "blue", "red", "green"]

plt.subplot(1,2,2)
for i in range(4):
    mask = (clusters == i)
    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=f"Cluster {i}",
        color=cluster_colors[i],
        alpha=0.7
    )

plt.title("KMeans Clusters (PC1 vs. PC2)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()

plt.tight_layout()
plt.show()

# Unsupervised K-means (k=4) clustering shows some natural biological separation exists between breast cancer subtypes but fails to fully recover the known classifications/ subtypes.

# Reflecting Basal tumors are very distinct biologically, Luminal A & B are very smilar, and HER2 can be heterogenous
ari = adjusted_rand_score(subtypes, clusters)
print("Adjusted Random Index:", ari)            # 0.40366


