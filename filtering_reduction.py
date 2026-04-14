from data_prep import *
import numpy as np
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


expression_data, subtypes = split_features_labels(merged_expr)
# n_genes = 10000 --- created more noise, clusters were less accurate
filtered_genes = variance_filtering(expression_data, n_genes=5000)

############################# Testing PCA #############################
pca = PCA(n_components=50)
X_pca = pca.fit_transform(filtered_genes)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=subtypes.factorize()[0], alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Gene Expression")
plt.show()

############################# Testing KMeans #############################
kmeans = KMeans(n_clusters=4, random_state=0)
clusters = kmeans.fit_predict(X_pca)

plt.figure(figsize=(12,5))

# True labels
plt.subplot(1,2,1)
# going through each subtype & finding the samples belonging to each subtype
for subtype in subtypes.unique():
    mask = subtypes == subtype
    # plotting samples for the currect subtype
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=subtype, alpha=0.7)
plt.title("True Subtypes")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()

# K-means clusters
plt.subplot(1,2,2)
# plot all the samples but color them by k-means cluster (c=clusters)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, alpha=0.7)
plt.title("K-means Clusters")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.tight_layout()
plt.show()