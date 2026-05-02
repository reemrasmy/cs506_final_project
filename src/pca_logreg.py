import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def variance_filtering(exp_data, n_genes=5000):
    gene_variances = exp_data.var(axis=0)
    top_genes = gene_variances.sort_values(ascending=False).head(n_genes).index
    filtered_expression = exp_data[top_genes]
    return filtered_expression


########################### Load Data ###########################
gene_expr_df = pd.read_csv("data/processed_brca_data.csv")

# Drop index column if it exists
gene_expr_df = gene_expr_df.drop(columns=["Unnamed: 0"], errors="ignore")

X = gene_expr_df.drop(columns=["sampleID", "subtype"])
y = gene_expr_df["subtype"]


########################### Train/Test Split ###########################
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

########################### Variance-Based Gene Filtering ###########################
# Remove zero-variance genes using training set only
train_var = X_train.var()
keep_genes = train_var[train_var > 0].index

X_train_nonzero = X_train[keep_genes]
X_test_nonzero = X_test[keep_genes]

# Select top 5000 variable genes using training set only
X_train_filtered = variance_filtering(X_train_nonzero, n_genes=5000)
top_genes = X_train_filtered.columns

# Apply same selected genes to test set
X_test_filtered = X_test_nonzero[top_genes]


########################### Normalize Gene Expression Values ###########################
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_filtered)
X_test_scaled = scaler.transform(X_test_filtered)


########################### PCA Feature Reduction ###########################
# PCA for MODELING, not just visualization
pca_model = PCA(n_components=150)

X_train_pca_model = pca_model.fit_transform(X_train_scaled)
X_test_pca_model = pca_model.transform(X_test_scaled)

print("Variance explained by 150 PCs:", pca_model.explained_variance_ratio_.sum())


########################### Logistic Regression on PCA Features ###########################
model = LogisticRegression(max_iter=1000)
model.fit(X_train_pca_model, y_train)

y_pred = model.predict(X_test_pca_model)

print("\nClassification Report: Logistic Regression with PCA Features")
print(classification_report(y_test, y_pred))


########################### Visualize Outcome in 2D PCA Space ###########################
classes = ["Basal", "Her2", "LumA", "LumB"]

# Separate PCA just for visualization
pca_plot = PCA(n_components=2)
X_train_pca_plot = pca_plot.fit_transform(X_train_scaled)
X_test_pca_plot = pca_plot.transform(X_test_scaled)

plt.figure(figsize=(12, 5))

# True labels
plt.subplot(1, 2, 1)
for cls in classes:
    mask = (y_test == cls)
    plt.scatter(
        X_test_pca_plot[mask, 0],
        X_test_pca_plot[mask, 1],
        label=cls,
        alpha=0.7
    )

plt.title("True Labels in PCA Space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()

# Predicted labels
plt.subplot(1, 2, 2)
for cls in classes:
    mask = (y_pred == cls)
    plt.scatter(
        X_test_pca_plot[mask, 0],
        X_test_pca_plot[mask, 1],
        label=cls,
        alpha=0.7
    )

plt.title("Predicted Labels in PCA Space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()

plt.tight_layout()
plt.show()


########################### F1 Score Visualization ###########################
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

df_report.loc[classes, "f1-score"].plot(kind="bar")

plt.title("F1 Score per Subtype: PCA Logistic Regression")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()


misclassified = (y_test != y_pred)

plt.figure(figsize=(6,5))

# plot all points by true label
for cls in classes:
    mask = (y_test == cls)
    plt.scatter(
        X_test_pca_model[mask, 0],
        X_test_pca_model[mask, 1],
        label=cls,
        alpha=0.5
    )

# highlight incorrect predictions
plt.scatter(
    X_test_pca_model[misclassified, 0],
    X_test_pca_model[misclassified, 1],
    facecolors="none",
    edgecolors="black",
    s=80,
    label="Misclassified"
)

plt.title("Misclassified Samples: PCA Logistic Regression")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()

"""
Results

Logistic Regression with PCA Features was shown to be slightly less accurate from logisitic regression alone, 
indicating the feature reduction of PCA is losing some signal that is important to classification.

"""