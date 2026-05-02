import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def variance_filtering(exp_data, n_genes=5000):
    # Computing the variance column-wise: the variance for each gene across all samples
    gene_variances = exp_data.var(axis=0)
    # sorting the genes by variance and taking the genes with the top variance as features
    top_genes = gene_variances.sort_values(ascending=False).head(n_genes).index

    # subset and return the expression data matrix for genes with top variance
    filtered_expression = exp_data[top_genes]

    return filtered_expression

gene_expr_df = pd.read_csv("data/processed_brca_data.csv")

X = gene_expr_df.drop(columns=["sampleID", "subtype"])
y = gene_expr_df["subtype"]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

########################### Variance-Based Gene Filtering ###########################
# Removing the 0 variance genes based on training set only
train_var = X_train.var()
keep_genes = train_var[train_var > 0].index

# Select only the genes with some variance
X_train_nonzero = X_train[keep_genes]
X_test_nonzero = X_test[keep_genes]

# Select the top 5000 variable genes from training set
X_train_filtered = variance_filtering(X_train_nonzero, n_genes=5000)
top_genes = X_train_filtered.columns

# Apply the same selected top genes to the test set
X_test_filtered = X_test_nonzero[top_genes]

########################### Normalize Gene Expression Values ###########################
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_filtered)
X_test_scaled = scaler.transform(X_test_filtered)


########################### Apply Baseline (1st) Model: Logisitic Regression ###########################
# fit logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

########################### Visualize Outcome ###########################
plt.figure(figsize=(12,5))

classes = ["Basal", "Her2", "LumA", "LumB"]

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

plt.figure(figsize=(12,5))

# True labels
plt.subplot(1,2,1)
for cls in classes:
    mask = (y_test == cls)
    plt.scatter(X_test_pca[mask, 0],
                X_test_pca[mask, 1],
                label=cls,
                alpha=0.7)

plt.title("True Labels")
plt.legend()

# Predicted labels
plt.subplot(1,2,2)
for cls in classes:
    mask = (y_pred == cls)
    plt.scatter(X_test_pca[mask, 0],
                X_test_pca[mask, 1],
                label=cls,
                alpha=0.7)

plt.title("Predicted Labels")
plt.legend()

plt.show()

###################### Classification Ststistcs & Vszualization ######################
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

df_report.loc[["Basal", "Her2", "LumA", "LumB"], "f1-score"].plot(kind="bar")

plt.title("F1 Score per Subtype")
plt.ylabel("F1 Score")
plt.ylim(0, 1)

plt.show()

"""
Results 

"""