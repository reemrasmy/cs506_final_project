import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

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



########################### Apply Model: Non-Linear SVM ###########################

# Untuned/default RBF SVM model
# This is used as the baseline nonlinear SVM comparison.
rbf_svm_default = SVC(
    kernel="rbf",
    C=1,
    gamma="scale",
    class_weight="balanced",
    random_state=42
)

rbf_svm_default.fit(X_train_scaled, y_train)
y_pred_rbf_default = rbf_svm_default.predict(X_test_scaled)

print("\nUntuned RBF SVM Results:")
print(classification_report(y_test, y_pred_rbf_default, zero_division=0))

# ---------------------- Tuning RBF SVM parameters ----------------------
# GridSearchCV tests multiple C/gamma combinations using cross-validation on the training set only not the test set.
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.001, 0.01, 0.1]
}

rbf_svm = SVC(
    kernel="rbf",
    class_weight="balanced",
    random_state=42
)

grid_search = GridSearchCV(
    estimator=rbf_svm,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

print("\nBest parameters:", grid_search.best_params_)
print("Best CV macro F1:", grid_search.best_score_)

# best_estimator_ means the best model learned AFTER fitting GridSearchCV
best_svm = grid_search.best_estimator_
y_pred_rbf_tuned = best_svm.predict(X_test_scaled)

print("\nTuned RBF SVM Results:")
print(classification_report(y_test, y_pred_rbf_tuned, zero_division=0))

# ---------------------- Determining the better test-performing SVM model for Visualization ----------------------

# The untuned RBF SVM generalized slightly better than the tuned model. (using y_pred_rbf for plotting)
y_pred_rbf = y_pred_rbf_default

########################### Visualize Outcome ###########################


# ---------------------- True vs. Predicted Classification (side-by-side) ----------------------
classes = ["Basal", "Her2", "LumA", "LumB"]

# Fit PCA on training only then transform the test data
# Important: PCA is used here just for 2D visualization NOT as part of the SVM model
pca_plot = PCA(n_components=2)
X_train_pca = pca_plot.fit_transform(X_train_scaled)
X_test_pca = pca_plot.transform(X_test_scaled)

plt.figure(figsize=(12,5))

# True labels
plt.subplot(1,2,1)
for cls in classes:
    mask = (y_test == cls)
    plt.scatter(X_test_pca[mask, 0],
                X_test_pca[mask, 1],
                label=cls,
                alpha=0.7)

plt.title("True Labels (PCA Space)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()

# Predicted labels
plt.subplot(1,2,2)
for cls in classes:
    mask = (y_pred_rbf == cls)
    plt.scatter(X_test_pca[mask, 0],
                X_test_pca[mask, 1],
                label=cls,
                alpha=0.7)

plt.title("SVM Predictions (PCA Space)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()

plt.tight_layout()
plt.show()

# ---------------------- Confusion Matrix ----------------------

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_rbf,
    display_labels=classes,
    cmap="Blues"
)

plt.title("Confusion Matrix: SVM")
plt.show()

# ---------------------- F1 Scores ----------------------
report = classification_report(y_test, y_pred_rbf, output_dict=True)
df_report = pd.DataFrame(report).transpose()

classes = ["Basal", "Her2", "LumA", "LumB"]

df_report.loc[classes, "f1-score"].plot(kind="bar")

plt.title("F1 Score per Subtype: SVM")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.show()

# ---------------------- Misclassified Points ----------------------

misclassified = (y_test != y_pred_rbf)

plt.figure(figsize=(6,5))

# plot all points by true label
for cls in classes:
    mask = (y_test == cls)
    plt.scatter(X_test_pca[mask, 0],
                X_test_pca[mask, 1],
                label=cls,
                alpha=0.5)

# highlight errors
plt.scatter(X_test_pca[misclassified, 0],
            X_test_pca[misclassified, 1],
            facecolors='none',
            edgecolors='black',
            s=80,
            label="Misclassified")

plt.title("Misclassified Samples (SVM)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()

"""

RBF SVM Results (no parameter tuning):
              precision    recall  f1-score   support

       Basal       1.00      1.00      1.00        28
        Her2       0.90      0.69      0.78        13
        LumA       0.95      0.88      0.91        85
        LumB       0.71      0.87      0.78        39

    accuracy                           0.88       165
   macro avg       0.89      0.86      0.87       165
weighted avg       0.90      0.88      0.89       165

RBF SVM Results (with parameter tuning):
              precision    recall  f1-score   support

       Basal       1.00      1.00      1.00        28
        Her2       0.89      0.62      0.73        13
        LumA       0.86      0.96      0.91        85
        LumB       0.82      0.69      0.75        39

    accuracy                           0.88       165
   macro avg       0.89      0.82      0.85       165
weighted avg       0.88      0.88      0.87       165
"""

