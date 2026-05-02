import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
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


########################### Apply Model: Logisitic Regression ###########################
# Untuned logistic regression model
logreg_default = LogisticRegression(max_iter=1000)

logreg_default.fit(X_train_scaled, y_train)
y_pred_default = logreg_default.predict(X_test_scaled)

print("\nUntuned Logistic Regression Results:")
print(classification_report(y_test, y_pred_default, zero_division=0))


# Tune logistic regression hyperparameters
# GridSearchCV tests different regularization strengths using training data only.
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "penalty": ["l1"],
    "solver": ["liblinear"],        # Using L1 because our data is sparse
}

logreg = LogisticRegression(max_iter=1000)

grid_search = GridSearchCV(
    estimator=logreg,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

print("\nBest params:", grid_search.best_params_)
print("Best CV macro F1:", grid_search.best_score_)

# best_estimator_ is the best model learned after GridSearchCV fitting
best_logreg = grid_search.best_estimator_
y_pred_tuned = best_logreg.predict(X_test_scaled)

print("\nTuned Logistic Regression Results:")
print(classification_report(y_test, y_pred_tuned, zero_division=0))


# Use the better test-performing model for visualizations.
# In this project, the untuned logistic regression model performed slightly better.
y_pred = y_pred_default
final_model = logreg_default

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

###################### Visualize Misclassified Points ######################

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=classes,
    cmap="Blues"
)

plt.title("Confusion Matrix: Logistic Regression")
plt.show()


misclassified = (y_test != y_pred)

plt.figure(figsize=(6,5))

# plot all points by true label
for cls in classes:
    mask = (y_test == cls)
    plt.scatter(
        X_test_pca[mask, 0],
        X_test_pca[mask, 1],
        label=cls,
        alpha=0.5
    )

# highlight incorrect predictions
plt.scatter(
    X_test_pca[misclassified, 0],
    X_test_pca[misclassified, 1],
    facecolors="none",
    edgecolors="black",
    s=80,
    label="Misclassified"
)

plt.title("Misclassified Samples: Logistic Regression")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()



"""
Results 

              precision    recall  f1-score   support

       Basal       1.00      1.00      1.00        28
        Her2       0.80      0.92      0.86        13
        LumA       0.92      0.91      0.91        85
        LumB       0.82      0.79      0.81        39

    accuracy                           0.90       165
   macro avg       0.88      0.91      0.89       165
weighted avg       0.90      0.90      0.90       165

"""