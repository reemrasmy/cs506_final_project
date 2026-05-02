import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

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


########################### Apply Decision Tree Model ###########################

dtree = DecisionTreeClassifier(
    max_depth=5,          # prevents overfitting
    random_state=42
)

dtree.fit(X_train_scaled, y_train)

y_pred_tree = dtree.predict(X_test_scaled)

print("Decision Tree Results:")
print(classification_report(y_test, y_pred_tree))

#Comparing Decision Tree Depths
depths = [3, 4, 5]

for depth in depths:
    dtree = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    dtree.fit(X_train_scaled, y_train)
    y_pred_tree = dtree.predict(X_test_scaled)

    print(f"\nDecision Tree Results | max_depth = {depth}")
    print(classification_report(y_test, y_pred_tree))

# Error: "Precision is ill-defined ... no predicted samples" -- for at least one subtype, the model never predicted that class (Her2)

########################### Visualize Decision Tree ###########################


dtree = DecisionTreeClassifier(
    max_depth=3,
    class_weight="balanced",    # added after model failed to predicted Her2
    random_state=43
)

dtree.fit(X_train_scaled, y_train)
y_pred_tree = dtree.predict(X_test_scaled)

plt.figure(figsize=(24, 12))
plot_tree(
    dtree,
    feature_names=top_genes,
    class_names=dtree.classes_,
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree Classifier")
plt.show()

########################### Visualization of Outcomes ###########################


classes = ["Basal", "Her2", "LumA", "LumB"]

# ---------------------- True vs. Predicted Classification ----------------------

pca_plot = PCA(n_components=2)
X_train_pca_plot = pca_plot.fit_transform(X_train_scaled)
X_test_pca_plot = pca_plot.transform(X_test_scaled)

plt.figure(figsize=(12, 5))

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

plt.subplot(1, 2, 2)
for cls in classes:
    mask = (y_pred_tree == cls)
    plt.scatter(
        X_test_pca_plot[mask, 0],
        X_test_pca_plot[mask, 1],
        label=cls,
        alpha=0.7
    )

plt.title("Decision Tree Predictions in PCA Space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()

plt.tight_layout()
plt.show()


# ---------------------- Confusion Matrix ----------------------

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_tree,
    display_labels=classes,
    cmap="Blues"
)

plt.title("Confusion Matrix: Decision Tree")
plt.show()

"""
RESULTS 

Modeling using a decision tree classifier. Iterating through different depths (d=[2,3,5,10]), the model showed at d=2, there was a
"Precision is ill-defined" error, which means the model failed to predict a certain class completely. A depth of d=3 showed 
the highest predictablility on the initial test set. However, further increases in depth (d = 5 and d = 10) led to a decline in performance, suggesting overfitting. 
Overall, a moderate tree depth (approximately d = 3–4) provided the best balance between model complexity and generalization.  

(Figure showing the decision tree -- note the genes for report!!)

Decision Tree Results | max_depth = 3
              precision    recall  f1-score   support

       Basal       0.96      0.89      0.93        28
        Her2       0.80      0.62      0.70        13
        LumA       0.86      0.80      0.83        85
        LumB       0.64      0.82      0.72        39

    accuracy                           0.81       165
   macro avg       0.82      0.78      0.79       165
weighted avg       0.82      0.81      0.81       165

Decision Tree Results | max_depth = 4
              precision    recall  f1-score   support

       Basal       0.96      0.89      0.93        28
        Her2       0.80      0.62      0.70        13
        LumA       0.84      0.91      0.87        85
        LumB       0.78      0.74      0.76        39

    accuracy                           0.84       165
   macro avg       0.85      0.79      0.81       165
weighted avg       0.84      0.84      0.84       165

"""