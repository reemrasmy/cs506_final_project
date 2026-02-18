### *Final Project Proposal* ###

## Predictive Modeling of Breast Cancer Subtypes from Gene Expression Data ##

### Project Description ###

Breast cancer is one of the most common type of cancers, but it is not a single uniform disease. Although pateints  receive the same diagnosis, tumors can differ significantly at a molecular level and these differences influence how aggressively a cancer behaves and how it responds to treatment. The classifications of breast cancer subtypes are Luminal A, Luminal B, HER2, and Basal. Identifying these subtypes traditionally requires time-intensive specialized laboratory testing. Howver, data generated through RNA sequencing (RNA-seq) has been proven to carry information for these classifications by essentially measuring the activity levels of all genes within a tumor and giving a gene expression profile. 

The high-dimensionaltiy of these datasets (each tumor holds expression measurements for ~ 20,000 genes) make them quite complex to analyze manually. I would like to develop supervised classification models that predict breast cancer subtypes from RNA-seq expression data and evaluate model performance. 

### Goals ###

#### Primary Goal ####

Assess the separaility of breast cancer molecular subtypes in gene expression space using supervised classificaiton models 

Secondary Goals: 

1. Compare the accuracy of linear and nonlinear approaches for classifying breast cancer subtypes

2. Evaluate how dimensionality reduction affects classification performance

3. Visualize subtype structure using PCA and clustering methods.


### Timeline ###

#### Week 1: Data Preparation ####
Download TCGA BRCA RNA-seq expression data & merge datasets to filter tumor samples 

Explore the class distribution 

#### Week 2: Dimensionality Reduction ####
Apply variance-based filtering 

Perform PCA/SVD

Generate initial subtype visuzalization 

#### Week 3: Clustering ####
Apply K-means clustering 

Compare clusters to known subtype labels & understand initial biological separability 

#### Week 4: Baseline Classification ####
Check-in #1 (After Spring Break) 

Implement first supervised model (KNN)

#### Week 5-6: Model Comparison ####

Implement additional models (Decision Tree, Linear SVM)
Compare linear vs. nonlinear approaches 

#### Week 7: Logistic Regression ####

Add logisitice regression model and analyze separability of subtypes

#### Week 8: Neural Networks (if time allows) & Summarization/Analysis ####

Try to implement a simple feedforward neural network 

Check in #2

### Data Collection ###

The TCGA Breast Invasive Carcinoma (BRCA) dataset files would be accessed via UCSC Xena Browser and downladed as CSV. 
1. Gene Expression Data 
    RNA-seq normalized gene expression values (1,200 samples X 20,000 genes per sample)

2. Phenotype Metadata 
    Contains molecular subtype labels, tumor vs. normal indicators & sample IDs (for merging)

### Modeling & Visualization (ideas) ###

 #### *Feature Extraction & Preprocessing* ####

Since the primary features would be log-normalized RNA-seq gene expression values and each sample contains many genes, I would apply feature preprocessing steps like variance-based filtering and SVD to reduce dimensionality and generate a feature set more suitable for classification. 

#### *Potential Modeling Techniques* ####

  1. Logistic Regression 
  2. SVM (linear)

#### *Visualization* ####

  1. PCA 
  2. Model Comparison Figure 

