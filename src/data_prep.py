import pandas as pd
import os

# Get project root (go up one level from src/)
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# build paths
data_dir = os.path.join(root_dir, 'data')

def process_expression_data(path):
    expr = pd.read_csv(path, sep='\t')

    expr = expr.set_index('sample').T

    expr.index = expr.index.astype(str).str.strip()
    expr.index.name = "sampleID"
    expr.columns.name = None

    expr = expr.reset_index()

    return expr

def process_clinical_metadata(path):
    clinical_meta = (
        pd.read_csv(path, sep='\t')[["sampleID", "PAM50Call_RNAseq", "sample_type"]]
        .rename(columns={"PAM50Call_RNAseq": "subtype"})
        .dropna(subset=["subtype"])
        .query("subtype != 'Normal'")
        .query("sample_type == 'Primary Tumor'")
        .assign(
            sampleID=lambda df: df["sampleID"].astype(str).str.strip(),
            subtype=lambda df: df["subtype"].astype(str).str.strip())
        [["sampleID", "subtype"]]

    )

    return clinical_meta

def merge_expression_metadata(expr_df, clin_meta_df):

    merged = pd.merge(expr_df, clin_meta_df, on="sampleID")

    return merged

expression_data = os.path.join(data_dir,  "TCGA.BRCA.sampleMap_HiSeqV2.csv")
clinical_data = os.path.join(data_dir, "TCGA.BRCA.sampleMap_BRCA_clinicalMatrix.csv")
processed_data = os.path.join(data_dir, "processed_brca_data.csv")


expression_df = process_expression_data(expression_data)
clinical_metadata = process_clinical_metadata(clinical_data)
merged_expr = merge_expression_metadata(expression_df, clinical_metadata)           # print(merged_expr.shape)      # (821, 20532)

print("Raw Processed Expression Matrix Shape:", merged_expr.shape)


merged_expr.to_csv(processed_data)
print("Gene Expression Processing DONE \nProcessed Gene Expression CSV in: data/")

