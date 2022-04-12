import os

survival_analysis_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/result_analysis/survival"
batch1_save_to = os.path.join(survival_analysis_dir, "batch1_metadata.tsv")
batch1_df.to_csv(batch1_save_to, sep="\t")

