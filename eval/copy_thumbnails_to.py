import os
import shutil

thumbnail_dir = "\\\\anonymized_dir\\thumbnails"
out_dir = "\\\\anonymized_dir\\StromalReaction\\review_cases"

# low stroma ratio (less than 200 patches)
low_stroma_ratio_cases = [""]



# cases with high TSR scores
high_TSR_score_cases = [""]

# SBOT cases with high false positive rate
SBOT_high_FP_cases = ["OCMC-022", "OCMC-023", "OCMC-025"]


#TODO: copy the thumbnails to the folders

for c in low_stroma_ratio_cases:
    source = os.path.join(thumbnail_dir, c)
    destination = os.path.join(out_dir, "LowStromaRatio_thumbnails", c)
    if not os.path.exists(destination):
        os.makedirs(destination, exist_ok=True)
        shutil.copytree(source, destination)

for c in high_TSR_score_cases:
    source = os.path.join(thumbnail_dir,  c)
    destination = os.path.join(out_dir, "HighTSR_thumbnails", c)
    if not os.path.exists(destination):
        os.makedirs(destination, exist_ok=True)
        shutil.copytree(source, destination)

for c in SBOT_high_FP_cases:
    source = os.path.join(thumbnail_dir,  c)
    destination = os.path.join(out_dir, "SBOT_high_false_positive_thumbnails", c)
    if not os.path.exists(destination):
        os.makedirs(destination, exist_ok=True)
        shutil.copytree(source, destination)

#TODO: paste some thumbnails to the slides.
#TODO: create QuPath projects

