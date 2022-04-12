import os
import numpy as np

header = "Fibrosis,Cellularity,Orientation,img_fn\n"

# train with SBOT cases as negative controls
shuffled_training_csv_file = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/PatchSampling/training_with_SBOT_cases.csv"
shuffled_validation_csv_file = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/PatchSampling/validation_with_SBOT_cases.csv"

training_lines = open(shuffled_training_csv_file, 'r').readlines()
validate_lines = open(shuffled_validation_csv_file, 'r').readlines()

train_HGSOC = []
train_neg_control = []
val_neg_control = []
val_HGSOC = []
for i in training_lines[1:]:
    ele = i.split(",")
    if "OCMC-" in ele[3]:
        train_neg_control.append([ele[0], ele[1], ele[2]])
    else:
        train_HGSOC.append([ele[0], ele[1], ele[2]])

for i in validate_lines[1:]:
    ele = i.split(",")
    if "OCMC-" in ele[3]:
        val_neg_control.append([ele[0], ele[1], ele[2]])
    else:
        val_HGSOC.append([ele[0], ele[1], ele[2]])

train_HGSOC = np.array(train_HGSOC).astype(int)
train_neg_control = np.array(train_neg_control).astype(int)
val_neg_control = np.array(val_neg_control).astype(int)
val_HGSOC = np.array(val_HGSOC).astype(int)


train_HGSOC_hist_fibrosis, _ = np.histogram(train_HGSOC[:, 0], bins=3, range=(0, 2))
train_HGSOC_hist_cellularity, _ = np.histogram(train_HGSOC[:, 1], bins=3, range=(0, 2))
train_HGSOC_hist_orientation, _ = np.histogram(train_HGSOC[:, 2], bins=3, range=(0, 2))

train_neg_control_hist_fibrosis, _ = np.histogram(train_neg_control[:, 0], bins=3, range=(0, 2))
train_neg_control_hist_cellularity, _ = np.histogram(train_neg_control[:, 1], bins=3, range=(0, 2))
train_neg_control_hist_orientation, _ = np.histogram(train_neg_control[:, 2], bins=3, range=(0, 2))

val_HGSOC_hist_fibrosis, _ = np.histogram(val_HGSOC[:, 0], bins=3, range=(0, 2))
val_HGSOC_hist_cellularity, _ = np.histogram(val_HGSOC[:, 1], bins=3, range=(0, 2))
val_HGSOC_hist_orientation, _ = np.histogram(val_HGSOC[:, 2], bins=3, range=(0, 2))

val_neg_control_hist_fibrosis, _ = np.histogram(val_neg_control[:, 0], bins=3, range=(0, 2))
val_neg_control_hist_cellularity, _ = np.histogram(val_neg_control[:, 1], bins=3, range=(0, 2))
val_neg_control_hist_orientation, _ = np.histogram(val_neg_control[:, 2], bins=3, range=(0, 2))

line1 = np.concatenate([train_HGSOC_hist_fibrosis, train_HGSOC_hist_cellularity, train_HGSOC_hist_orientation]).astype(str)
print(",".join(line1))

line1 = np.concatenate([train_neg_control_hist_fibrosis, train_neg_control_hist_cellularity, train_neg_control_hist_orientation]).astype(str)
print(",".join(line1))

line1 = np.concatenate([val_HGSOC_hist_fibrosis, val_HGSOC_hist_cellularity, val_HGSOC_hist_orientation]).astype(str)
print(",".join(line1))

line1 = np.concatenate([val_neg_control_hist_fibrosis, val_neg_control_hist_cellularity, val_neg_control_hist_orientation]).astype(str)
print(",".join(line1))

print("Done")




