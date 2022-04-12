
# get non-color related features
import os
data_dir = "/anonymized_dir/Dataset/OvaryData/StromaReaction/survival"
data_fn = os.path.join(data_dir, "batch1_metadata.tsv")

lines = open(data_fn, 'r').readlines()
header = lines[0].split("\t")
features_range = range(18, 195)
idx = []
for i in features_range:
    if "Hematoxylin" in header[i] or "Eosin" in header[i] or "OD" in header[i]:
        pass
    else:
        print(header[i])
        idx.append(i)

print(idx)

