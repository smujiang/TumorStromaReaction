import pandas as pd
import os
import matplotlib.pyplot as plt
from pprint import pprint



f_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/result_analysis"
xlsx_fn = "All_cases_metadata_12_08_2021.xlsx"

fn = os.path.join(f_dir, xlsx_fn)
# fn = "/anonymized_dir/Dataset/OvaryData/incomplete_cases BT_JJ_09272021.xlsx"
df = pd.read_excel(fn, engine='openpyxl')

header = list(df.head())

tissue_site = df["tissue_site"]
fresh_cut = df["fresh_cut"]
histology_type = df["histology_type"]
# clinic_num = df["clinic_num"]
feature_text = header[16:]  # index 14
cellular_features = df[feature_text]
# x = cellular_features.iloc[5]
# _features = df[""]

tceom = df[cellular_features["tumor Cytoplasm: Eosin OD max_mean"] < 0.25]
selected = tceom[tceom["fresh_cut"] == 1.0]
case_ids = list(selected["deidentified_id"])
print(case_ids)

all_case_ids = list(df["deidentified_id"])[0:973]
contains_space = []
for idx, aci in enumerate(all_case_ids):
    print(aci)
    print(idx)
    if " " in aci:
        contains_space.append(aci)

not_contains_space = []
for idx,aci in enumerate(all_case_ids):
    if aci not in contains_space:
        not_contains_space.append(aci)

from PIL import Image, ImageDraw
import numpy as np
thumbnail_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/thumbnails"
output_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/batch_effect"
file_list_txt = "file_names.txt"
fp = open(os.path.join(output_dir, file_list_txt), 'w')
wrt_str = "idx, case_1, case_2\n"
for idx, aci in enumerate(contains_space):
    sv_img_arr = np.zeros((512, 1024,  3), dtype=np.uint8)
    left_img_fn = os.path.join(thumbnail_dir, contains_space[idx], contains_space[idx] + "_thumbnail.png")
    left_img = Image.open(left_img_fn).resize([512, 512])
    right_img_fn = os.path.join(thumbnail_dir, not_contains_space[idx], not_contains_space[idx] + "_thumbnail.png")
    right_img = Image.open(right_img_fn).resize([512, 512])
    sv_img_arr[:, 0:512, :] = np.array(left_img)[:, :, 0:3]
    sv_img_arr[:, 512:1024, :] = np.array(right_img)[:, :, 0:3]
    save_to = os.path.join(output_dir, str(idx)+".jpg")
    Image.fromarray(sv_img_arr).save(save_to)
    wrt_str += str(idx) + "," + contains_space[idx] + "," + not_contains_space[idx] + "\n"

fp.write(wrt_str)
fp.close()


# draw figure
start_x = 40
selected_feature_text = list(feature_text)[start_x:start_x + 8]
fig, ax = plt.subplots(len(selected_feature_text), len(selected_feature_text))
fig.set_figheight(30)
fig.set_figwidth(30)
for idx1, ft1 in enumerate(selected_feature_text):
    for idx2, ft2 in enumerate(selected_feature_text):
        if idx1 > idx2:
            x = cellular_features[ft1]
            y = cellular_features[ft2]
            ax[idx1, idx2].plot(list(x), list(y), ".r")
            ax[idx1, idx2].set_xlabel(ft1)
            ax[idx1, idx2].set_ylabel(ft2)
        else:
            print("don't show")
plt.show()
print("Done")
