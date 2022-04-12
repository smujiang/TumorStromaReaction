from PIL import Image
import os
import numpy as np
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk
import matplotlib.pyplot as plt

thumbnail_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/thumbnails"
output_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/result_analysis"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

fn_seg_ext = "_segmentation.png"
fn_tsr_fib_ext = "_Fibrosis_TSR-score.png"
fn_tsr_cel_ext = "_Cellularity_TSR-score.png"
fn_tsr_ori_ext = "_Orientation_TSR-score.png"

seg_color_map = [[0, 255, 255], [255, 255, 0]]  # tumor & stroma
color_map = [[[100, 0, 0], [180, 0, 0], [255, 0, 0]],  # Fibrosis: 0, 1, 2
             [[0, 100, 0], [0, 180, 0], [0, 255, 0]],
             [[0, 0, 100], [0, 0, 180], [0, 0, 255]]]
Class_label_list = ["Fibrosis", "Cellularity", "Orientation"]


def decode_color_to_score(color, color_list, tolerance=10):
    for idx, i in enumerate(color_list):
        if i[0] - tolerance < color[0] < i[0] + tolerance \
                and i[1] - tolerance < color[1] < i[1] + tolerance \
                and i[2] - tolerance < color[2] < i[2] + tolerance:
            return idx
    return -1

TSR_hist_fn = os.path.join(output_dir, "all_hist_rec.csv")
already_processed_cases = []
if os.path.exists(TSR_hist_fn):
    fp = open(os.path.join(output_dir, "all_hist_rec.csv"), "r")
    for l in fp.readlines()[1:]:
        ele = l.split(",")
        already_processed_cases.append(ele[0])
else:
    already_processed_cases = []

def get_unprocessed_cases(overall_list, processed_list):
    ids = [f for f in overall_list if f not in processed_list]
    return ids

# over_all_case_id_list = sorted(os.listdir(thumbnail_dir))
# case_id_list = get_unprocessed_cases(over_all_case_id_list, already_processed_cases)
#################################################
batch1_case_id_list_fn = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/result_analysis/batch1_case_id.txt"
fp = open(batch1_case_id_list_fn, 'r')
lines = fp.readlines()
batch_1_cases = []
for i in lines:
    if len(i.strip()) > 0:
        batch_1_cases.append(i.strip())
case_id_list = batch_1_cases

# save to file
save_to = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/result_analysis/raw_tsr_count.tsv"
wrt_str = "case_id,fibrosis0,fibrosis1,fibrosis2,cellularity_0,cellularity_1,cellularity_2,orientation_0,orientation_1,orientation_2\n"

#################################################

case_fib_tsr_score_lists = []
case_cel_tsr_score_lists = []
case_ori_tsr_score_lists = []
for idx, case_id in enumerate(case_id_list):
    case_fib_tsr_scores = []
    case_cel_tsr_scores = []
    case_ori_tsr_scores = []

    print("processing %s. %d/%d" % (case_id, idx + 1, len(case_id_list)))
    seg_fn = os.path.join(thumbnail_dir, case_id, case_id + fn_seg_ext)
    tumor_stroma_img = Image.open(seg_fn, 'r')
    tumor_stroma_img_arr = np.array(tumor_stroma_img)[:, :, 0:3]

    '''
    # get tumor core
    '''
    k = tumor_stroma_img_arr == seg_color_map[0]
    tumor = np.all(k, axis=2) * 1
    # # plt.imshow(tumor, cmap="gray")
    # # plt.show()
    #
    selem = disk(3)
    dilated = dilation(tumor, selem)
    tumor_core = erosion(dilated, selem)
    # plt.imshow(tumor_core, cmap="gray")
    # plt.axis(False)
    # plt.show()
    dilated_tumor_core = dilation(tumor_core, selem)
    # plt.imshow(dilated_tumor_core, cmap="gray")
    # plt.axis(False)
    # plt.show()

    '''
    #get stroma core
    '''
    k = tumor_stroma_img_arr == seg_color_map[1]
    stroma = np.all(k, axis=2) * 1
    # plt.imshow(stroma, cmap="gray")
    # plt.show()

    selem = disk(3)
    dilated = dilation(stroma, selem)
    stroma_core = erosion(dilated, selem)
    # plt.imshow(stroma_core, cmap="gray")
    # plt.axis(False)
    # plt.show()

    # print(tumor.shape)

    '''
    # get entangling area
    '''
    selem = disk(8)
    dilated_stroma_core = dilation(stroma_core, selem)
    # plt.imshow(dilated_stroma_core, cmap="gray")
    # plt.axis(False)
    # plt.show()
    eroded_stroma_core = erosion(stroma_core, selem)
    # plt.imshow(eroded_stroma_core, cmap="gray")
    # plt.axis(False)
    # plt.show()
    entangle_area = np.logical_xor(dilated_stroma_core, eroded_stroma_core)
    # plt.imshow(entangle_area, cmap="gray")
    # plt.axis(False)
    # plt.show()

    entangle_area = np.logical_and(entangle_area, stroma_core)
    entangle_area = np.logical_and(entangle_area, dilated_tumor_core)
    # plt.imshow(entangle_area, cmap="gray")
    # plt.axis(False)
    # plt.show()

    # print(entangle_area.shape)
    # h, w = entangle_area.shape

    fn_tsr_fib = os.path.join(thumbnail_dir, case_id, case_id + fn_tsr_fib_ext)
    tsr_fib = np.array(Image.open(fn_tsr_fib, 'r'))[:, :, 0:3]
    fib_all_scores_encoded = tsr_fib[entangle_area == 1]
    # stroma_core_fib_all_scores_encoded = tsr_fib[stroma_core == 1]

    fn_tsr_cel = os.path.join(thumbnail_dir, case_id, case_id + fn_tsr_cel_ext)
    tsr_cel = np.array(Image.open(fn_tsr_cel, 'r'))[:, :, 0:3]
    cel_all_scores_encoded = tsr_cel[entangle_area == 1]
    # stroma_core_cel_all_scores_encoded = tsr_cel[stroma_core == 1]

    fn_tsr_ori = os.path.join(thumbnail_dir, case_id, case_id + fn_tsr_ori_ext)
    tsr_ori = np.array(Image.open(fn_tsr_ori, 'r'))[:, :, 0:3]
    ori_all_scores_encoded = tsr_ori[entangle_area == 1]
    # stroma_core_ori_all_scores_encoded = tsr_ori[stroma_core == 1]

    for i in fib_all_scores_encoded:
        score = decode_color_to_score(i, color_map[0])
        if score is not -1:
            case_fib_tsr_scores.append(score)
    for i in cel_all_scores_encoded:
        score = decode_color_to_score(i, color_map[1])
        if score is not -1:
            case_cel_tsr_scores.append(score)
    for i in ori_all_scores_encoded:
        score = decode_color_to_score(i, color_map[2])
        if score is not -1:
            case_ori_tsr_scores.append(score)

    case_fib_tsr_score_lists.append(case_fib_tsr_scores)
    case_cel_tsr_score_lists.append(case_cel_tsr_scores)
    case_ori_tsr_score_lists.append(case_ori_tsr_scores)

# raw hist
raw_heatmap_list = []
for idx, case_cel_tsr in enumerate(case_cel_tsr_score_lists):
    case_fib_feature, _ = np.histogram(case_fib_tsr_score_lists[idx], density=False, bins=[-0.5, 0.5, 1.5, 2.5])
    case_cel_feature, _ = np.histogram(case_cel_tsr, density=False, bins=[-0.5, 0.5, 1.5, 2.5])
    case_ori_feature, _ = np.histogram(case_ori_tsr_score_lists[idx], density=False, bins=[-0.5, 0.5, 1.5, 2.5])
    raw_heatmap_list.append(np.concatenate([case_fib_feature, case_cel_feature, case_ori_feature]))

for idx, rh in enumerate(raw_heatmap_list):
    wrt_str += case_id_list[idx] + ","
    string_rh = [str(r) for r in rh]
    wrt_str += ",".join(string_rh) + "\n"
fp = open(save_to, "w")
fp.write(wrt_str)
fp.close()


# case level histogram as case feature
heatmap_list = []
for idx, case_cel_tsr in enumerate(case_cel_tsr_score_lists):
    case_fib_feature, _ = np.histogram(case_fib_tsr_score_lists[idx], density=True, bins=[-0.5, 0.5, 1.5, 2.5], normed=True)
    case_cel_feature, _ = np.histogram(case_cel_tsr, density=True, bins=[-0.5, 0.5, 1.5, 2.5], normed=True)
    case_ori_feature, _ = np.histogram(case_ori_tsr_score_lists[idx], density=True, bins=[-0.5, 0.5, 1.5, 2.5], normed=True)
    heatmap_list.append(np.concatenate([case_fib_feature, case_cel_feature, case_ori_feature]))

heatmap_array = np.array(heatmap_list)

# save heatmap to csv
wrt_str = "case_id,fibrosis_0,fibrosis_1,fibrosis_2,cellularity_0,cellularity_1,cellularity_2,orientation_0,orientation_1,orientation_2\n"
for idx, c in enumerate(case_id_list):
    wrt_str += str(c) + ","
    for s in heatmap_array[idx, :]:
        wrt_str += str(s) + ","
    wrt_str = wrt_str[0:-1]
    wrt_str += "\n"
fp = open(os.path.join(output_dir, "all_hist_rec.csv"), "w")
fp.write(wrt_str)
fp.close()


plt.figure(1)
plt.imshow(heatmap_array, cmap='Blues')
ax = plt.gca()
ax.set_yticklabels(case_id_list)
plt.show()


