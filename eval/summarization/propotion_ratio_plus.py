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

SBOT_fib_tsr_scores = []
SBOT_cel_tsr_scores = []
SBOT_ori_tsr_scores = []
HGSOC_fib_tsr_scores = []
HGSOC_cel_tsr_scores = []
HGSOC_ori_tsr_scores = []

# SBOT_tumor_core_fib_tsr_scores = []
# SBOT_tumor_core_cel_tsr_scores = []
# SBOT_tumor_core_ori_tsr_scores = []
# HGSOC_tumor_core_fib_tsr_scores = []
# HGSOC_tumor_core_cel_tsr_scores = []
# HGSOC_tumor_core_ori_tsr_scores = []
#
# SBOT_stroma_core_fib_tsr_scores = []
# SBOT_stroma_core_cel_tsr_scores = []
# SBOT_stroma_core_ori_tsr_scores = []
# HGSOC_stroma_core_fib_tsr_scores = []
# HGSOC_stroma_core_cel_tsr_scores = []
# HGSOC_stroma_core_ori_tsr_scores = []

HGSOC_case_id_list = ["OCMC-001", "OCMC-002", "OCMC-003", "OCMC-004", "OCMC-005", "OCMC-006", "OCMC-007", "OCMC-008", "OCMC-009", "OCMC-010", "OCMC-011", "OCMC-012", "OCMC-013", "OCMC-014", "OCMC-015"]

SBOT_case_id_list = ["OCMC-016", "OCMC-017", "OCMC-018", "OCMC-019", "OCMC-020", "OCMC-021", "OCMC-022",
                     "OCMC-023", "OCMC-024", "OCMC-025", "OCMC-026", "OCMC-027", "OCMC-028", "OCMC-029", "OCMC-030"]

SBOT_training = ["OCMC-016", "OCMC-018", "OCMC-020", "OCMC-022", "OCMC-024", "OCMC-026"]
HGSOC_training = [""]

# SBOT_training = []
# HGSOC_training = []


def decode_color_to_score(color, color_list, tolerance=10):
    for idx, i in enumerate(color_list):
        if i[0] - tolerance < color[0] < i[0] + tolerance \
                and i[1] - tolerance < color[1] < i[1] + tolerance \
                and i[2] - tolerance < color[2] < i[2] + tolerance:
            return idx
    return -1


SBOT_testing_case_id_list = []
case_SBOT_fib_tsr_score_lists = []
case_SBOT_cel_tsr_score_lists = []
case_SBOT_ori_tsr_score_lists = []
for idx, case_id in enumerate(SBOT_case_id_list):
    case_SBOT_fib_tsr_scores = []
    case_SBOT_cel_tsr_scores = []
    case_SBOT_ori_tsr_scores = []
    if case_id not in SBOT_training:
        SBOT_testing_case_id_list.append(case_id)
        print("processing %s. %d/%d" % (case_id, idx + 1, len(SBOT_case_id_list)))
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
                SBOT_fib_tsr_scores.append(score)
                case_SBOT_fib_tsr_scores.append(score)
        for i in cel_all_scores_encoded:
            score = decode_color_to_score(i, color_map[1])
            if score is not -1:
                SBOT_cel_tsr_scores.append(score)
                case_SBOT_cel_tsr_scores.append(score)
        for i in ori_all_scores_encoded:
            score = decode_color_to_score(i, color_map[2])
            if score is not -1:
                SBOT_ori_tsr_scores.append(score)
                case_SBOT_ori_tsr_scores.append(score)

        case_SBOT_fib_tsr_score_lists.append(case_SBOT_fib_tsr_scores)
        case_SBOT_cel_tsr_score_lists.append(case_SBOT_cel_tsr_scores)
        case_SBOT_ori_tsr_score_lists.append(case_SBOT_ori_tsr_scores)

        # for i in stroma_core_fib_all_scores_encoded:
        #     score = decode_color_to_score(i, color_map[0])
        #     if score is not -1:
        #         SBOT_stroma_core_fib_tsr_scores.append(score)
        # for i in stroma_core_cel_all_scores_encoded:
        #     score = decode_color_to_score(i, color_map[1])
        #     if score is not -1:
        #         SBOT_stroma_core_cel_tsr_scores.append(score)
        # for i in stroma_core_ori_all_scores_encoded:
        #     score = decode_color_to_score(i, color_map[2])
        #     if score is not -1:
        #         SBOT_stroma_core_ori_tsr_scores.append(score)

case_HGSOC_fib_tsr_score_lists = []
case_HGSOC_cel_tsr_score_lists = []
case_HGSOC_ori_tsr_score_lists = []
HGSOC_testing_case_id_list = []
for idx, case_id in enumerate(HGSOC_case_id_list):
    case_HGSOC_fib_tsr_scores = []
    case_HGSOC_cel_tsr_scores = []
    case_HGSOC_ori_tsr_scores = []
    if case_id not in HGSOC_training:
        HGSOC_testing_case_id_list.append(case_id)
        print("processing %s. %d/%d" % (case_id, idx + 1, len(HGSOC_case_id_list)))
        seg_fn = os.path.join(thumbnail_dir, case_id, case_id + fn_seg_ext)
        tumor_stroma_img = Image.open(seg_fn, 'r')
        tumor_stroma_img_arr = np.array(tumor_stroma_img)[:, :, 0:3]

        '''
        # get tumor core
        '''
        k = tumor_stroma_img_arr == seg_color_map[0]
        tumor = np.all(k, axis=2) * 1
        # plt.imshow(tumor, cmap="gray")
        # plt.show()
        #
        selem = disk(3)
        dilated = dilation(tumor, selem)
        tumor_core = erosion(dilated, selem)
        dilated_tumor_core = dilation(tumor_core, selem)
        # # plt.imshow(tumor_core, cmap="gray")
        # # plt.show()

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
        # plt.show()

        # print(tumor.shape)

        '''
        # get entangling area
        '''
        selem = disk(8)
        dilated_stroma_core = dilation(stroma_core, selem)
        # plt.imshow(dilated_tumor_core, cmap="gray")
        # plt.show()
        eroded_stroma_core = erosion(stroma_core, selem)
        # plt.imshow(eroded_tumor_core, cmap="gray")
        # plt.show()
        entangle_area = np.logical_xor(dilated_stroma_core, eroded_stroma_core)
        # plt.imshow(entangle_area, cmap="gray")
        # plt.show()

        entangle_area = np.logical_and(entangle_area, stroma_core)
        entangle_area = np.logical_and(entangle_area, dilated_tumor_core)
        # plt.imshow(entangle_area, cmap="gray")
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
                HGSOC_fib_tsr_scores.append(score)
                case_HGSOC_fib_tsr_scores.append(score)
        for i in cel_all_scores_encoded:
            score = decode_color_to_score(i, color_map[1])
            if score is not -1:
                HGSOC_cel_tsr_scores.append(score)
                case_HGSOC_cel_tsr_scores.append(score)
        for i in ori_all_scores_encoded:
            score = decode_color_to_score(i, color_map[2])
            if score is not -1:
                HGSOC_ori_tsr_scores.append(score)
                case_HGSOC_ori_tsr_scores.append(score)

        case_HGSOC_fib_tsr_score_lists.append(case_HGSOC_fib_tsr_scores)
        case_HGSOC_cel_tsr_score_lists.append(case_HGSOC_cel_tsr_scores)
        case_HGSOC_ori_tsr_score_lists.append(case_HGSOC_ori_tsr_scores)

        # for i in stroma_core_fib_all_scores_encoded:
        #     score = decode_color_to_score(i, color_map[0])
        #     if score is not -1:
        #         HGSOC_stroma_core_fib_tsr_scores.append(score)
        # for i in stroma_core_cel_all_scores_encoded:
        #     score = decode_color_to_score(i, color_map[1])
        #     if score is not -1:
        #         HGSOC_stroma_core_cel_tsr_scores.append(score)
        # for i in stroma_core_ori_all_scores_encoded:
        #     score = decode_color_to_score(i, color_map[2])
        #     if score is not -1:
        #         HGSOC_stroma_core_ori_tsr_scores.append(score)

# case level histogram as case feature
SBOT_heatmap_list = []
for idx, case_cel_tsr in enumerate(case_SBOT_cel_tsr_score_lists):
    case_fib_feature, _ = np.histogram(case_SBOT_fib_tsr_score_lists[idx], density=True, bins=[-0.5, 0.5, 1.5, 2.5], normed=True)
    case_cel_feature, _ = np.histogram(case_cel_tsr, density=True, bins=[-0.5, 0.5, 1.5, 2.5], normed=True)
    case_ori_feature, _ = np.histogram(case_SBOT_ori_tsr_score_lists[idx], density=True, bins=[-0.5, 0.5, 1.5, 2.5], normed=True)
    SBOT_heatmap_list.append(np.concatenate([case_fib_feature, case_cel_feature, case_ori_feature]))

HGSOC_heatmap_list = []
for idx, case_cel_tsr in enumerate(case_HGSOC_cel_tsr_score_lists):
    case_fib_feature, _ = np.histogram(case_HGSOC_fib_tsr_score_lists[idx], density=True, bins=[-0.5, 0.5, 1.5, 2.5], normed=True)
    case_cel_feature, _ = np.histogram(case_cel_tsr, density=True, bins=[-0.5, 0.5, 1.5, 2.5], normed=True)
    case_ori_feature, _ = np.histogram(case_HGSOC_ori_tsr_score_lists[idx], density=True, bins=[-0.5, 0.5, 1.5, 2.5], normed=True)
    HGSOC_heatmap_list.append(np.concatenate([case_fib_feature, case_cel_feature, case_ori_feature]))

SBOT_heatmap_array = np.array(SBOT_heatmap_list)
HGSOC_heatmap_array = np.array(HGSOC_heatmap_list)
plt.figure(1)
plt.imshow(SBOT_heatmap_array, cmap='Blues')
ax = plt.gca()
ax.set_yticklabels(SBOT_testing_case_id_list)
plt.show()

plt.figure(2)
plt.imshow(HGSOC_heatmap_array, cmap='Blues')
ax = plt.gca()
ax.set_yticklabels(HGSOC_testing_case_id_list)
plt.show()

# save heatmap to csv
wrt_str = "case_id,fibrosis_0,fibrosis_1,fibrosis_2,cellularity_0,cellularity_1,cellularity_2,orientation_0,orientation_1,orientation_2\n"
for idx, c in enumerate(SBOT_testing_case_id_list):
    wrt_str += str(c) + ","
    for s in SBOT_heatmap_array[idx, :]:
        wrt_str += str(s) + ","
    wrt_str = wrt_str[0:-1]
    wrt_str += "\n"
fp = open(os.path.join(output_dir, "SBOT_hist_rec.csv"), "w")
fp.write(wrt_str)
fp.close()

wrt_str = "case_id,fibrosis_0,fibrosis_1,fibrosis_2,cellularity_0,cellularity_1,cellularity_2,orientation_0,orientation_1,orientation_2\n"
for idx, c in enumerate(HGSOC_testing_case_id_list):
    wrt_str += str(c) + ","
    for s in HGSOC_heatmap_array[idx, :]:
        wrt_str += str(s) + ","
    wrt_str = wrt_str[0:-1]
    wrt_str += "\n"
fp = open(os.path.join(output_dir, "HGSOC_hist_rec.csv"), "w")
fp.write(wrt_str)
fp.close()

# ovarall histogram
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(8, 6))
weights = np.ones_like(SBOT_fib_tsr_scores) / len(SBOT_fib_tsr_scores)
axs[0, 0].hist(SBOT_fib_tsr_scores, bins=3, histtype='bar', rwidth=0.8, color='white', fc='red', weights=weights,
               edgecolor='red')
axs[0, 0].set_title('SBOT_fibrosis')
axs[0, 0].set_xticks(range(3))
axs[0, 0].set_ylim([0, 1])
axs[0, 0].grid(True)
weights = np.ones_like(SBOT_cel_tsr_scores) / len(SBOT_cel_tsr_scores)
axs[0, 1].hist(SBOT_cel_tsr_scores, bins=3, histtype='bar', rwidth=0.8, color='white', fc='green', weights=weights,
               edgecolor='green')
axs[0, 1].set_title('SBOT_cellularity')
axs[0, 1].set_xticks(range(3))
axs[0, 1].set_ylim([0, 1])
axs[0, 1].grid(True)
weights = np.ones_like(SBOT_ori_tsr_scores) / len(SBOT_ori_tsr_scores)
axs[0, 2].hist(SBOT_ori_tsr_scores, bins=3, histtype='bar', rwidth=0.8, color='white', fc='blue', weights=weights,
               edgecolor='blue')
axs[0, 2].set_title('SBOT_orientation')
axs[0, 2].set_xticks(range(3))
axs[0, 2].set_ylim([0, 1])
axs[0, 2].grid(True)

weights = np.ones_like(HGSOC_fib_tsr_scores) / len(HGSOC_fib_tsr_scores)
axs[1, 0].hist(HGSOC_fib_tsr_scores, bins=3, histtype='bar', rwidth=0.8, color='white', fc='red', weights=weights,
               edgecolor='red')
axs[1, 0].set_title('HGSOC_fibrosis')
axs[1, 0].set_xticks(range(3))
axs[1, 0].set_ylim([0, 1])
axs[1, 0].grid(True)
weights = np.ones_like(HGSOC_cel_tsr_scores) / len(HGSOC_cel_tsr_scores)
axs[1, 1].hist(HGSOC_cel_tsr_scores, bins=3, histtype='bar', rwidth=0.8, color='white', fc='green', weights=weights,
               edgecolor='green')
axs[1, 1].set_title('HGSOC_cellularity')
axs[1, 1].set_xticks(range(3))
axs[1, 1].set_ylim([0, 1])
axs[1, 1].grid(True)
weights = np.ones_like(HGSOC_ori_tsr_scores) / len(HGSOC_ori_tsr_scores)
axs[1, 2].hist(HGSOC_ori_tsr_scores, bins=3, histtype='bar', rwidth=0.8, color='white', fc='blue', weights=weights,
               edgecolor='blue')
axs[1, 2].set_title('HGSOC_orientation')
axs[1, 2].set_xticks(range(3))
axs[1, 2].set_ylim([0, 1])
axs[1, 2].grid(True)
plt.show()

# fig, axs = plt.subplots(nrows=2, ncols=3)
# axs[0, 0].hist(SBOT_stroma_core_fib_tsr_scores, bins=3, histtype='bar', density='true', stacked='true', rwidth=0.8, color='white',weights=weights, edgecolor='red')
# axs[0, 0].set_title('SBOT_core_fibrosis')
# axs[0, 0].set_xticks(range(3))
# axs[0, 1].hist(SBOT_stroma_core_cel_tsr_scores, bins=3, histtype='bar', density='true', stacked='true', rwidth=0.8, color='white',weights=weights, edgecolor='red')
# axs[0, 1].set_title('SBOT_core_cellularity')
# axs[0, 1].set_xticks(range(3))
# axs[0, 2].hist(SBOT_stroma_core_ori_tsr_scores, bins=3, histtype='bar', density='true', stacked='true', rwidth=0.8, color='white',weights=weights, edgecolor='red')
# axs[0, 2].set_title('SBOT_core_orientation')
# axs[0, 2].set_xticks(range(3))
#
# axs[1, 0].hist(HGSOC_stroma_core_fib_tsr_scores, bins=3, histtype='bar', density='true', stacked='true', rwidth=0.8, color='white',weights=weights, edgecolor='red')
# axs[1, 0].set_title('HGSOC_core_fibrosis')
# axs[1, 0].set_xticks(range(3))
# axs[1, 1].hist(HGSOC_stroma_core_cel_tsr_scores, bins=3, histtype='bar', density='true', stacked='true', rwidth=0.8, color='white',weights=weights, edgecolor='red')
# axs[1, 1].set_title('HGSOC_core_cellularity')
# axs[1, 1].set_xticks(range(3))
# axs[1, 2].hist(HGSOC_stroma_core_ori_tsr_scores, bins=3, histtype='bar', density='true', stacked='true', rwidth=0.8, color='white',weights=weights, edgecolor='red')
# axs[1, 2].set_title('HGSOC_core_orientation')
# axs[1, 2].set_xticks(range(3))
# plt.show()
