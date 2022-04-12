from PIL import Image
import os
import numpy as np
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk
import matplotlib.pyplot as plt

thumbnail_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/thumbnails"
output_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/result_analysis"

shuffled_training_csv_file = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/PatchSampling/training_with_SBOT_cases.csv"
shuffled_validation_csv_file = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/PatchSampling/validation_with_SBOT_cases.csv"

fn_seg_ext = "_segmentation.png"
fn_tsr_fib_ext = "_Fibrosis_TSR-score.png"
fn_tsr_cel_ext = "_Cellularity_TSR-score.png"
fn_tsr_ori_ext = "_Orientation_TSR-score.png"

ds = 128  # down sampling ratio
seg_color_map = [[0, 255, 255], [255, 255, 0]]  # tumor & stroma
color_map = [[[100, 0, 0], [180, 0, 0], [255, 0, 0]],  # Fibrosis: 0, 1, 2
             [[0, 100, 0], [0, 180, 0], [0, 255, 0]],
             [[0, 0, 100], [0, 0, 180], [0, 0, 255]]]
Class_label_list = ["Fibrosis", "Cellularity", "Orientation"]

SBOT_case_list = ["OCMC-016","OCMC-017","OCMC-018","OCMC-019","OCMC-020","OCMC-021", "OCMC-022", "OCMC-023",
                  "OCMC-024","OCMC-025","OCMC-026","OCMC-027","OCMC-028","OCMC-029","OCMC-030"]

batch1_case_id_list_fn = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/result_analysis/batch1_case_id.txt"
fp = open(batch1_case_id_list_fn, 'r')
lines = fp.readlines()
batch_1_cases = []
for i in lines:
    if len(i.strip()) > 0:
        batch_1_cases.append(i.strip())

all_cases_list = sorted(batch_1_cases + SBOT_case_list)

def decode_color_to_score(color, color_list, tolerance=20):
    for idx, i in enumerate(color_list):
        if i[0] - tolerance < color[0] < i[0] + tolerance \
                and i[1] - tolerance < color[1] < i[1] + tolerance \
                and i[2] - tolerance < color[2] < i[2] + tolerance:
            return idx
    return -1

training_lines = open(shuffled_training_csv_file, 'r').readlines()
validate_lines = open(shuffled_validation_csv_file, 'r').readlines()
all_lines = training_lines + validate_lines
training_SBOT_case_list = set()
for i in all_lines[1:]:
    ele = i.split(",")
    # if "OCMC-" in ele[3]:
    case_id = os.path.split(ele[3])[1].split("_")[0]
    training_SBOT_case_list.add(case_id)

#TODO: 1) get stroma patch cnt
#      2) get score distribution
#      3)

def get_tsr_scores(case_id, training_patch_locations=None):
    case_fib_tsr_score_lists = []
    case_cel_tsr_score_lists = []
    case_ori_tsr_score_lists = []
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

    selem = disk(3)
    dilated = dilation(tumor, selem)
    tumor_core = erosion(dilated, selem)
    # plt.imshow(tumor_core, cmap="gray")
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
    # plt.show()

    # print(tumor.shape)

    '''
    # get entangling area
    '''
    selem = disk(8)
    dilated_tumor_core = dilation(tumor_core, selem)
    # plt.imshow(dilated_tumor_core, cmap="gray")
    # plt.show()
    eroded_tumor_core = erosion(tumor_core, selem)
    # plt.imshow(eroded_tumor_core, cmap="gray")
    # plt.show()
    entangle_area = np.logical_xor(dilated_tumor_core, eroded_tumor_core)
    # plt.imshow(entangle_area, cmap="gray")
    # plt.show()

    entangle_area = np.logical_and(entangle_area, stroma_core)
    entangle_area = np.logical_and(entangle_area, dilated_tumor_core)
    # plt.imshow(entangle_area, cmap="gray")
    # plt.show()
    entangle_area_patch_cnt = np.count_nonzero(entangle_area)

    # get all stroma locations
    loc = np.where(stroma == 1)

    fn_tsr_fib = os.path.join(thumbnail_dir, case_id, case_id + fn_tsr_fib_ext)
    tsr_fib = np.array(Image.open(fn_tsr_fib, 'r'))[:, :, 0:3]
    fib_all_scores_encoded = tsr_fib[stroma == 1]

    fn_tsr_cel = os.path.join(thumbnail_dir, case_id, case_id + fn_tsr_cel_ext)
    tsr_cel = np.array(Image.open(fn_tsr_cel, 'r'))[:, :, 0:3]
    cel_all_scores_encoded = tsr_cel[stroma == 1]

    fn_tsr_ori = os.path.join(thumbnail_dir, case_id, case_id + fn_tsr_ori_ext)
    tsr_ori = np.array(Image.open(fn_tsr_ori, 'r'))[:, :, 0:3]
    ori_all_scores_encoded = tsr_ori[stroma == 1]

    for idx, i in enumerate(fib_all_scores_encoded):
        score = decode_color_to_score(i, color_map[0])
        if score is not -1:
            if training_patch_locations is not None:
                # print([loc[0][idx], loc[1][idx]])
                # training_patch_locations.append([loc[0][idx], loc[1][idx]])
                if not ([loc[0][idx], loc[1][idx]] in training_patch_locations):
                    case_fib_tsr_score_lists.append(score)
            else:
                case_fib_tsr_score_lists.append(score)
        else:
            print(case_id + ":" + str(i))

    for idx, i in enumerate(cel_all_scores_encoded):
        score = decode_color_to_score(i, color_map[1])
        if score is not -1:
            if training_patch_locations is not None:
                if not ([loc[0][idx], loc[1][idx]] in training_patch_locations):
                    case_cel_tsr_score_lists.append(score)
            else:
                case_cel_tsr_score_lists.append(score)
        else:
            print(case_id + ":" + str(i))
    for idx, i in enumerate(ori_all_scores_encoded):
        score = decode_color_to_score(i, color_map[2])
        if score is not -1:
            if training_patch_locations is not None:
                if not ([loc[0][idx], loc[1][idx]] in training_patch_locations):
                    case_ori_tsr_score_lists.append(score)
            else:
                case_ori_tsr_score_lists.append(score)
        else:
            print(case_id + ":" + str(i))
    return case_fib_tsr_score_lists, case_cel_tsr_score_lists, case_ori_tsr_score_lists, entangle_area_patch_cnt

wrt_str = "case_id,stroma_patch_total_cnt,entangle_patch_cnt,training_patch_cnt,train_Fibrosis_0_cnt,train_Fibrosis_1_cnt,train_Fibrosis_2_cnt," \
          "train_cellularity_0_cnt,train_cellularity_1_cnt,train_cellularity_2_cnt,train_orientation_0_cnt,train_orientation_1_cnt,train_orientation_2_cnt," \
          "testing_patch_cnt,test_Fibrosis_0_cnt,test_Fibrosis_1_cnt,test_Fibrosis_2_cnt,test_cellularity_0_cnt,test_cellularity_1_cnt,test_cellularity_2_cnt," \
          "test_orientation_0_cnt,test_orientation_1_cnt,test_orientation_2_cnt\n"
for idx, case_id in enumerate(all_cases_list):
    print("processing %s. %d/%d" % (case_id, idx + 1, len(all_cases_list)))
    seg_img_fn = os.path.join(thumbnail_dir, case_id, case_id+fn_seg_ext)
    tumor_stroma_img = Image.open(seg_img_fn)
    tumor_stroma_img_arr = np.array(tumor_stroma_img)[:, :, 0:3]
    k = tumor_stroma_img_arr == seg_color_map[1]
    stroma = np.all(k, axis=2) * 1
    totoal_stroma_cnt = np.count_nonzero(stroma)
    training_SBOT_locations = []
    if case_id in training_SBOT_case_list:
        for l in all_lines:
            ele = l.split(",")
            if case_id in ele[3]:
                training_loc_ele = os.path.split(ele[3])[1].replace(".jpg", "").split("_")
                training_loc = [int(int(training_loc_ele[1])/ds), int(int(training_loc_ele[2].strip())/ds)]
                training_SBOT_locations.append(training_loc)
        training_patch_cnt = len(training_SBOT_locations)
        train_Fibrosis_0_cnt = training_patch_cnt
        train_Fibrosis_1_cnt = 0
        train_Fibrosis_2_cnt = 0
        train_cellularity_0_cnt = training_patch_cnt
        train_cellularity_1_cnt = 0
        train_cellularity_2_cnt = 0
        train_orientation_0_cnt = training_patch_cnt
        train_orientation_1_cnt = 0
        train_orientation_2_cnt = 0

        testing_patch_cnt = totoal_stroma_cnt - training_patch_cnt
    else:
        training_patch_cnt = 0
        train_Fibrosis_0_cnt = 0
        train_Fibrosis_1_cnt = 0
        train_Fibrosis_2_cnt = 0
        train_cellularity_0_cnt = 0
        train_cellularity_1_cnt = 0
        train_cellularity_2_cnt = 0
        train_orientation_0_cnt = 0
        train_orientation_1_cnt = 0
        train_orientation_2_cnt = 0

        testing_patch_cnt = totoal_stroma_cnt

    case_fib_tsr_score_lists, case_cel_tsr_score_lists, case_ori_tsr_score_lists, entangle_area_patch_cnt = get_tsr_scores(case_id, training_SBOT_locations)

    hist_fibrosis, _ = np.histogram(case_fib_tsr_score_lists, bins=3, range=(0, 2))
    hist_cellularity, _ = np.histogram(case_cel_tsr_score_lists, bins=3, range=(0, 2))
    hist_orientation, _ = np.histogram(case_ori_tsr_score_lists, bins=3, range=(0, 2))
    #
    test_Fibrosis_0_cnt, test_Fibrosis_1_cnt, test_Fibrosis_2_cnt = hist_fibrosis
    test_cellularity_0_cnt, test_cellularity_1_cnt, test_cellularity_2_cnt = hist_cellularity
    test_orientation_0_cnt, test_orientation_1_cnt, test_orientation_2_cnt = hist_orientation

    wrt_str += case_id + "," + str(totoal_stroma_cnt) + "," + str(entangle_area_patch_cnt) + "," + str(training_patch_cnt) + "," + \
               str(train_Fibrosis_0_cnt)+ "," + str(train_Fibrosis_1_cnt)+ "," + str(train_Fibrosis_2_cnt)+ "," + \
               str(train_cellularity_0_cnt)+ "," + str(train_cellularity_1_cnt)+ "," + str(train_cellularity_2_cnt)+ "," + \
               str(train_orientation_0_cnt)+ "," + str(train_orientation_1_cnt)+ "," + str(train_orientation_2_cnt)+ "," + \
               str(testing_patch_cnt) + "," +\
               str(test_Fibrosis_0_cnt)+ "," + str(test_Fibrosis_1_cnt)+ "," + str(test_Fibrosis_2_cnt)+ "," +\
               str(test_cellularity_0_cnt)+ "," + str(test_cellularity_1_cnt)+ "," + str(test_cellularity_2_cnt)+ "," + \
               str(test_orientation_0_cnt)+ "," + str(test_orientation_1_cnt)+ "," + str(test_orientation_2_cnt) + "\n"

save_to = os.path.join(output_dir, "All_TSR_patch_statistics.csv")
fp = open(save_to, 'w')
fp.write(wrt_str)
fp.close()

print("Done")


