import multiprocessing
from PIL import Image
import os
import numpy as np
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk
import matplotlib.pyplot as plt
import skimage
from scipy import ndimage

thumbnail_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/thumbnails"
output_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/result_analysis/cell_rois"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

fn_seg_ext = "_segmentation.png"
# fn_tsr_fib_ext = "_Fibrosis_TSR-score.png"
# fn_tsr_cel_ext = "_Cellularity_TSR-score.png"
# fn_tsr_ori_ext = "_Orientation_TSR-score.png"

seg_color_map = [[0, 255, 255], [255, 255, 0]]  # tumor & stroma
color_map = [[[100, 0, 0], [180, 0, 0], [255, 0, 0]],  # Fibrosis: 0, 1, 2
             [[0, 100, 0], [0, 180, 0], [0, 255, 0]],
             [[0, 0, 100], [0, 0, 180], [0, 0, 255]]]
Class_label_list = ["Fibrosis", "Cellularity", "Orientation"]

# HGSOC_case_id_list_1 = ["OCMC-001", "OCMC-002", "OCMC-003", "OCMC-004", "OCMC-005", "OCMC-006", "OCMC-007", "OCMC-008", "OCMC-009", "OCMC-010", "OCMC-011", "OCMC-012", "OCMC-013", "OCMC-014", "OCMC-015"]
#
# SBOT_case_id_list = ["OCMC-016", "OCMC-017", "OCMC-018", "OCMC-019", "OCMC-020", "OCMC-021", "OCMC-022",
#                      "OCMC-023", "OCMC-024", "OCMC-025", "OCMC-026", "OCMC-027", "OCMC-028", "OCMC-029", "OCMC-030"]
# HGSOC_case_id_list = [""]
#
# SBOT_training = ["OCMC-016", "OCMC-018", "OCMC-020", "OCMC-022", "OCMC-024", "OCMC-026"]
# HGSOC_training = [""]

# SBOT_training = []
# HGSOC_training = []

roi_cnt = 5
wsi_scale = 128
def decode_color_to_score(color, color_list, tolerance=10):
    for idx, i in enumerate(color_list):
        if i[0] - tolerance < color[0] < i[0] + tolerance \
                and i[1] - tolerance < color[1] < i[1] + tolerance \
                and i[2] - tolerance < color[2] < i[2] + tolerance:
            return idx
    return -1

# SBOT_testing_case_id_list = []
# for idx, case_id in enumerate(SBOT_case_id_list):
#     case_SBOT_fib_tsr_scores = []
#     case_SBOT_cel_tsr_scores = []
#     case_SBOT_ori_tsr_scores = []
#     if case_id not in SBOT_training:
#         SBOT_testing_case_id_list.append(case_id)
#         print("processing %s. %d/%d" % (case_id, idx + 1, len(SBOT_case_id_list)))
#         seg_fn = os.path.join(thumbnail_dir, case_id, case_id + fn_seg_ext)
#         tumor_stroma_img = Image.open(seg_fn, 'r')
#         tumor_stroma_img_arr = np.array(tumor_stroma_img)[:, :, 0:3]
#
#         '''
#         # get tumor core
#         '''
#         k = tumor_stroma_img_arr == seg_color_map[0]
#         tumor = np.all(k, axis=2) * 1
#         # # plt.imshow(tumor, cmap="gray")
#         # # plt.show()
#         #
#         selem = disk(3)
#         dilated = dilation(tumor, selem)
#         tumor_core = erosion(dilated, selem)
#         # plt.imshow(tumor_core, cmap="gray")
#         # plt.axis(False)
#         # plt.show()
#         dilated_tumor_core = dilation(tumor_core, selem)
#         # plt.imshow(dilated_tumor_core, cmap="gray")
#         # plt.axis(False)
#         # plt.show()
#
#         '''
#         #get stroma core
#         '''
#         k = tumor_stroma_img_arr == seg_color_map[1]
#         stroma = np.all(k, axis=2) * 1
#         # plt.imshow(stroma, cmap="gray")
#         # plt.show()
#
#         selem = disk(3)
#         dilated = dilation(stroma, selem)
#         stroma_core = erosion(dilated, selem)
#         # plt.imshow(stroma_core, cmap="gray")
#         # plt.axis(False)
#         # plt.show()
#
#         # print(tumor.shape)
#
#         '''
#         # get entangling area
#         '''
#         selem = disk(8)
#         dilated_stroma_core = dilation(stroma_core, selem)
#         # plt.imshow(dilated_stroma_core, cmap="gray")
#         # plt.axis(False)
#         # plt.show()
#         eroded_stroma_core = erosion(stroma_core, selem)
#         # plt.imshow(eroded_stroma_core, cmap="gray")
#         # plt.axis(False)
#         # plt.show()
#         entangle_area = np.logical_xor(dilated_stroma_core, eroded_stroma_core)
#         # plt.imshow(entangle_area, cmap="gray")
#         # plt.axis(False)
#         # plt.show()
#
#         entangle_area = np.logical_and(entangle_area, stroma_core)
#         entangle_area = np.logical_and(entangle_area, dilated_tumor_core)
#         plt.imshow(entangle_area, cmap="gray")
#         plt.axis(False)
#         plt.show()
#
#         # print(entangle_area.shape)
#         # h, w = entangle_area.shape


# case_id_list = HGSOC_case_id_list + SBOT_case_id_list + HGSOC_case_id_list_1 + SBOT_training + HGSOC_training

# def get_list_from_folder_complimentary(dir_whole, dir_sub):
#     case_ids_1 = os.listdir(dir_whole)
#     case_ids_2 = os.listdir(dir_sub)
#     ids = [f for f in case_ids_1 if f not in case_ids_2]
#     return ids
#
# case_id_list = get_list_from_folder_complimentary(thumbnail_dir, output_dir)

case_id_list = sorted(os.listdir(thumbnail_dir))

# HGSOC_testing_case_id_list = []
# for idx, case_id in enumerate(case_id_list):
    # if case_id not in HGSOC_training:
def process_cases(case_id_tuple):
    idx, case_id = case_id_tuple
    print("Processing %s" % case_id)
    save_to = os.path.join(output_dir, case_id + "_roi_box.csv")
    if not os.path.exists(save_to):
        print("")
        fp = open(save_to, 'w')
        wrt_str = "top_left_x, top_left_y, bottom_right_x, bottom_right_y\n"
        # HGSOC_testing_case_id_list.append(case_id)
        # print("processing %s. %d/%d" % (case_id, idx + 1, len(case_id_list)))
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
        labeled_image, label_cnt = skimage.morphology.label(entangle_area, connectivity=2, return_num=True)
        if label_cnt >= roi_cnt:
            cc_area = np.bincount(labeled_image.flat)[1:]
            # https://stackoverflow.com/questions/16878715/how-to-find-the-index-of-n-largest-elements-in-a-list-or-np-array-python
            cc_area_selected = np.argsort(cc_area)[-roi_cnt:]  # find the index of n largest elements in a list or np.array
            for cc in cc_area_selected:
                largestCC = labeled_image == cc+1
                # plt.imshow(largestCC, cmap="gray")
                # plt.axis(False)
                # plt.show()

                slice_y, slice_x = ndimage.find_objects(largestCC)[0]
                tl_x, br_x = slice_x.start*wsi_scale, slice_x.stop*wsi_scale
                tl_y, br_y = slice_y.start*wsi_scale, slice_y.stop*wsi_scale
                wrt_str += str(tl_x) + "," + str(tl_y) + "," + str(br_x) + "," + str(br_y) + "\n"
        else:  # TODO: what if there isn't ROI for selection
            print("")

        fp.write(wrt_str)
        fp.close()
    else:
        print("")


a_pool = multiprocessing.Pool()
a_pool.map(process_cases, enumerate(case_id_list))




