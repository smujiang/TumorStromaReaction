import numpy as np
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk
import os
import openslide
from PIL import Image
import glob

#Step 1: create segmentation thumbnail
#TODO: refer to ./visualization/create_thumbnail.py
seg_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/segmentation"
wsi_dir = "/Jun_anonymized_dir/OvaryCancer/WSIs"
output_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/thumbnails"
reaction_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/reaction_prediction_NC"

reaction_case_IDs = set(os.listdir(reaction_dir))
thumb_case_IDs = set(os.listdir(output_dir))

case_list = list(reaction_case_IDs - thumb_case_IDs)

img_rows = 256
img_cols = 256
img_channel = 3
img_rescale_rate = 2

thumbnail_downsample = 128
patch_thumb_size = int(img_rows*img_rescale_rate/thumbnail_downsample)

def get_coord_from_fn(fn):
    img_fn = os.path.split(fn)[1]
    ele = img_fn.split("_")
    loc_x = int(ele[-2])
    loc_y = int(ele[-1][0:-4])
    return loc_x, loc_y

for case in case_list:
    print("Processing %s" % case)
    save_to_dir = os.path.join(output_dir, case)
    if not os.path.exists(save_to_dir):
        os.makedirs(save_to_dir)
    save_to = os.path.join(save_to_dir, case + "_segmentation.png")
    if not os.path.exists(save_to):
        print("     Saving segmentation thumbnail.")
        wsi_fn = os.path.join(wsi_dir, case + ".svs")
        wsi_obj = openslide.open_slide(wsi_fn)
        thumb_size = np.array(wsi_obj.dimensions) / thumbnail_downsample
        thumb_img = wsi_obj.get_thumbnail(thumb_size)
        # save segmentation thumbnail image
        seg_thumb_img_arr = np.ones((thumb_img.height, thumb_img.width, 4), dtype=np.uint8) + 255
        seg_img_fns = glob.glob(os.path.join(seg_dir, case, "*.png"))
        for seg_img_fn in seg_img_fns:
            loc_x, loc_y = get_coord_from_fn(seg_img_fn)
            x = int(loc_x/thumbnail_downsample)
            y = int(loc_y/thumbnail_downsample)
            seg_thumb = Image.open(seg_img_fn, 'r')
            xx = seg_thumb.resize((patch_thumb_size, patch_thumb_size))
            xx.putalpha(255)
            # print(y+patch_thumb_size, x+patch_thumb_size)
            # print(patch_thumb_size)
            seg_thumb_img_arr[y:y+patch_thumb_size, x:x+patch_thumb_size, :] = np.array(xx)
        Image.fromarray(seg_thumb_img_arr).save(save_to)
    else:
        print("     Segmentation thumbnail already exists")

seg_color_map = [[0, 255, 255], [255, 255, 0]]  # tumor & stroma
#Step 2:
def calculate_tumor_stroma_entangling_area(tumor_stroma_img_arr, seg_color_map):
    '''
    # get tumor core
    '''
    k = tumor_stroma_img_arr == seg_color_map[0]
    tumor = np.all(k, axis=2) * 1
    selem = disk(3)
    dilated = dilation(tumor, selem)
    tumor_core = erosion(dilated, selem)
    dilated_tumor_core = dilation(tumor_core, selem)

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
    return entangle_area

def get_relevant_patches(binary_img_arr, case_id, data_dir, patch_size=512, scale_rate=128, stride=512, ext=".jpg"):
    pos_indices = np.where(binary_img_arr > 0)
    loc_y = (np.array(pos_indices[0]) * scale_rate).astype(np.int)
    loc_x = (np.array(pos_indices[1]) * scale_rate).astype(np.int)
    loc_x_selected = []
    loc_y_selected = []
    x_lim = [min(loc_x), max(loc_x)]
    y_lim = [min(loc_y), max(loc_y)]
    for x in range(x_lim[0], x_lim[1], stride):
        for y in range(y_lim[0], y_lim[1], stride):
            x_idx = int(x / scale_rate)
            y_idx = int(y / scale_rate)
            x_idx_1 = int((x+patch_size) / scale_rate)
            y_idx_1 = int((y+patch_size) / scale_rate)
            if x_idx_1 >= binary_img_arr.shape[1]:
                x_idx_1 = x_idx
            if y_idx_1 >= binary_img_arr.shape[0]:
                y_idx_1 = y_idx
            if np.count_nonzero(binary_img_arr[y_idx:y_idx_1, x_idx:x_idx_1]) > 0:
                loc_x_selected.append(int(x))
                loc_y_selected.append(int(y))

    return_img_fn_list = []
    img_list = os.listdir(data_dir)
    for loc_x, loc_y in zip([loc_x_selected, loc_y_selected]):
        img_fn = case_id + "_" + str(loc_x) + "_" + str(loc_y) + ext
        if img_fn in img_list:
            return_img_fn_list.append(img_fn)
        else:
            raise Exception("File not in the folder")
    return return_img_fn_list










