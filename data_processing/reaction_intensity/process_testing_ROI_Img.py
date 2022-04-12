import os
import numpy as np
import io
import glob
from skimage.color import rgb2hsv
from PIL import Image
from skimage.color import rgb2lab


def filter_by_content_area(rgb_image_array, area_threshold=0.5, brightness=85):
    """
    Takes an RGB image array as input,
        converts into LAB space
        checks whether the brightness value exceeds the threshold
        returns a boolean indicating whether the amount of tissue > minimum required

    :param rgb_image_array:
    :param area_threshold:
    :param brightness:
    :return:
    """
    # TODO: Alternative tissue detectors, not just RGB->LAB->Thresh
    # rgb_image_array[np.any(rgb_image_array == [0, 0, 0], axis=-1)] = [255, 255, 255]
    lab_img = rgb2lab(rgb_image_array)
    l_img = lab_img[:, :, 0]
    binary_img_array_1 = np.array(0 < l_img)
    binary_img_array_2 = np.array(l_img < brightness)
    binary_img = np.logical_and(binary_img_array_1, binary_img_array_2) * 255
    tissue_size = np.where(binary_img > 0)[0].size
    tissue_ratio = tissue_size * 3 / rgb_image_array.size  # 3 channels
    if tissue_ratio > area_threshold:
        return True
    else:
        return False



'''
Sample a serials of patches from an image array
img_arr:    image array, type: numpy ndarray
patch_size: height and width, eg. [50,50]
step:       sampling step, eg. [50,50]
area_range: define the sampling start and end with where, [start_x_cor,end_x_cor,start_y_cor,end_y_cor],
            eg. [10,810,100,900]
# Reference: sklearn.feature_extraction.image.extract_patches_2d()            
'''
def slide_img_sampling(img_arr, patch_size, step, area_range=None):
    '''

    :param img_arr:
    :param patch_size:
    :param step:
    :param area_range:
    :return:
    '''
    img_size = img_arr.shape
    channels = img_size[2]
    if area_range is None:
        w_min = 0
        w_max = img_size[1]
        h_min = 0
        h_max = img_size[0]
    else:
        w_min = area_range[0]
        w_max = area_range[1]
        h_min = area_range[2]
        h_max = area_range[3]
    if w_min < 0 | h_min < 0 | w_max > img_size[1] | h_max > img_size[0]:
        print("area_range parameters error.")
        return False
    ex_w = range(w_min, w_max - patch_size[1] + step[1], step[1])
    ex_h = range(h_min, h_max - patch_size[0] + step[0], step[0])
    nd_array_patches = np.empty((len(ex_w) * len(ex_h), patch_size[0], patch_size[1], channels), dtype=np.uint8,
                                order='C')
    offsets = []
    patch_num = 0
    for h in ex_h:
        for w in ex_w:
            offsets.append([w, h])
            patch_arr = img_arr[h:(h + patch_size[0]), w:(w + patch_size[1]), :]
            nd_array_patches[patch_num, :, :, :] = patch_arr
            patch_num += 1
    return nd_array_patches, offsets


def get_label_by_color(color_map, color):
    for i in range(color_map.shape[0]):
        for j in range(color_map.shape[1]):
            if all(color_map[i, j] == np.array(color)):
                return i, j
    return -1, -1


def write_to_csv(fp, img_fn_list, label_list):
    for idx, img_fn in enumerate(img_fn_list):
        labels = label_list[3*idx:3*(idx+1)]
        Fibrosis_score = labels[0][1]
        Cellularity_score = labels[1][1]
        Orientation_score = labels[2][1]
        fp.write(str(Fibrosis_score) + "," + str(Cellularity_score) + "," + str(Orientation_score) + "," + img_fn + "\n")

# VALIDATION = True
# SAVE_IMG = True
# VALIDATION = False
# SAVE_IMG = False
# SAVE_TRAINING = True

count_all_samples = 0
count_eligible_samples = 0

if __name__ == "__main__":
    patch_sz = [512, 512]
    rescale_to = [256, 256]
    stride = [128, 128]
    # stride = [32, 32]

    data_root = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/Testing_ROIs"
    data_out_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/Testing_ROI_Patches"

    case_list = [""]

    for c in case_list:
        img_fn_list = glob.glob(os.path.join(data_root, c, "*.jpg"))
        for img_fn in img_fn_list:
            print("Processing %s" % img_fn)
            img_arr = np.array(Image.open(img_fn))
            img_arr_patches, offsets = slide_img_sampling(img_arr, patch_sz, stride, area_range=None)
            for patch, offset in zip(img_arr_patches, offsets):
                count_all_samples += 1
                # if count_all_samples > 100:  # For Debug
                #     break
                if filter_by_content_area(patch):  # Check if the image patch has enough (50%+) tissue or not
                    temp_fn = img_fn.replace(data_root, data_out_dir)
                    patch_img_fn = temp_fn.replace(".jpg", "_" + str(offset[0]) + "_" + str(offset[1]) + ".jpg")
                    patch_out_dir = os.path.split(patch_img_fn)[0]
                    if not os.path.exists(patch_out_dir):
                        os.makedirs(patch_out_dir)

                    patch_Img = Image.fromarray(patch)
                    count_eligible_samples += 1
                    patch_Img.resize(rescale_to).save(patch_img_fn)
