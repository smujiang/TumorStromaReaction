import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.data import Dataset
import sys,os,time
import glob
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from skimage.color import rgb2lab
sys.path.insert(1, "./utils")
from utils import load_configuration
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk
from scipy import ndimage

seg_color_map = [[0, 255, 255], [255, 255, 0]]  # tumor & stroma

def calculate_stroma_area(img_arr, stroma_color):
    h, w = img_arr.shape[:2]
    # Get list of unique colours...
    colours, counts = np.unique(img_arr.reshape(-1, 3), axis=0, return_counts=True)
    # Iterate through unique colours
    for index, colour in enumerate(colours):
        if np.all(stroma_color == colour):
            count = counts[index]
            proportion = (100 * count) / (h * w)
            return proportion
    return 0

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


def get_relevant_patches(binary_img_arr, wsi_thumb_img, case_id, data_dir, patch_size=512, scale_rate=128, stride=512, ext=".jpg"):
    lab_img = rgb2lab(wsi_thumb_img)
    l_img = lab_img[:, :, 0]
    # tissue is darker than background, recommend threshold value: 85
    binary_img_array_1 = np.array(0 < l_img)
    binary_img_array_2 = np.array(l_img < 85)
    thumb_img_mask = np.logical_and(binary_img_array_1, binary_img_array_2)
    thumb_img_mask = ndimage.binary_erosion(thumb_img_mask)
    binary_img_arr = np.logical_and(binary_img_arr, thumb_img_mask)
    pos_indices = np.where(binary_img_arr > 0)
    loc_y = (np.array(pos_indices[0]) * scale_rate).astype(np.int32)
    loc_x = (np.array(pos_indices[1]) * scale_rate).astype(np.int32)
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
    img_list = os.listdir(os.path.join(data_dir, case_id))
    for loc_x, loc_y in zip(loc_x_selected, loc_y_selected):
        # img_fn = case_id + "_" + str(loc_x) + "_" + str(loc_y) + ext
        img_loc_range_x = range(loc_x - patch_size, loc_x + patch_size)
        img_loc_range_y = range(loc_y - patch_size, loc_y + patch_size)
        Found_IMG = False
        for i_fn in img_list:
            # get locations from image file name
            ele = i_fn.split("_")
            i_x = int(ele[-2])
            i_y = int(ele[-1][0:-4])
            ''''''
            if i_x in img_loc_range_x and i_y in img_loc_range_y:
                # print("Found the image patch")
                Found_IMG = True
                return_img_fn_list.append(os.path.join(data_dir, case_id, i_fn))
                # break
                # print(img_fn)
        if not Found_IMG:
            print("Warning: File not in the folder, ignore this image")
    return set(return_img_fn_list)

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config",
                    dest='conf_fn',
                    required=True,
                    help="file name for configuration")
args = parser.parse_args()
conf = load_configuration.CombinedTest(args.conf_fn)
all_class_list = eval(conf.all_class_list)
labels = eval(conf.labels)
all_model_list = eval(conf.all_model_list)
output_dir = conf.output_dir
num_classes = len(labels)
IMG_SHAPE = eval(conf.IMG_SHAPE)
model_weights_path = conf.model_weights_path
case_list = eval(conf.case_list)
patch_root = conf.patch_dir
img_rows, img_cols, img_channel = tuple(IMG_SHAPE)
seg_mask_thumb_dir = conf.seg_mask_thumb_dir
segmentation_output_dir = conf.segmentation_output_dir
stroma_color = eval(conf.tumor_stroma_color)[1]
stroma_reaction_score_colors = eval(conf.stroma_reaction_score_colors)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

VGG16_MODEL = VGG16(include_top=True, weights=None, input_tensor=None, input_shape=IMG_SHAPE,
                    pooling=None, classes=num_classes)



def img_gen(fn_list, batch_size):
    resd = len(f_list) % batch_size
    f_list.append(f_list[-1] * (batch_size - resd))
    batch_cnt = len(fn_list)/batch_size
    batch_index = 0
    for i in range(batch_cnt):
        yield f_list[i*batch_size, (i+1)*batch_size]


for idx, m in enumerate(all_class_list):
    # print("Evaluating %s on testing data" % m)
    model_ckpt = os.path.join(model_weights_path, m, all_model_list[idx])
    VGG16_MODEL.load_weights(model_ckpt)
    for case in case_list:
        save_to_dir = os.path.join(output_dir, case, m)
        if not os.path.exists(save_to_dir):
            if os.path.exists(os.path.join(patch_root, case)) and os.path.exists(
                    os.path.join(segmentation_output_dir, case)):
                print("Evaluating %s on testing case %s " % (m, case))
                start = time.time()

                from tensorflow.keras.preprocessing.image import ImageDataGenerator

                batch_size = 8
                test_datagen = ImageDataGenerator(rescale=1. / 255)
                target_size = (256, 256)
                test_generator = test_datagen.flow_from_directory(os.path.join(patch_root, case),
                                                                  target_size=target_size,
                                                                    batch_size=batch_size)

                for test_gen in test_generator:
                    idx = (test_gen.batch_index - 1) * test_gen.batch_size
                    print(test_gen.filenames[idx: idx + test_gen.batch_size])

                f_list = glob.glob(os.path.join(patch_root, case, "*.jpg"))
                resd = len(f_list)%batch_size
                f_list.append(f_list[-1]*(batch_size-resd))  # add some elements to the list so that all the batches are full

                for


                    'data/train/',
                    target_size=target_size,
                    batch_size=32)

                test_x = np.empty((1, img_rows, img_cols, img_channel), dtype=np.float32)
                for f in f_list:
                    #
                    # find corresponding mask image from Mask RCNN segmentation, calculate stroma area from prediction mask image
                    mask_fn = os.path.join(segmentation_output_dir, case, os.path.split(f)[1].replace(".jpg", ".png"))
                    if os.path.exists(mask_fn):
                        mask_img_arr = np.array(Image.open(mask_fn, 'r'))[:, :, 0:3]
                        # plt.imshow(mask_img_arr)
                        # plt.show()
                        stroma_area = calculate_stroma_area(mask_img_arr, stroma_color)
                        result = np.zeros([img_rows, img_cols, img_channel], dtype=np.uint8) + 255
                        if stroma_area > 0.5:
                            test_x[0, :, :, :] = np.array(Image.open(f, 'r')).astype(np.float32) / 255.0
                            out = VGG16_MODEL.predict(test_x)
                            # TODO: calculate intersection of stroma mask and stroma reaction intensity
                            max_val = max(out[0])
                            max_idx = list(out[0]).index(max_val)
                            pixel_index = np.all(mask_img_arr.reshape(-1, 3) == stroma_color, axis=1)
                            result.reshape(-1, 3)[pixel_index] = stroma_reaction_score_colors[idx][max_idx]
                            result = result.reshape(img_rows, img_cols, img_channel)
                            # print(pixel_index)
                            # print(stroma_reaction_score_colors[idx][max_idx])

                            save_to = os.path.join(output_dir, case, m, os.path.split(f)[1].replace(".jpg", ".png"))

                            if not os.path.exists(os.path.join(output_dir, case, m)):
                                os.makedirs(os.path.join(output_dir, case, m))
                            Img = Image.fromarray(result)
                            Img.putalpha(255)
                            Img.save(save_to)

                end = time.time()
                hours, rem = divmod(end - start, 3600)
                minutes, seconds = divmod(rem, 60)
                print("Time elapse:{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
            else:
                print("Can't locate stroma segmentation results for case: %s" % case)
        else:
            print("Results may already exist")
