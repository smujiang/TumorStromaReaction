import numpy as np
from tensorflow.keras.applications import VGG16
import sys,os,time
import glob
from PIL import Image
import matplotlib.pyplot as plt
import argparse
sys.path.insert(1, "./utils")
import load_configuration

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
segmentation_output_dir = conf.segmentation_output_dir
stroma_color = eval(conf.tumor_stroma_color)[1]
stroma_reaction_score_colors = eval(conf.stroma_reaction_score_colors)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

VGG16_MODEL = VGG16(include_top=True, weights=None, input_tensor=None, input_shape=IMG_SHAPE,
                    pooling=None, classes=num_classes)

for idx, m in enumerate(all_class_list):
    # print("Evaluating %s on testing data" % m)
    model_ckpt = os.path.join(model_weights_path, m, all_model_list[idx])
    VGG16_MODEL.load_weights(model_ckpt)
    for case in case_list:
        save_to_dir = os.path.join(output_dir, case, m)
        if os.path.exists(save_to_dir):
            print("Results may already exist")

        if os.path.exists(os.path.join(patch_root, case)) and os.path.exists(os.path.join(segmentation_output_dir, case)):
            print("Evaluating %s on testing case %s " % (m, case))
            start = time.time()

            f_list = glob.glob(os.path.join(patch_root, case, "*.jpg"))
            test_x = np.empty((1, img_rows, img_cols, img_channel), dtype=np.float32)
            for f in f_list:
                save_to = os.path.join(output_dir, case, m, os.path.split(f)[1].replace(".jpg", ".png"))
                if not os.path.exists(save_to):
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
                            #TODO: calculate intersection of stroma mask and stroma reaction intensity
                            max_val = max(out[0])
                            max_idx = list(out[0]).index(max_val)
                            pixel_index = np.all(mask_img_arr.reshape(-1, 3) == stroma_color, axis=1)
                            result.reshape(-1, 3)[pixel_index] = stroma_reaction_score_colors[idx][max_idx]
                            result = result.reshape(img_rows, img_cols, img_channel)
                            # print(pixel_index)
                            # print(stroma_reaction_score_colors[idx][max_idx])

                            # save_to = os.path.join(output_dir, case, m, os.path.split(f)[1].replace(".jpg", ".png"))

                            if not os.path.exists(os.path.join(output_dir, case, m)):
                                os.makedirs(os.path.join(output_dir, case, m))
                            Img = Image.fromarray(result)
                            Img.putalpha(255)
                            Img.save(save_to)

            end = time.time()
            hours, rem = divmod(end-start, 3600)
            minutes, seconds = divmod(rem, 60)
            print("Time elapse:{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
        else:
            print("Can't locate stroma segmentation results for case: %s" % case)
        # else:
        #     print("Results may already exist")
