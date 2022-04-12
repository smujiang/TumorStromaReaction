from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
import os
import numpy as np
from PIL import Image
import time

import sys
import argparse
sys.path.insert(1, "./utils")
import load_configuration

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config",
                    dest='conf_fn',
                    required=True,
                    help="file name for configuration")
args = parser.parse_args()
conf = load_configuration.TestSegmentationConfig(args.conf_fn)

save_validation = False
MODEL_DIR = conf.model_dir
id_text_list = eval(conf.class_text_list)
color_list = eval(conf.color_list)



class TumorStromaConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "tumor_stroma"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + len(id_text_list)  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

class TumorStromaInferenceConfig(TumorStromaConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7
    BATCH_SIZE = 1

inference_config = TumorStromaInferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(model_out_dir, ".h5 file name here")
model_path = model.find_last()
# model_path = conf.model_fn

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

data_root_dir = conf.patch_dir
data_out_root_dir = conf.output_dir
testing_cases = eval(conf.case_list)

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

BATCH_SIZE = 8
for case in testing_cases:
    if os.path.exists(os.path.join(data_root_dir, case)):
        print("Tumor-stroma segmentation in case: %s" % case)
        start = time.time()

        img_fns = os.listdir(os.path.join(data_root_dir, case))
        # for i in img_fns:
        for ifn_s in batch(img_fns, BATCH_SIZE):
            true_ifn_s = ifn_s.copy()
            imgs = []
            for i in ifn_s:
                img = np.array(Image.open(os.path.join(data_root_dir, case, i)))
                imgs.append(img)
            while len(imgs) < BATCH_SIZE:
                img = np.array(Image.open(os.path.join(data_root_dir, case, ifn_s[0])))
                imgs.append(img)
                true_ifn_s.append(ifn_s[0])

            results = model.detect(imgs, verbose=0)
            for res_idx, r in enumerate(results):
                mask_img = np.ones([256, 256, 3], dtype=np.uint8) * 255
                for class_idx, class_id in enumerate(r['class_ids']):
                    class_txt = id_text_list[class_id-1]
                    mask_img[r['masks'][:, :, class_idx]] = color_list[class_id-1]
                    # mask_img = r['masks'][:, :, class_idx].astype(np.uint8) * 255
                    # print(mask_img.shape)
                save_to = os.path.join(data_out_root_dir, case, true_ifn_s[res_idx].replace(".jpg", ".png"))
                if not os.path.exists(os.path.join(data_out_root_dir, case)):
                    os.makedirs(os.path.join(data_out_root_dir, case))
                if save_validation:
                    img_mask_side_by_side = np.ones([256, 512, 3], dtype=np.uint8) * 255
                    img_mask_side_by_side[:, 0:256, :] = imgs[res_idx]
                    img_mask_side_by_side[:, 256:, :] = mask_img
                    # Image.fromarray(mask_img).save(save_to)
                    Image.fromarray(img_mask_side_by_side).save(save_to)
                else:
                    Image.fromarray(mask_img).save(save_to)

        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Time elapsed:{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    else:
        print("Can't locate extracted patches for case: %s" % case)









