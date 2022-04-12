import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
# Root directory of the project
# ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
# sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
import skimage.io
from skimage import measure
from mrcnn.model import log

import matplotlib
# sys.path.append(os.path.abspath('../../../../Evaluation'))
from label_csv_manager import label_color_CSVManager

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

model_out_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/TumorStroma_MaskRCNN/model_larger_dataset_low_resolution"
train_dataset_root_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/TumorStroma_MaskRCNN/ROI_Masks_out_train_val_test_split/train"
val_dataset_root_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/TumorStroma_MaskRCNN/ROI_Masks_out_train_val_test_split/val"
COCO_MODEL_PATH = "/Jun_anonymized_dir/ToyDataset/mask_rcnn_balloon.h5"

if not os.path.exists(model_out_dir):
    os.makedirs(model_out_dir)

# Directory to save logs and trained model
MODEL_DIR = os.path.join(model_out_dir, "Mask_RCNN_logs")

#Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

label_id_csv = "./label_color_table.csv"
mask_data_lcm = label_color_CSVManager(label_id_csv)
mask_colors = mask_data_lcm.get_color_list()
mask_ids = mask_data_lcm.get_label_id_list()
case_ids = [""]
id_text_list = ["Tumor", "Stroma"]

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
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + len(id_text_list)  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128  # 512
    IMAGE_MAX_DIM = 128  # 512

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
    IMAGES_PER_GPU = 8
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


config = TumorStromaConfig()
config.display()


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

class TumorStromaDataset(utils.Dataset):
    def load_TumorStroma(self, dataset_root_dir, case_ids, color_list, id_text_list, id_list):
        self.case_ids = case_ids
        self.dataset_root_dir = dataset_root_dir
        self.color_list = color_list
        self.id_text_list = id_text_list
        self.id_list = id_list

        for idx, txt in enumerate(id_text_list):
            self.add_class("TumorStroma", id_list[idx], txt)

        for case_id in case_ids:
            dataset_dir = os.path.join(dataset_root_dir, case_id)
            image_fns = os.listdir(dataset_dir)
            for idx, img_fn in enumerate(image_fns):
                self.add_image(
                    "TumorStroma",
                    image_id=idx,
                    path=os.path.join(dataset_dir, img_fn))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        case_id = os.path.split(os.path.dirname(info['path']))[1]
        fn = os.path.split(info['path'])[1].replace(".jpg", ".png")
        # data_root_dir = os.path.dirname(os.path.dirname(info['path']))
        mask_img_fn = os.path.join(self.dataset_root_dir, case_id+"_mask", fn)
        mask_arr = skimage.io.imread(mask_img_fn)[:,:,0:3]
        mask = []
        mask_ids = []
        for idx, color in enumerate(self.color_list):
            m_cells = np.all(mask_arr == np.zeros(mask_arr.shape, dtype=np.uint8) + np.array(color), axis=2)
            blobs_labels, num = measure.label(m_cells, background=0, return_num=True)
            for i in range(1, num + 1):
                region = np.array(blobs_labels == i, dtype=np.bool) #* 255
                mask.append(region)
                mask_ids.append(self.id_list[idx])
        mask = np.stack(mask, axis=-1)
        return np.array(mask, dtype=np.bool), np.array(mask_ids, dtype=np.uint32)

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "TumorStroma":
            return info["id"]
        else:
            super(self.__class__).image_reference(self)

# Training dataset
dataset_train = TumorStromaDataset()
dataset_train.load_TumorStroma(train_dataset_root_dir, case_ids, mask_colors, id_text_list, mask_ids)
dataset_train.prepare()

# Training dataset
dataset_val = TumorStromaDataset()
dataset_val.load_TumorStroma(val_dataset_root_dir, case_ids, mask_colors, id_text_list, mask_ids)
dataset_val.prepare()

# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)


augmentation = iaa.SomeOf((0, 2), [
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.OneOf([iaa.Affine(rotate=90),
               iaa.Affine(rotate=180),
               iaa.Affine(rotate=270)]),
    iaa.Multiply((0.8, 1.5)),
    iaa.GaussianBlur(sigma=(0.0, 5.0))
])
# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val,
            augmentation=augmentation,
            learning_rate=config.LEARNING_RATE,
            epochs=40,
            layers='heads')
# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val,
            augmentation=augmentation,
            learning_rate=config.LEARNING_RATE,
            epochs=80,
            layers="all")

# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_QBRC.h5")
# model.keras_model.save_weights(model_path)
inference_config = TumorStromaInferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(model_out_dir, ".h5 file name here")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
    modellib.load_image_gt(dataset_val, inference_config,
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                            dataset_train.class_names, figsize=(8, 8))

results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            dataset_val.class_names, r['scores'], ax=get_ax())

# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps = \
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)

print("mAP: ", np.mean(APs))
