from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
import os
import numpy as np
from PIL import Image

model_out_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/TumorStroma_MaskRCNN/model"
id_text_list = ["Tumor", "Stroma"]
color_list = [[0, 255, 255], [255, 255, 0]]
MODEL_DIR = os.path.join(model_out_dir, "Mask_RCNN_logs")

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
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

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
    BATCH_SIZE = 8

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

data_root_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/TumorStroma_MaskRCNN/testing_cases"
data_out_root_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/TumorStroma_MaskRCNN/testing_cases_out"
testing_cases = [""]

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

BATCH_SIZE = 8
for case in testing_cases:
    print("Processing case: %s" % case)
    img_fns = os.listdir(os.path.join(data_root_dir, case))
    # for i in img_fns:
    for ifn_s in batch(img_fns, BATCH_SIZE):
        imgs = []
        for i in ifn_s:
            img = np.array(Image.open(os.path.join(data_root_dir, case, i)))
            imgs.append(img)
        results = model.detect(imgs, verbose=0)
        for res_idx, r in enumerate(results):
            mask_img = np.ones([512, 512, 3], dtype=np.uint8) * 255
            for class_idx, class_id in enumerate(r['class_ids']):
                class_txt = id_text_list[class_id-1]
                mask_img[r['masks'][:, :, class_idx]] = color_list[class_id-1]
                # mask_img = r['masks'][:, :, class_idx].astype(np.uint8) * 255
                # print(mask_img.shape)
            img_mask_side_by_side = np.ones([512, 1024, 3], dtype=np.uint8) * 255
            img_mask_side_by_side[:, 0:512, :] = imgs[res_idx]
            img_mask_side_by_side[:, 512:, :] = mask_img
            save_to = os.path.join(data_out_root_dir, case, os.path.split(ifn_s[res_idx])[1].replace(".jpg", ".png"))
            if not os.path.exists(os.path.join(data_out_root_dir, case)):
                os.makedirs(os.path.join(data_out_root_dir, case))
            # Image.fromarray(mask_img).save(save_to)
            Image.fromarray(img_mask_side_by_side).save(save_to)












