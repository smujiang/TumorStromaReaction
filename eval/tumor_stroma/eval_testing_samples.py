from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
import os
import numpy as np
from PIL import Image
from sklearn.metrics import jaccard_score, roc_curve, precision_recall_fscore_support, average_precision_score, f1_score

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
    IMAGES_PER_GPU = 1

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

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

data_root_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/TumorStroma_MaskRCNN/ROI_Masks_out_train_val_test_split/test"
data_out_root_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/TumorStroma_MaskRCNN/ROI_Masks_out_train_val_test_split/test_out"
testing_cases = [""]
eval_log_fn = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/TumorStroma_MaskRCNN/ROI_Masks_out_train_val_test_split/test_out/eval_log.csv"

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def color_mask_to_label_mask(mask_array, color_list):
    dim = mask_array.shape
    mask_label_img = np.zeros([dim[0], dim[1], 1], dtype=np.uint8)
    for idx, c in enumerate(color_list):
        x = mask_array == c
        xx = np.all(x, axis=2)
        mask_label_img[xx] = idx
    return mask_label_img

fp = open(eval_log_fn, 'w')
wrt_str = "img_fn,jaccard_similarity,average_precision,DICE(F1)\n"

BATCH_SIZE = 1
scores_all = []
f1_scores = []
ap_scores = []
for case in testing_cases:
    print("Processing case: %s" % case)
    img_fns = os.listdir(os.path.join(data_root_dir, case))
    # for i in img_fns:
    for ifn_s in batch(img_fns, BATCH_SIZE):
        imgs = []
        masks = []
        for i in ifn_s:
            img = np.array(Image.open(os.path.join(data_root_dir, case, i)))
            mask = np.array(Image.open(os.path.join(data_root_dir, case+"_mask", i)))
            imgs.append(img[:, :, 0:3])
            masks.append(mask[:, :, 0:3])
        results = model.detect(imgs, verbose=0)
        for res_idx, r in enumerate(results):
            mask_img = np.ones([512, 512, 3], dtype=np.uint8) * 255
            mask_label_img = np.zeros([512, 512, 1], dtype=np.uint8)
            for class_idx, class_id in enumerate(r['class_ids']):
                class_txt = id_text_list[class_id-1]
                mask_img[r['masks'][:, :, class_idx]] = color_list[class_id-1]
                mask_label_img[r['masks'][:, :, class_idx]] = class_id - 1
                # mask_img = r['masks'][:, :, class_idx].astype(np.uint8) * 255
                # print(mask_img.shape)

            # save original image, ground truth mask and predicted mask
            # img_mask_side_by_side = np.ones([512, 1536, 3], dtype=np.uint8) * 255
            # img_mask_side_by_side[:, 0:512, :] = imgs[res_idx]
            # img_mask_side_by_side[:, 512:1024, :] = masks[res_idx]
            # img_mask_side_by_side[:, 1024:, :] = mask_img
            # save_to = os.path.join(data_out_root_dir, case, os.path.split(ifn_s[res_idx])[1].replace(".jpg", ".png"))
            # if not os.path.exists(os.path.join(data_out_root_dir, case)):
            #     os.makedirs(os.path.join(data_out_root_dir, case))
            # Image.fromarray(img_mask_side_by_side).save(save_to)

            # import matplotlib.pyplot as plt
            # plt.imshow(np.squeeze(mask_label_img))
            # plt.show()

            # calculate IOU
            y_truth = color_mask_to_label_mask(masks[res_idx], color_list)
            score = jaccard_score(y_truth.flatten(), mask_label_img.flatten(), pos_label=1, average="binary")
            # roc_auc_score
            # fpr, tpr, thresholds = roc_curve(y_truth.flatten(), mask_label_img.flatten())
            ap = average_precision_score(y_truth.flatten(), mask_label_img.flatten(), pos_label=1)
            # f1 = f1_score(y_truth.flatten(), mask_label_img.flatten(), average="macro")
            precision, recall, f, s = precision_recall_fscore_support(y_truth.flatten(), mask_label_img.flatten(), pos_label=1, average="binary")
            scores_all.append(score)
            ap_scores.append(ap)
            f1_scores.append(f)
            wrt_str += "%s,%.4f,%s,%.4f\n" % (ifn_s[res_idx], score, str(ap), f)
fp.write(wrt_str)
fp.close()
print("Averaged Jaccard similarity: %.4f" % np.mean(np.array(scores_all)))
print("Averaged average_precision_score scores: %.4f" % np.mean(np.array(ap_scores)))
print("Averaged F1 scores: %.4f" % np.mean(np.array(f1_scores)))
# excel : =AVERAGEIF(B2:B1111,"<>0")
# Averaged Jaccard similarity: 0.8697








