import multiprocessing
import os
import glob
from PIL import Image
import numpy as np
import openslide
import matplotlib.pyplot as plt

# case_list = ["OCMC-015", "OCMC-016"]

patch_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/patches"
seg_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/segmentation"
# score_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/reaction_prediction"
score_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/reaction_prediction_NC"
# wsi_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/WSIs"
wsi_dir = "/Jun_anonymized_dir/OvaryCancer/WSIs"
output_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/thumbnails"

case_list = sorted(os.listdir(score_dir))

img_rows = 256
img_cols = 256
img_channel = 3
img_rescale_rate = 2

thumbnail_downsample = 128
patch_thumb_size = int(img_rows*img_rescale_rate/thumbnail_downsample)
all_class_list = ["Fibrosis", "Cellularity", "Orientation"]

def get_coord_from_fn(fn):
    img_fn = os.path.split(fn)[1]
    ele = img_fn.split("_")
    loc_x = int(ele[-2])
    loc_y = int(ele[-1][0:-4])
    return loc_x, loc_y

def get_list_from_folder_complimentary(dir_whole, dir_sub):
    case_ids_1 = os.listdir(dir_whole)
    case_ids_2 = os.listdir(dir_sub)
    ids = [f for f in case_ids_1 if f not in case_ids_2]
    return ids


def create_thumbnails(case):
    print("Processing %s" % case)
    wsi_fn = os.path.join(wsi_dir, case + ".svs")
    wsi_obj = openslide.open_slide(wsi_fn)
    thumb_size = np.array(wsi_obj.dimensions) / thumbnail_downsample
    thumb_img = wsi_obj.get_thumbnail(thumb_size)

    save_to_dir = os.path.join(output_dir, case)
    if not os.path.exists(save_to_dir):
        os.makedirs(save_to_dir)
    # save WSI thumbnail image

    save_to = os.path.join(save_to_dir, case + "_thumbnail.png")
    if not os.path.exists(save_to):
        print("     Saving %s WSI thumbnail." % case)
        thumb_img.save(save_to)
    else:
        print("     WSI %s thumbnail already exist." % case)

    # save segmentation thumbnail image
    save_to = os.path.join(save_to_dir, case + "_segmentation.png")
    if not os.path.exists(save_to):
        print("     Saving %s segmentation thumbnail." % case)
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
        print("     %s Segmentation thumbnail already exist." % case)


    # save stroma reaction score thumbnail image
    for class_txt in all_class_list:
        save_to = os.path.join(save_to_dir, case + "_" + class_txt + "_TSR-score.png")
        if not os.path.exists(save_to):
            print("     Saving %s %s TSR score thumbnail." % (case, class_txt))
            score_thumb_img_arr = np.ones((thumb_img.height, thumb_img.width, 4), dtype=np.uint8) + 255
            score_img_fns = glob.glob(os.path.join(score_dir, case, class_txt, "*.png"))
            for score_img_fn in score_img_fns:
                loc_x, loc_y = get_coord_from_fn(score_img_fn)
                x = int(loc_x/thumbnail_downsample)
                y = int(loc_y/thumbnail_downsample)
                score_thumb = Image.open(score_img_fn, 'r')
                xx = score_thumb.resize((patch_thumb_size, patch_thumb_size))
                # print(y+seg_patch_thumb_size, x+seg_patch_thumb_size)
                # print(seg_patch_thumb_size)
                score_thumb_img_arr[y:y+patch_thumb_size, x:x+patch_thumb_size, :] = np.array(xx)
            Image.fromarray(score_thumb_img_arr).save(save_to)
        else:
            print("     %s %s TSR score thumbnail already exist." % (case, class_txt))


a_pool = multiprocessing.Pool()
a_pool.map(create_thumbnails, case_list)

















