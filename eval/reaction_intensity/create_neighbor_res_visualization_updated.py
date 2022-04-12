import os
import glob
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import openslide
import random
import matplotlib.pyplot as plt

# case_list = ["OCMC-015", "OCMC-016"]

patch_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/patches"
seg_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/segmentation"
score_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/reaction_prediction_NC"
# wsi_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/WSIs"
wsi_dir = "/Jun_anonymized_dir/OvaryCancer/WSIs"
output_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/neighbor_visualization_plus"

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

fnt = ImageFont.truetype("/usr/share/fonts/gnu-free/FreeMono.ttf", 15)
fnt_bold = ImageFont.truetype("/usr/share/fonts/gnu-free/FreeMonoBold.ttf", 15)

color_map = [[[100, 0, 0], [180, 0, 0], [255, 0, 0]],
             [[0, 100, 0], [0, 180, 0], [0, 255, 0]],
             [[0, 0, 100], [0, 0, 180], [0, 0, 255]]]

def get_grade(mask_arr, color_map, metric_idx):
    colors = color_map[metric_idx]
    cnts = []
    ret_str = ["low", "medium", "high"]
    for c in colors:
        k = mask_arr[:, :, 0:3] == c
        cnt = np.count_nonzero(np.all(k, axis=2))
        cnts.append(cnt)
    v = max(cnts)
    idx = cnts.index(v)
    return ret_str[idx]


for case in case_list:
    print("Processing %s" % case)
    wsi_fn = os.path.join(wsi_dir, case + ".svs")
    wsi_obj = openslide.open_slide(wsi_fn)

    save_to_dir = os.path.join(output_dir, case)
    if not os.path.exists(save_to_dir):
        os.makedirs(save_to_dir)

    # load segmentation images
    seg_img_fns_all = glob.glob(os.path.join(seg_dir, case, "*.png"))
    seg_img_fns = random.sample(seg_img_fns_all, 100)
    for seg_img_fn in seg_img_fns:
        Img = Image.new("RGBA", (256*4+100, 256*3), 'white')

        loc_x, loc_y = get_coord_from_fn(seg_img_fn)
        orig_x = loc_x - img_cols*img_rescale_rate
        orig_y = loc_y - img_rows*img_rescale_rate
        roi_img = wsi_obj.read_region((orig_x, orig_y), 0, (img_cols*img_rescale_rate*3, img_rows*img_rescale_rate*3))

        xx = roi_img.resize((img_cols*3, img_rows*3))
        xx.putalpha(255)
        # roi_img.putalpha(255)
        Img.paste(xx, (0, 0))
        shape = [(256, 256), (256*2, 256*2)]
        draw = ImageDraw.Draw(Img)
        draw.rectangle(shape, fill=None, outline="green")


        # save_to = os.path.join(save_to_dir, os.path.split(seg_img_fn)[1].replace(".png", "_1.png"))
        # img_patch_fn = os.path.join(patch_dir, case, os.path.split(seg_img_fn)[1].replace(".png", ".jpg"))
        # Image.open(img_patch_fn, 'r').save(save_to)

        resize_pred = (192, 192)
        seg_img = Image.open(seg_img_fn, 'r')
        seg_img.putalpha(255)
        Img.paste(seg_img.resize(resize_pred), (256*3+10, 0))

        # Stroma
        shape = [(256*3+192+10+5, 20), (256*3+192+10+5+20, 35)]
        draw.rectangle(shape, fill=(255, 255, 0), outline=None)
        xy = [256*3+192+10+5+20+3, 20]
        draw.text(xy, "Stroma", font=fnt, fill=(0, 0, 0))

        # Tumor
        shape = [(256*3+192+10+5, 40), (256*3+192+10+5+20, 55)]
        draw.rectangle(shape, fill=(0, 255, 255), outline=None)
        xy = [256*3+192+10+5+20+3, 40]
        draw.text(xy, "Tumor", font=fnt, fill=(0, 0, 0))

        save_ = True
        for idx, class_txt in enumerate(all_class_list):
            score_img_fn = seg_img_fn.replace("segmentation", "reaction_prediction_NC")
            score_img_fn = os.path.join(os.path.split(score_img_fn)[0], class_txt, os.path.split(seg_img_fn)[1])
            paste_loc = (256*3+10, 192*(idx+1))

            if os.path.exists(score_img_fn):
                score_img = Image.open(score_img_fn, 'r')
                Img.paste(score_img.resize(resize_pred), paste_loc)

                # grade
                txt_loc = (256*3+192+10+3, 192*(idx+1)+13)
                draw.text(txt_loc, class_txt + ":", font=fnt_bold, fill=(0, 0, 0))
                txt_loc = (256*3+192+10+3, 192*(idx+1)+43)
                grade = get_grade(np.array(score_img), color_map, idx)
                draw.text(txt_loc, grade, font=fnt, fill=(0, 0, 0))
            else:
                save_ = False
                print(score_img_fn)
                # print("Warning: TSR score prediction not found")
        if save_:
            save_to = os.path.join(save_to_dir, os.path.split(seg_img_fn)[1])
            Img.save(save_to)



















