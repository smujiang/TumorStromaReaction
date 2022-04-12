import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

dr = 128 # downsample_rate

case_id = "case_id_HE"

roi_csv_dir = "\\\\anonymized_dir\\result_analysis\\cell_rois"
thumbnail_dir = "\\\\anonymized_dir\\thumbnails"

# output_dir = "\\\\anonymized_dir\\result_analysis\\ROI_Img_mask_cell_orientation"
output_dir = "\\\\anonymized_dir\\result_analysis\\TS_interface_eval"
case_list = [""]
for case_id in case_list:

    rois_fn = os.path.join(roi_csv_dir, case_id+"_roi_box.csv")
    roi_lines = open(rois_fn, 'r').readlines()[1:]
    boxes = []
    for l in roi_lines:
        ele = l.split(",")
        boxes.append([int(ele[0]), int(ele[1]), int(ele[2]), int(ele[3])])

    thumbnail_fn = os.path.join(thumbnail_dir, case_id, case_id + "_thumbnail.png")

    img = Image.open(thumbnail_fn)
    img_draw = ImageDraw.Draw(img)
    for b in boxes:
        img_draw.rectangle([int(b[0]/dr), int(b[1]/dr), int(b[2]/dr), int(b[3]/dr)], outline=(0,0,255))

    save_to = os.path.join(output_dir, case_id + "_interface_ROIs.png")
    img.save(save_to)


print("Done")

