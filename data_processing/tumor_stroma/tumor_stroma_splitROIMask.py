import os
from PIL import Image, ImageDraw
import numpy as np

data_root = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/Annotation_tumor_stroma"

case_list = [""]

data_out_root = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/TumorStroma_MaskRCNN/ROI_Masks_out"


PatchSize = 512
Stride = 256

SaveValidate = False   # choose to save the validation of patch extraction

# "OCMC-001_(1.00,35460,33137,1024,772).jpg"
# def get_coordinates_from_fn(filename):
#     l = filename.index("(")
#     r = filename.index(")")
#     coord = filename[l:r+1]
#     rescale, x, y, w, h = eval(coord)
#     return x, y
#
# def get_case_id_from_fn(filename):
#     l = filename.index("(")
#     return filename[0:l-1]

# "OCMC-001_33137_1024.jpg"
# def get_coordinates_from_fn(filename):
#     l = filename.index("_")
#     r = filename.rindex("_")
#     x = filename[l+1:r]
#     e = filename.index(".")
#     y = filename[r+1:e]
#     return int(x), int(y)
#
# def get_case_id_from_fn(filename):
#     l = filename.index("_")
#     return filename[0:l]

# "Wo-1-A5_RIO1338_HE_10637_23947_5120_3072.jpg"
def get_coordinates_from_fn(filename):
    dash_pos = [pos for pos, char in enumerate(filename) if char == "_"]
    x = filename[dash_pos[-4]+1:dash_pos[-3]]
    y = filename[dash_pos[-3]+1:dash_pos[-2]]
    return int(x), int(y)

def get_case_id_from_fn(filename):
    dash_pos = [pos for pos, char in enumerate(filename) if char == "_"]
    return filename[0:dash_pos[-4]]

def get_optimal_xy(xy, patch_size):
    for i in range(patch_size):
        if ((xy-i) % patch_size) == 0:
            return xy-i

ROI_list = []
Mask_list = []
for case in case_list:
    file_list = os.listdir(os.path.join(data_root, case))
    for f in file_list:
        if "mask" not in f:
            ROI_list.append(os.path.join(data_root, case, f))
            Mask_list.append(os.path.join(data_root, case, f.replace(".jpg", "-mask.png")))


for idx, roi in enumerate(ROI_list):
    ROI = Image.open(roi)
    Mask = Image.open(Mask_list[idx])
    head, tail = os.path.split(roi)
    # rescale, x, y, w, h = get_coordinates_from_fn(tail)
    x, y = get_coordinates_from_fn(tail)

    case_id = get_case_id_from_fn(tail)

    # extract_validate_fn = os.path.join(data_out_root, case_id, case_id + "_" + str((rescale, x, y, w, h)) + "-validate.jpg")
    extract_validate_fn = os.path.join(data_out_root, case_id, case_id + "_" + str(x)+"_"+str(y) + "-validate.jpg")
    draw = ImageDraw.Draw(ROI)

    ROI_array = np.array(ROI)
    Mask_array = np.array(Mask)

    opt_w = get_optimal_xy(ROI.width, PatchSize)
    opt_h = get_optimal_xy(ROI.height, PatchSize)
    ex_grid_x = range(0, opt_w, Stride)
    ex_grid_y = range(0, opt_h, Stride)
    for xx in ex_grid_x:
        for yy in ex_grid_y:
            if xx+PatchSize <= ROI.width and yy+PatchSize <= ROI.height:
                ROI_patch = ROI_array[yy:yy + PatchSize, xx:xx+PatchSize, :]
                Mask_patch = Mask_array[yy:yy + PatchSize, xx:xx + PatchSize, :]
                # info_patch = (rescale, x+xx, y+yy, PatchSize, PatchSize)
                if not os.path.exists(os.path.join(data_out_root, case_id)):
                    os.makedirs(os.path.join(data_out_root, case_id))
                # ROI_patch_fn = os.path.join(data_out_root, case_id, case_id+"_"+str(info_patch)+".jpg")
                # Mask_patch_fn = os.path.join(data_out_root, case_id, case_id + "_" + str(info_patch) + "-mask.png")
                ROI_patch_fn = os.path.join(data_out_root, case_id, case_id + "_" + str(x+xx)+"_"+str(y+yy) + ".jpg")
                Mask_patch_fn = os.path.join(data_out_root, case_id, case_id + "_" + str(x+xx)+"_"+str(y+yy) + "-mask.png")
                # must specify RGB, and save as PNG, so that it won't smooth the edge of the mask
                Image.fromarray(ROI_patch, "RGB").save(ROI_patch_fn)
                # must specify RGB, and save as PNG, so that it won't smooth the edge of the mask
                Image.fromarray(Mask_patch, "RGB").save(Mask_patch_fn)
                if ROI_patch.shape[1] < 512 or ROI_patch.shape[0] < 512:
                    print("bug")
                if SaveValidate:
                    xy = [xx, yy, xx+PatchSize, yy+PatchSize]
                    draw.rectangle(xy, outline='yellow')
    # ROI.save(extract_validate_fn)
