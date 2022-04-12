import os
import random
import shutil
from PIL import Image

# split dataset into training and validation subset

train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

data_root_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/TumorStroma_MaskRCNN/ROI_Masks_out"
data_out_root_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/TumorStroma_MaskRCNN/ROI_Masks_out_train_val_test_split"
train_data_dir = os.path.join(data_out_root_dir, 'train')
val_data_dir = os.path.join(data_out_root_dir, 'val')
test_data_dir = os.path.join(data_out_root_dir, 'test')

case_id_list = os.listdir(data_root_dir)

JPG2PNG = True

for case_id in case_id_list:
    all_img_list = os.listdir(os.path.join(data_root_dir, case_id))
    img_list = []
    for a in all_img_list:
        if "mask" not in a:
            img_list.append(a)
    img_cnt = len(img_list)
    train_cnt = int(img_cnt * train_ratio)
    val_cnt = int(img_cnt * val_ratio)
    random.shuffle(img_list)
    train_imgs = img_list[0:train_cnt]
    val_imgs = img_list[train_cnt:train_cnt+val_cnt]
    test_imgs = img_list[train_cnt+val_cnt:]
    if not os.path.exists(os.path.join(train_data_dir, case_id)):
        os.makedirs(os.path.join(train_data_dir, case_id))
        os.makedirs(os.path.join(train_data_dir, case_id + "_mask"))
        os.makedirs(os.path.join(val_data_dir, case_id))
        os.makedirs(os.path.join(val_data_dir, case_id + "_mask"))
        os.makedirs(os.path.join(test_data_dir, case_id))
        os.makedirs(os.path.join(test_data_dir, case_id + "_mask"))
    for img in train_imgs:
        img_fn = os.path.join(data_root_dir, case_id, img)
        # img_mask_fn = os.path.join(data_root_dir, case_id+"_mask", img.replace(".jpg", ".png"))
        img_mask_fn = os.path.join(data_root_dir, case_id, img.replace(".jpg", "-mask.png"))
        img_fn_to = os.path.join(train_data_dir, case_id, img)
        img_mask_fn_to = os.path.join(train_data_dir, case_id + "_mask", img.replace(".jpg", ".png"))
        # shutil.copyfile(img_fn, img_fn_to)
        # shutil.copyfile(img_mask_fn, img_mask_fn_to)
        # shutil.move(img_fn, img_fn_to)
        if JPG2PNG:
            I = Image.open(img_fn).convert("RGBA")
            dst_img_fn = img_fn_to.replace(".jpg", ".png")
            I.save(dst_img_fn)
        else:
            shutil.move(img_fn, img_fn_to)
        shutil.move(img_mask_fn, img_mask_fn_to)
    for img in val_imgs:
        img_fn = os.path.join(data_root_dir, case_id, img)
        # img_mask_fn = os.path.join(data_root_dir, case_id+"_mask", img.replace(".jpg", ".png"))
        img_mask_fn = os.path.join(data_root_dir, case_id, img.replace(".jpg", "-mask.png"))
        img_fn_to = os.path.join(val_data_dir, case_id, img)
        img_mask_fn_to = os.path.join(val_data_dir, case_id + "_mask", img.replace(".jpg", ".png"))
        # shutil.copyfile(img_fn, img_fn_to)
        # shutil.copyfile(img_mask_fn, img_mask_fn_to)
        # shutil.move(img_fn, img_fn_to)
        if JPG2PNG:
            I = Image.open(img_fn).convert("RGBA")
            dst_img_fn = img_fn_to.replace(".jpg", ".png")
            I.save(dst_img_fn)
        else:
            shutil.move(img_fn, img_fn_to)
        shutil.move(img_mask_fn, img_mask_fn_to)
    for img in test_imgs:
        img_fn = os.path.join(data_root_dir, case_id, img)
        # img_mask_fn = os.path.join(data_root_dir, case_id+"_mask", img.replace(".jpg", ".png"))
        img_mask_fn = os.path.join(data_root_dir, case_id, img.replace(".jpg", "-mask.png"))
        img_fn_to = os.path.join(test_data_dir, case_id, img)
        img_mask_fn_to = os.path.join(test_data_dir, case_id + "_mask", img.replace(".jpg", ".png"))
        # shutil.copyfile(img_fn, img_fn_to)
        # shutil.copyfile(img_mask_fn, img_mask_fn_to)
        # shutil.move(img_fn, img_fn_to)
        if JPG2PNG:
            I = Image.open(img_fn).convert("RGBA")
            dst_img_fn = img_fn_to.replace(".jpg", ".png")
            I.save(dst_img_fn)
        else:
            shutil.move(img_fn, img_fn_to)
        shutil.move(img_mask_fn, img_mask_fn_to)
    # for img in val_imgs:
    #     img_fn = os.path.join(data_root_dir, case_id, img)
    #     # img_mask_fn = os.path.join(data_root_dir, case_id+"_mask", img.replace(".jpg", ".png"))
    #     img_mask_fn = os.path.join(data_root_dir, case_id, img.replace(".jpg", "-mask.png"))
    #     img_fn_to = os.path.join(val_data_dir, case_id, img)
    #     img_mask_fn_to = os.path.join(val_data_dir, case_id + "_mask", img.replace(".jpg", ".png"))
    #     # shutil.copyfile(img_fn, img_fn_to)
    #     # shutil.copyfile(img_mask_fn, img_mask_fn_to)
    #     shutil.move(img_fn, img_fn_to)
    #     shutil.move(img_mask_fn, img_mask_fn_to)


