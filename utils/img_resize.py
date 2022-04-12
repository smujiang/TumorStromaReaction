import os
from PIL import Image

img_dir = "/anonymized_dir/Dataset/OvaryData/Patches/OCMC-004"
img_out_dir = "/anonymized_dir/Dataset/OvaryData/Patches/OCMC-small-004"

if not os.path.exists(img_out_dir):
    os.makedirs(img_out_dir)

img_list = os.listdir(img_dir)
for img_fn in img_list:
    if ".jpg" in img_fn:
        img = Image.open(os.path.join(img_dir, img_fn))
        im_resized = img.resize((256, 256))
        im_resized.save(os.path.join(img_out_dir, img_fn))


