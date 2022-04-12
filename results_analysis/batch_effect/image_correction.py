import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2


gamma = 1/2.2

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table), table

thumbnail_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/thumbnails"
out_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/gamma_correlation"
case_list = os.listdir(thumbnail_dir)

for cl in case_list:
    case_thumb_dir = os.path.join(thumbnail_dir, cl)
    thumb_list = os.listdir(case_thumb_dir)
    for tl in thumb_list:
        if "thumb" in tl and "7111256" in tl:
            thumb_fn = os.path.join(thumbnail_dir, cl, tl)
            img = cv2.imread(thumb_fn)
            new_img, table = adjust_gamma(img, gamma)
            save_to = os.path.join(out_dir, cl, tl.replace(".png", "_gamma_corr.png"))
            if not os.path.exists(os.path.join(out_dir, cl)):
                os.makedirs(os.path.join(out_dir, cl))
            Image.fromarray(new_img).save(save_to)



