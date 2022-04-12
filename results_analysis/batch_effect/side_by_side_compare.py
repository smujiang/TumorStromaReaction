import os
from PIL import Image
import numpy as np

thumbnail_dir = "\\\\anonymized_dir\\thumbnails"
output_dir = "\\\\anonymized_dir\\result_analysis\\batch_effect"

###### group1:
# tumor_b_hist_7, stroma_g_hist_10, tumor_g_hist_6
# tumor_r_hist_11, stroma_r_hist_29, stroma_r_hist_27
# tumor_b_hist_12, tumor_b_hist_31, tumor_g_hist_28
# tumor_g_hist_26, tumor_b_hist_30, stroma_r_hist_28
group1_case_ids = [""]

###### group2:
group2_case_ids = [""]

for idx, c in enumerate(group2_case_ids):
    img_arr = np.zeros([512, 1024, 3]).astype(np.uint8)
    # print(os.path.exists(os.path.join(thumbnail_dir, c)))
    img_fn = os.path.join(thumbnail_dir, c, c+"_thumbnail.png")
    img1 = Image.open(img_fn).resize((512, 512))

    img_fn = os.path.join(thumbnail_dir, group1_case_ids[idx], group1_case_ids[idx] + "_thumbnail.png")
    # print(os.path.exists(os.path.join(thumbnail_dir, group1_case_ids[idx])))
    img2 = Image.open(img_fn).resize((512, 512))
    img_arr[:, 0:512, :] = np.array(img1)
    img_arr[:, 512:1024, :] = np.array(img2)
    sv_Img = Image.fromarray(img_arr)

    save_to = os.path.join(output_dir, "side_by_side_" + str(idx) + ".png")
    sv_Img.save(save_to)

















