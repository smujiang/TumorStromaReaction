from PIL import Image, ImageDraw
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip

out_dir = "H:\\Jun_anonymized_dir\\OvaryCancer\\StromaReaction\\result_analysis\\survival_mp4"
input_dir = "H:\\Jun_anonymized_dir\\OvaryCancer\\StromaReaction\\result_analysis\\survival"

## sort by the file creation time
# file_list = list(filter(os.path.isfile, glob.glob(os.path.join(input_dir, "*"))))
# file_list.sort(key=lambda x: os.path.getmtime(x))

# sort by file name
file_list = sorted(glob.glob(os.path.join(input_dir, "*batch1*")))


f_cnt = len(file_list)

im_list = []
for idx in range(f_cnt):
    batch_1_fn = file_list[idx]
    img_batch1 = Image.open(os.path.join(input_dir, batch_1_fn)).convert('RGB')
    im_list.append(img_batch1)

im0 = im_list[0]
save_to = os.path.join(out_dir, 'batch1_feature_survival_asso.pdf')
im0.save(save_to, "PDF", dpi=(300.0, 300.0), save_all=True, append_images=im_list[1:])

























