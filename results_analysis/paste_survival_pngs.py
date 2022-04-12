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
file_list = sorted(glob.glob(os.path.join(input_dir, "*")))


f_cnt = int(len(file_list)/3)

gif_all_img_list = []
for idx in range(f_cnt):
    img = np.zeros([350*3, 600, 3]).astype(np.uint8)
    batch_1_fn = file_list[idx*3]
    batch_2_fn = file_list[idx * 3 + 1]
    batch_3_fn = file_list[idx * 3 + 2]
    assert batch_1_fn[0: -len("_batch1_survivial.png")] == batch_2_fn[0: -len("_batch1_survivial.png")] == batch_3_fn[0: -len("_batch1_survivial.png")], "Different images"
    img_batch1 = Image.open(os.path.join(input_dir, batch_1_fn)).convert('RGB')
    img_batch2 = Image.open(os.path.join(input_dir, batch_2_fn)).convert('RGB')
    img_batch3 = Image.open(os.path.join(input_dir, batch_3_fn)).convert('RGB')
    draw = ImageDraw.Draw(img_batch1)
    draw.text((10, 350 - 30), "Batch 1", fill=(255, 0, 0))
    draw = ImageDraw.Draw(img_batch2)
    draw.text((10, 350 - 30,), "Batch 2", fill=(255, 0, 0))
    draw = ImageDraw.Draw(img_batch3)
    draw.text((10, 350 - 30), "Batch 3", fill=(255, 0, 0))
    img[0: 350, :, :] = np.array(img_batch1)
    img[350: 350*2, :, :] = np.array(img_batch2)
    img[350*2: 350*3, :, :] = np.array(img_batch3)

    gif_all_img_list.append(np.array(img))

clip = ImageSequenceClip(gif_all_img_list, fps=1)

save_to = os.path.join(out_dir, 'feature_survival_asso.mp4')
clip.write_videofile(save_to)

im0 = Image.fromarray(gif_all_img_list[0])
im_list = []
for i in gif_all_img_list[1:]:
    im_list.append(Image.fromarray(i))
save_to = os.path.join(out_dir, 'feature_survival_asso.pdf')
im0.save(save_to, "PDF", dpi=(300.0, 300.0), save_all=True, append_images=im_list)

























