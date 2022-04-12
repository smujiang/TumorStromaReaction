from PIL import Image
import os
import numpy as np
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk
import matplotlib.pyplot as plt

thumbnail_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/thumbnails"

fn_seg_ext = "_segmentation.png"
fn_tsr_fib_ext = "_Fibrosis_TSR-score.png"
fn_tsr_cel_ext = "_Cellularity_TSR-score.png"
fn_tsr_ori_ext = "_Orientation_TSR-score.png"

seg_color_map = [[0, 255, 255], [255, 255, 0]]  # tumor & stroma
color_map = [[[100, 0, 0], [180, 0, 0], [255, 0, 0]],  # Fibrosis: 0, 1, 2
             [[0, 100, 0], [0, 180, 0], [0, 255, 0]],
             [[0, 0, 100], [0, 0, 180], [0, 0, 255]]]
Class_label_list = ["Fibrosis", "Cellularity", "Orientation"]

SBOT_fib_tsr_scores = []
SBOT_cel_tsr_scores = []
SBOT_ori_tsr_scores = []
HGSOC_fib_tsr_scores = []
HGSOC_cel_tsr_scores = []
HGSOC_ori_tsr_scores = []

case_id = "OCMC-015"


seg_fn = os.path.join(thumbnail_dir, case_id, case_id+fn_seg_ext)
tumor_stroma_img = Image.open(seg_fn, 'r')
tumor_stroma_img_arr = np.array(tumor_stroma_img)[:,:,0:3]

'''
# get tumor core
'''
k = tumor_stroma_img_arr == seg_color_map[0]
tumor = np.all(k, axis=2)*1
plt.imshow(tumor, cmap="gray")
plt.show()

selem = disk(3)
dilated = dilation(tumor, selem)
tumor_core = erosion(dilated, selem)
plt.imshow(tumor_core, cmap="gray")
plt.show()



'''
#get stroma core
'''
k = tumor_stroma_img_arr == seg_color_map[1]
stroma = np.all(k, axis=2)*1
plt.imshow(stroma, cmap="gray")
plt.show()

selem = disk(3)
dilated = dilation(stroma, selem)
stroma_core = erosion(dilated, selem)
plt.imshow(stroma_core, cmap="gray")
plt.show()

# print(tumor.shape)

'''
# get entangling area
'''
selem = disk(8)
dilated_tumor_core = dilation(tumor_core, selem)
plt.imshow(dilated_tumor_core, cmap="gray")
plt.show()
eroded_tumor_core = erosion(tumor_core, selem)
plt.imshow(eroded_tumor_core, cmap="gray")
plt.show()
entangle_area = np.logical_xor(dilated_tumor_core, eroded_tumor_core)
plt.imshow(entangle_area, cmap="gray")
plt.show()

entangle_area = np.logical_and(entangle_area, tumor_core, stroma_core)
plt.imshow(entangle_area, cmap="gray")
plt.show()

# print(entangle_area.shape)
# h, w = entangle_area.shape

fn_tsr_fib = os.path.join(thumbnail_dir, case_id, case_id+fn_tsr_fib_ext)
tsr_fib = np.array(Image.open(fn_tsr_fib, 'r'))[:, :, 0:3]
fib_all_scores_encoded = tsr_fib[entangle_area == 1]

# print(fib_all_scores_encoded)


def decode_color_to_score(color, color_list, tolerance=10):
    for idx, i in enumerate(color_list):
        if i[0] - tolerance <color[0] < i[0] + tolerance \
                and i[1] - tolerance <color[1] < i[1] + tolerance \
                and i[2] - tolerance <color[2] < i[2] + tolerance:
            return idx
    return -1


fib_score_list = []
for i in fib_all_scores_encoded:
    score = decode_color_to_score(i, color_map[0])
    if score is not -1:
        fib_score_list.append(score)

fig = plt.figure(1)
plt.hist(fib_score_list, bins=3, histtype='bar', rwidth=0.8, color='white', edgecolor='red')
plt.xticks(range(3))
plt.show()
