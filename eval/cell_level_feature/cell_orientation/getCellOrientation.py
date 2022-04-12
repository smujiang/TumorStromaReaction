import os
import cv2
from sklearn.decomposition import PCA
import numpy as np
from PIL import Image
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import math
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
import scipy.ndimage as ndimage


data_dir = "\\\\anonymized_dir\\result_analysis\\ROI_Img_mask"
out_dir = "\\\\anonymized_dir\\result_analysis\\ROI_Img_mask_cell_orientation"

case_ids = os.listdir(data_dir)


color_list = [[0, 255, 255], [255, 255, 0]]

def detect_orientation(mask_img_arr, color=None):
    cleared = np.all(mask_img_arr == color, axis=2)
    cell_centroids = []
    ang_list = []
    length_list = []
    vec_list = []

    pca = PCA(n_components=2)

    label_image = label(cleared)
    for region in regionprops(label_image):
        x = region.coords
        if x.shape[0] >= 10 and x.shape[0] <= 1000:
            pca = pca.fit(x)
            if pca.explained_variance_[0] / pca.explained_variance_[1] > 1.5:
                cell_centroids.append(region.centroid)
                length = pca.explained_variance_[0]
                vector = pca.components_[0]
                length_list.append(length)
                vec_list.append([vector[0], vector[1]])
                ang = math.atan(vector[0] / vector[1])
                ang_list.append(ang)

    return ang_list, cell_centroids, length_list, vec_list


for c_id in case_ids:
    fd = os.path.join(data_dir, c_id)
    if os.path.isdir(fd):
        fn_list = sorted(os.listdir(fd))
        if len(fn_list) % 2 != 0:
            print("Missing mask or image")
            raise Exception("Missing mask or image")
        else:
            case_out_put_dir = os.path.join(out_dir, c_id)
            if not os.path.exists(case_out_put_dir):
                os.makedirs(case_out_put_dir)

            for i in range(int(len(fn_list)/2)):
                fn_1 = fn_list[2 * i]
                fn_2 = fn_list[2 * i]
                if "mask" in fn_1:
                    mask_fn = os.path.join(fd, fn_1)
                else:
                    mask_fn = os.path.join(fd, fn_2)

                img_fn = os.path.join(fd, mask_fn.replace("-mask.png", ".jpg"))
                img_fn_p = os.path.split(img_fn)[1]
                mask_img_arr = np.array(Image.open(mask_fn, 'r'))
                ang_list, cell_centroids, length_list, vec_list = detect_orientation(mask_img_arr, color_list[1])

                img_arr = np.array(Image.open(img_fn, 'r'))
                for idx, c_c in enumerate(cell_centroids):
                    cc = (int(c_c[1]), int(c_c[0]))
                    vv = (int(vec_list[idx][1]*30), int(vec_list[idx][0]*30))
                    color_arrow = [0, 0, 255]
                    img_arr = cv2.arrowedLine(img_arr, cc, (cc[0]+vv[0], cc[1]+vv[1]), color_arrow, 2)
                plt.imshow(img_arr)
                plt.show()
                save_to = os.path.join(case_out_put_dir, img_fn_p.replace(".jpg", "_arrow.jpg"))
                Image.fromarray(img_arr).save(save_to, dpi=(200, 200))
                print(save_to)





#
# B_img_fn = "H:\\Jun_anonymized_dir\\YYY_replication\\OC3-mask.JPG"
# C_img_fn = "H:\\Jun_anonymized_dir\\YYY_replication\\OC3.jpg"
#
# # B_img_fn = "H:\\Jun_anonymized_dir\\YYY_replication\\extra_tesingt-mask.jpg"
# # C_img_fn = "H:\\Jun_anonymized_dir\\YYY_replication\\extra_tesingt.png"
#
#
# Img = Image.open(B_img_fn, 'r')
# cImg = Image.open(C_img_fn, 'r')
# image = np.array(Img)
# # plt.imshow(image, cmap='gray')
#
# plt.figure(1, [10,8])
# c_image = np.array(cImg)
# plt.imshow(c_image)
#
# thresh = threshold_otsu(image)
# # bw = closing(image > thresh, square(3))
# #
# # remove artifacts connected to image border
# # cleared = clear_border(bw)
#
# cleared = (image < 1)
# # plt.imshow(cleared, cmap='gray')
# # plt.show()
# # label image regions
# label_image = label(cleared)
#
# ang_list = []
# cell_points = []
# for region in regionprops(label_image):
#     x = region.coords
#     cell_points.append(region.centroid)
#     if x.shape[0] >= 10 and x.shape[0] <= 100:
#         centroid = region.centroid
#         # cell_points.append(centroid)
#         pca = pca.fit(x)
#         # plt.scatter(x[:, 0], x[:, 1], alpha=0.2)
#         if pca.explained_variance_[0]/pca.explained_variance_[1] > 3:
#             length = pca.explained_variance_[0]
#             vector = pca.components_[0]
#             v = vector * 10 * np.sqrt(length)
#             xy = [pca.mean_[1], pca.mean_[0]]
#             d = pca.mean_ + v
#             xy_1 = [d[1], d[0]]
#             draw_vector(xy, xy_1)
#             ang = math.atan(vector[0]/vector[1])
#             ang_list.append(ang)
#
# plt.show()
#
# print(np.std(np.array(ang_list)))
#
# plt.figure(2)
# num_bins = 30
# cos_list = []
# for ang in ang_list:
#     cos_list.append(math.cos(ang))
# n, bins, patches = plt.hist(cos_list, num_bins, facecolor='blue', alpha=0.5)
# plt.show()
#
# cell_points = np.array(cell_points)
# x = np.roll(cell_points, 1, axis=1)
# vor = Voronoi(x)
# fig = plt.figure(3, figsize=(20, 20))
# ax = fig.add_subplot(111)
# # ax.imshow(ndimage.rotate(c_image, 90))
# ax.imshow(c_image)
# voronoi_plot_2d(vor, point_size=1, ax=ax)
# plt.show()
#
# tri = Delaunay(x)
# fig = plt.figure(4, figsize=(20, 20))
# ax = fig.add_subplot(111)
# ax.imshow(c_image)
# voronoi_plot_2d(vor, point_size=1, ax=ax)
# plt.triplot(x[:, 0], x[:, 1], tri.simplices, color='yellow')
# plt.plot(x[:, 0], x[:, 1], 'o')
# plt.show()
#
# print(np.std(np.array(cos_list)))
#         # principalComponents = pca.fit_transform(x)
# print(label_image)





