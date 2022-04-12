import glob
import os, shutil
from openslide import OpenSlide
import numpy as np
from pprint import pprint
import pickle
import matplotlib.pyplot as plt
import csv
from PIL import Image, ImageDraw
from scipy.spatial import distance
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram
import seaborn as sns
import matplotlib.cm as cm
from scipy.cluster.hierarchy import ward, fcluster
from matplotlib import ticker
import pandas as pd

sns.set_theme(color_codes=True)

# refer to compare_wsi_info.py, get key "openslide.comment"
wsi_dir = "/Jun_anonymized_dir/OvaryCancer/WSIs"
out_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction_results"
thumbnail_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/thumbnails"

# wsi_dir = "/anonymized_dir/Dataset/OvaryData/StromaReaction/WSIs"
# out_dir = "/anonymized_dir/Dataset/OvaryData/StromaReaction/StromaReaction_results"


exposure_time_based_batches_data_fn = os.path.join(out_dir, "exposure_time_based_batches.pkl")
ICC_pro_based_batches_data_fn = os.path.join(out_dir, "ICC_pro_based_batches.pkl")
gamma_val_based_batches_data_fn = os.path.join(out_dir, "gamma_val_based_batches.pkl")


def get_exposure_time(description):
    start_idx = description.index("Exposure Time =")
    end_idx = description.index("|", start_idx)
    et = description[start_idx + len("Exposure Time ="):end_idx].strip()  # get exposure time
    return et


def get_attri_from_description(description, start_str, end_str):
    start_idx = description.index(start_str)
    end_idx = description.index(end_str, start_idx + 1)
    attri = description[start_idx + len(start_str):end_idx].strip()  # get attribute time
    return attri


# start
if os.path.exists(exposure_time_based_batches_data_fn) \
        and os.path.exists(ICC_pro_based_batches_data_fn) \
        and os.path.exists(gamma_val_based_batches_data_fn):

    file_to_read = open(exposure_time_based_batches_data_fn, "rb")
    exposure_time_based_batches = pickle.load(file_to_read)
    exposure_time_set = exposure_time_based_batches.keys()
    file_to_read.close()
    file_to_read = open(ICC_pro_based_batches_data_fn, "rb")
    ICC_pro_based_batches = pickle.load(file_to_read)
    ICC_pro_set = ICC_pro_based_batches.keys()
    file_to_read.close()
    file_to_read = open(gamma_val_based_batches_data_fn, "rb")
    gamma_val_based_batches = pickle.load(file_to_read)
    gamma_set = gamma_val_based_batches.keys()
    file_to_read.close()

else:
    exposure_time_based_batches = {}
    exposure_time_set = set([])

    ICC_pro_based_batches = {"Null": []}
    ICC_pro_set = set([])

    gamma_val_based_batches = {"Null": []}
    gamma_set = set([])

    wsi_fn_list = os.listdir(wsi_dir)
    for idx, wsi_fn in enumerate(wsi_fn_list):
        wsi_fn_pure, ext = os.path.splitext(wsi_fn)
        print("%d/%d, Reading information from %s" % (idx, len(wsi_fn_list), wsi_fn))
        wsi_obj = OpenSlide(os.path.join(wsi_dir, wsi_fn))
        wsi_prop = dict(wsi_obj.properties)
        des = wsi_prop["openslide.comment"]

        # use exposure time as the criteria
        assert "exposure time" in des.lower(), "No exposure time"
        et = get_exposure_time(des)
        if et in exposure_time_based_batches.keys():
            current_val = exposure_time_based_batches.get(et)
            exposure_time_based_batches[et] = current_val + [wsi_fn_pure]
        else:
            exposure_time_based_batches[et] = [wsi_fn_pure]
        exposure_time_set.add(et)

        #
        icc_pro = "Null"
        if "|ICC Profile" in des:
            ele = des.split("|")

            for e in ele:
                if "ICC Profile" in e:
                    icc_pro = e.replace("ICC Profile =", "").strip()
                    break
            if icc_pro in ICC_pro_based_batches.keys():
                current_val = ICC_pro_based_batches.get(icc_pro)
                ICC_pro_based_batches[icc_pro] = current_val + [wsi_fn_pure]
            else:
                ICC_pro_based_batches[icc_pro] = [wsi_fn_pure]
        else:
            current_val = ICC_pro_based_batches["Null"]
            if current_val:
                ICC_pro_based_batches["Null"] = current_val + [wsi_fn_pure]
            else:
                ICC_pro_based_batches["Null"] = [wsi_fn_pure]
        ICC_pro_set.add(icc_pro)
        #
        #     #
        gamma = "Null"
        if "|Gamma" in des:
            gamma = get_attri_from_description(des, "|Gamma =", "|")
            if gamma in gamma_val_based_batches.keys():
                current_val = gamma_val_based_batches.get(gamma)
                gamma_val_based_batches[gamma] = current_val + [wsi_fn_pure]
            else:
                gamma_val_based_batches[gamma] = [wsi_fn_pure]
        else:
            current_val = gamma_val_based_batches["Null"]
            if current_val:
                gamma_val_based_batches["Null"] = current_val + [wsi_fn_pure]
            else:
                gamma_val_based_batches["Null"] = [wsi_fn_pure]
        gamma_set.add(gamma)
    #

    f = open(exposure_time_based_batches_data_fn, "wb")
    pickle.dump(exposure_time_based_batches, f)
    f = open(ICC_pro_based_batches_data_fn, "wb")
    pickle.dump(ICC_pro_based_batches, f)
    f = open(gamma_val_based_batches_data_fn, "wb")
    pickle.dump(gamma_val_based_batches, f)

# # pprint(exposure_time_based_batches)
print("There are %d different exposure time." % len(exposure_time_set))
print(exposure_time_set)
print("There are %d different ICC profile." % len(ICC_pro_set))
print(ICC_pro_set)
print("There are %d different gamma values." % len(gamma_set))
print(gamma_set)

ET_45 = exposure_time_based_batches["45"]
ET_109 = exposure_time_based_batches["109"]
ET_8 = exposure_time_based_batches["8"]

ICC_AT2 = ICC_pro_based_batches["AT2"]
ICC_Null = ICC_pro_based_batches["Null"]
ICC_SSV1 = ICC_pro_based_batches["ScanScope v1"]

G_2_2 = gamma_val_based_batches["2.2"]
G_Null = gamma_val_based_batches["Null"]

subset_method1 = [ET_45, ET_109, ET_8]
subset_method2 = [ICC_AT2, ICC_Null, ICC_SSV1]
subset_method3 = [G_2_2, G_Null]


###########################
def get_case_ids(case_list):
    case_ids = []
    for c in case_list:
        case_id = c.split("_")[0]
        case_ids.append(case_id)
    return case_ids, len(set(case_ids))


case_ids, case_cnt = get_case_ids(ET_45)
print(case_cnt)

###########################################
# check RGB distribution differences in tumor and stroma regions
# (based-on tumor-stroma segmentation)
###########################################
id_text_list = ["Tumor", "Stroma"]
color_list = [[0, 255, 255], [255, 255, 0]]

# n_bins = 256
n_bins = 32

tumor_stroma_all_hist_fn = os.path.join(out_dir, "tumor_stroma_all_hist.pkl")
tumor_stroma_area_list_fn = os.path.join(out_dir, "tumor_stroma_area_list.pkl")
USED_CASE_IDS_fn = os.path.join(out_dir, "USED_CASE_IDS.pkl")
if os.path.exists(tumor_stroma_all_hist_fn) and os.path.exists(tumor_stroma_area_list_fn) and os.path.exists(
        USED_CASE_IDS_fn):
    file_to_read = open(tumor_stroma_all_hist_fn, "rb")
    tumor_r_hist, tumor_g_hist, tumor_b_hist, stroma_r_hist, stroma_g_hist, stroma_b_hist, \
    tumor_h_hist, tumor_s_hist, tumor_v_hist, stroma_h_hist, stroma_s_hist, stroma_v_hist = pickle.load(file_to_read)

    tumor_stroma_area_list_reader = open(tumor_stroma_area_list_fn, "rb")
    tumor_area_list, stroma_area_list = pickle.load(tumor_stroma_area_list_reader)

    USED_CASE_IDS_reader = open(USED_CASE_IDS_fn, "rb")
    USED_CASE_IDS = pickle.load(USED_CASE_IDS_reader)
else:
    USED_CASE_IDS = []
    fig, axs = plt.subplots(2, 3, tight_layout=True)
    tumor_r_hist = []
    tumor_g_hist = []
    tumor_b_hist = []
    stroma_r_hist = []
    stroma_g_hist = []
    stroma_b_hist = []

    tumor_h_hist = []
    tumor_s_hist = []
    tumor_v_hist = []
    stroma_h_hist = []
    stroma_s_hist = []
    stroma_v_hist = []

    tumor_area_list = []
    stroma_area_list = []
    for c_idx, case_id in enumerate(ET_45):
        print("%d/%d: processing %s" % (c_idx, len(ET_45), case_id))
        # if c_idx > 10:
        #     break
        thumbnail_fn = os.path.join(thumbnail_dir, case_id, case_id + "_thumbnail.png")
        segmentation_fn = os.path.join(thumbnail_dir, case_id, case_id + "_segmentation.png")
        if os.path.exists(thumbnail_fn) and os.path.exists(segmentation_fn):
            USED_CASE_IDS.append(case_id)
            thumbnail_img_rgb = Image.open(thumbnail_fn)
            thumbnail_rgb_arr = np.array(thumbnail_img_rgb)
            thumbnail_img_hsv = thumbnail_img_rgb.convert('HSV')
            thumbnail_hsv_arr = np.array(thumbnail_img_hsv)
            segmentation_arr = np.array(Image.open(segmentation_fn))[:, :, 0:3]

            stroma_idx = np.all(segmentation_arr == color_list[1], axis=2)
            stroma_area_list.append(np.count_nonzero(stroma_idx))
            stroma_rgb = thumbnail_rgb_arr[stroma_idx]
            tumor_idx = np.all(segmentation_arr == color_list[0], axis=2)
            tumor_area_list.append(np.count_nonzero(tumor_idx))
            tumor_rgb = thumbnail_rgb_arr[tumor_idx]

            stroma_hsv = thumbnail_hsv_arr[stroma_idx]
            tumor_hsv = thumbnail_hsv_arr[tumor_idx]

            # axs[0, 0].hist(tumor_rgb[:,0], edgecolor='red', bins=n_bins, histtype=u'step', density=True)  # tumor R
            # axs[0, 1].hist(tumor_rgb[:,1], edgecolor='green', bins=n_bins, histtype=u'step', density=True)  # tumor G
            # axs[0, 2].hist(tumor_rgb[:,2], edgecolor='blue', bins=n_bins, histtype=u'step', density=True)  # tumor B
            # axs[1, 0].hist(stroma_rgb[:,0], edgecolor='red', bins=n_bins, histtype=u'step', density=True)  # stroma R
            # axs[1, 1].hist(stroma_rgb[:,1], edgecolor='green', bins=n_bins, histtype=u'step', density=True)  # stroma G
            # axs[1, 2].hist(stroma_rgb[:,2], edgecolor='blue', bins=n_bins, histtype=u'step', density=True)  # stroma B

            sns.kdeplot(tumor_rgb[:, 0], bw_adjust=256 / n_bins, ax=axs[0, 0])
            sns.kdeplot(tumor_rgb[:, 1], bw_adjust=256 / n_bins, ax=axs[0, 1])
            sns.kdeplot(tumor_rgb[:, 2], bw_adjust=256 / n_bins, ax=axs[0, 2])
            sns.kdeplot(stroma_rgb[:, 0], bw_adjust=256 / n_bins, ax=axs[1, 0])
            sns.kdeplot(stroma_rgb[:, 1], bw_adjust=256 / n_bins, ax=axs[1, 1])
            sns.kdeplot(stroma_rgb[:, 2], bw_adjust=256 / n_bins, ax=axs[1, 2])

            r_hist, _ = np.histogram(tumor_rgb[:, 0], bins=n_bins, range=(0.0, 255.0), density=True)
            tumor_r_hist.append(r_hist)
            g_hist, _ = np.histogram(tumor_rgb[:, 1], bins=n_bins, range=(0.0, 255.0), density=True)
            tumor_g_hist.append(g_hist)
            b_hist, _ = np.histogram(tumor_rgb[:, 2], bins=n_bins, range=(0.0, 255.0), density=True)
            tumor_b_hist.append(b_hist)
            r_hist, _ = np.histogram(stroma_rgb[:, 0], bins=n_bins, range=(0.0, 255.0), density=True)
            stroma_r_hist.append(r_hist)
            g_hist, _ = np.histogram(stroma_rgb[:, 1], bins=n_bins, range=(0.0, 255.0), density=True)
            stroma_g_hist.append(g_hist)
            b_hist, _ = np.histogram(stroma_rgb[:, 2], bins=n_bins, range=(0.0, 255.0), density=True)
            stroma_b_hist.append(b_hist)

            h_hist, _ = np.histogram(tumor_hsv[:, 0], bins=n_bins, range=(0.0, 255.0), density=True)
            tumor_h_hist.append(h_hist)
            s_hist, _ = np.histogram(tumor_hsv[:, 1], bins=n_bins, range=(0.0, 255.0), density=True)
            tumor_s_hist.append(s_hist)
            v_hist, _ = np.histogram(tumor_hsv[:, 2], bins=n_bins, range=(0.0, 255.0), density=True)
            tumor_v_hist.append(v_hist)
            h_hist, _ = np.histogram(stroma_hsv[:, 0], bins=n_bins, range=(0.0, 255.0), density=True)
            stroma_h_hist.append(h_hist)
            s_hist, _ = np.histogram(stroma_hsv[:, 1], bins=n_bins, range=(0.0, 255.0), density=True)
            stroma_s_hist.append(s_hist)
            v_hist, _ = np.histogram(stroma_hsv[:, 2], bins=n_bins, range=(0.0, 255.0), density=True)
            stroma_v_hist.append(v_hist)

    f = open(tumor_stroma_all_hist_fn, "wb")
    pickle.dump([tumor_r_hist, tumor_g_hist, tumor_b_hist, stroma_r_hist, stroma_g_hist, stroma_b_hist,
                 tumor_h_hist, tumor_s_hist, tumor_v_hist, stroma_h_hist, stroma_s_hist, stroma_v_hist], f)

    f = open(tumor_stroma_area_list_fn, "wb")
    pickle.dump([tumor_area_list, stroma_area_list], f)

    f = open(USED_CASE_IDS_fn, "wb")
    pickle.dump(USED_CASE_IDS, f)

    axs[0, 0].set_title("Tumor R")
    axs[0, 1].set_title("Tumor G")
    axs[0, 2].set_title("Tumor B")
    axs[1, 0].set_title("Stroma R")
    axs[1, 1].set_title("Stroma G")
    axs[1, 2].set_title("Stroma B")
    save_to = os.path.join(out_dir, "Tumor_Stroma_RGB_distribution.png")
    plt.savefig(save_to)
    plt.show()

##############################
case_ids, case_cnt = get_case_ids(USED_CASE_IDS)
case_per_img_cnt = []
for i in set(case_ids):
    case_per_img = 0
    for c in case_ids:
        if c == i:
            case_per_img += 1
    case_per_img_cnt.append(case_per_img)
    print(case_per_img)

max(case_per_img_cnt)
for i in set(case_per_img_cnt):
    pc = len(np.nonzero(np.array(case_per_img_cnt) == i)[0])
    print("%d per case count: %d" % (i, pc))
print(case_cnt)

##############################
a = np.array(tumor_r_hist)
b = np.array(tumor_g_hist)
c = np.array(tumor_b_hist)
d = np.array(stroma_r_hist)
e = np.array(stroma_g_hist)
f = np.array(stroma_b_hist)

g = np.array(tumor_h_hist)
h = np.array(tumor_s_hist)
i = np.array(tumor_v_hist)
j = np.array(stroma_h_hist)
k = np.array(stroma_s_hist)
h = np.array(stroma_v_hist)

All_distribution = np.concatenate([a, b, c, d, e, f], axis=1)
# All_distribution = np.concatenate([g, h, i, j, k, h], axis=1)
##############################
row_linkage = hierarchy.linkage(
    distance.pdist(All_distribution), method='average')

fig = plt.figure(figsize=(35, 6), dpi=150)
dn = dendrogram(row_linkage)
plt.show()
#########################
# # get the outliers according to dendrogram
#########################
# outliers_idx = dn["ivl"][0:9]
# outliers_idx = [int(i) for i in outliers_idx]
# outliers_case_id = [USED_CASE_IDS[i] for i in outliers_idx]
# new_folder = os.path.join(out_dir, "dendrogram_outliers")
# if not os.path.exists(new_folder):
#     os.mkdir(new_folder)
# for f in outliers_case_id:
#     case_path_regx = os.path.join(thumbnail_dir, f, f+"*.png")
#     for fn in glob.glob(case_path_regx):
#         shutil.copy(fn, new_folder)
#########################
#########################
# # save RGB distribution and tumor stroma area size into files
# wrt_str = "case_id,tumor_area,stroma_area"
# for i in range(n_bins):
#     wrt_str += ",tumor_r_hist_" + str(i)
# for i in range(n_bins):
#     wrt_str += ",tumor_g_hist_" + str(i)
# for i in range(n_bins):
#     wrt_str += ",tumor_b_hist_" + str(i)
# for i in range(n_bins):
#     wrt_str += ",stroma_r_hist_" + str(i)
# for i in range(n_bins):
#     wrt_str += ",stroma_g_hist_" + str(i)
# for i in range(n_bins):
#     wrt_str += ",stroma_b_hist_" + str(i)
# wrt_str += "\n"
# for idx, c in enumerate(USED_CASE_IDS):
#     string_t_r = [str(h) for h in tumor_r_hist[idx]]
#     string_t_g = [str(h) for h in tumor_g_hist[idx]]
#     string_t_b = [str(h) for h in tumor_b_hist[idx]]
#     string_s_r = [str(h) for h in stroma_r_hist[idx]]
#     string_s_g = [str(h) for h in stroma_g_hist[idx]]
#     string_s_b = [str(h) for h in stroma_b_hist[idx]]
#     wrt_str += USED_CASE_IDS[idx] + "," + str(tumor_area_list[idx]) + "," + str(stroma_area_list[idx]) + "," \
#                + ",".join(string_t_r) + "," + ",".join(string_t_g) + "," + ",".join(string_t_b) + "," \
#                + ",".join(string_s_r) + "," + ",".join(string_s_g) + "," + ",".join(string_s_b) + "\n"
#     # print(wrt_str)
# ele = wrt_str.split("\n")
# fp = open(os.path.join(out_dir, "Area-RGB_hist.csv"), "w")
# fp.write(wrt_str)
# fp.close()
#########################
# Refer to this site:
# http://dawnmy.github.io/2016/10/24/Plot-heatmaap-with-side-color-indicating-the-class-of-variables/
# add Tumor area bar to  RGB clustermap
normalized_tumor_area_list = (np.array(tumor_area_list) - min(tumor_area_list)) / (
        max(tumor_area_list) - min(tumor_area_list))
row_colors = [cm.get_cmap("Blues")(i)[:3] for i in normalized_tumor_area_list]
# g = sns.clustermap(All_distribution, row_colors=row_colors, linewidths=0, xticklabels=False, yticklabels=False)
g = sns.clustermap(All_distribution, col_cluster=False, row_colors=row_colors, linewidths=0, xticklabels=False, yticklabels=False)
# plt.show()
save_to = os.path.join(out_dir, "Tumor_Stroma_RGB_distribution.png")
# save_to = os.path.join(out_dir, "Tumor_Stroma_hsv_distribution.png")
plt.savefig(save_to)

##########################################################
cluster_id_list = fcluster(row_linkage, t=4, criterion='maxclust')
cluster_size_list = []
clustered_tumor_area_dict = {}
for i in sorted(set(cluster_id_list)):
    cluster_size_list.append(len(np.nonzero(cluster_id_list == i)[0]))
    tmp_list = []
    for idx, _ in enumerate(cluster_id_list):
        if cluster_id_list[idx] == i:
            tmp_list.append(normalized_tumor_area_list[idx])
    clustered_tumor_area_dict["cluster_" + str(i)] = tmp_list

max_cluster_size = max(cluster_size_list)

df_list = []
for k in clustered_tumor_area_dict.keys():
    c = clustered_tumor_area_dict[k]
    if len(c) > 5:
        df = pd.DataFrame()
        df[k] = c
        df_list.append(df)

dfa = pd.concat(df_list, ignore_index=True, axis=1)
boxes_sep = 0.4
sns.boxplot(data=dfa, width=boxes_sep)
plt.show()
##########################################################
# sort RGB distribution with tumor area size
####################################################
sorted_idx = np.argsort(normalized_tumor_area_list)
sorted_idx_arr = np.repeat(np.expand_dims(sorted_idx, 1), All_distribution.shape[1], axis=1)
new_all_distribution = np.take_along_axis(All_distribution, sorted_idx_arr, axis=0)
plt.figure(figsize=(10, 8), dpi=150)
sns.heatmap(new_all_distribution, yticklabels=sorted_idx)
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(50))
plt.grid(False)
plt.show()

All_distribution_corr = np.corrcoef(All_distribution)
sns.clustermap(All_distribution_corr)
save_to = os.path.join(out_dir, "Tumor_Stroma_RGB_distribution_corr.png")
plt.savefig(save_to)

###################
###################
fig2, axs2 = plt.subplots(2, 3, tight_layout=True)
tumor_r_corr = np.corrcoef(tumor_r_hist)
axs2[0, 0].imshow(tumor_r_corr, cmap="jet")  # tumor R
tumor_g_corr = np.corrcoef(tumor_g_hist)
axs2[0, 1].imshow(tumor_g_corr, cmap="jet")  # tumor G
tumor_b_corr = np.corrcoef(tumor_b_hist)
axs2[0, 2].imshow(tumor_b_corr, cmap="jet")  # tumor B
stroma_r_corr = np.corrcoef(stroma_r_hist)
axs2[1, 0].imshow(stroma_r_corr, cmap="jet")  # stroma R
stroma_g_corr = np.corrcoef(stroma_g_hist)
axs2[1, 1].imshow(stroma_g_corr, cmap="jet")  # stroma G
stroma_b_corr = np.corrcoef(stroma_b_hist)
axs2[1, 2].imshow(stroma_b_corr, cmap="jet")  # stroma B

axs2[0, 0].set_title("Tumor R distribution corr")
axs2[0, 1].set_title("Tumor G distribution corr")
axs2[0, 2].set_title("Tumor B distribution corr")
axs2[1, 0].set_title("Stroma R distribution corr")
axs2[1, 1].set_title("Stroma G distribution corr")
axs2[1, 2].set_title("Stroma B distribution corr")
plt.show()

sns.clustermap(stroma_r_corr)
save_to = os.path.join(out_dir, "Stroma R distribution corr.png")
plt.savefig(save_to)

sns.clustermap(stroma_g_corr)
save_to = os.path.join(out_dir, "Stroma G distribution corr.png")
plt.savefig(save_to)

sns.clustermap(stroma_b_corr)
save_to = os.path.join(out_dir, "Stroma B distribution corr.png")
plt.savefig(save_to)

sns.clustermap(tumor_r_corr)
save_to = os.path.join(out_dir, "Tumor R distribution corr.png")
plt.savefig(save_to)

sns.clustermap(tumor_g_corr)
save_to = os.path.join(out_dir, "Tumor G distribution corr.png")
plt.savefig(save_to)

sns.clustermap(tumor_b_corr)
save_to = os.path.join(out_dir, "Tumor B distribution corr.png")
plt.savefig(save_to)
###########################################
# check tissue years
###########################################
print("Total number in batch 1 %d." % len(USED_CASE_IDS))
years = []
year_case_dict = {}
no_year_case_list = []
for c in USED_CASE_IDS:
    if "_HE" in c:
        trim_c = c.replace("_HE", "")
        ele = trim_c.split("-")[-1]
        if ele.isnumeric():
            years.append(ele)
            ys = year_case_dict.get(ele)
            if ys:
                year_case_dict[ele] = ys + [c]
            else:
                year_case_dict[ele] = [c]
        else:
            no_year_case_list.append(c)
    else:
        no_year_case_list.append(c)
years_set = sorted(set(years))
print(years_set)
print("Uniq year count: %d " % len(years_set))
###########################################

output_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/batch_effect/years_side_by_side"
year_group_cnt_list = []
for key, value in year_case_dict.items():
    year_group_cnt_list.append(len([item for item in value if item]))
year_group_cnt = min(year_group_cnt_list[0:-1])
for i in range(year_group_cnt):
    sv_img_arr = np.zeros((256 * 4, 256 * 4, 3), dtype=np.uint8)
    for idx, y in enumerate(years_set[0:-1]):
        case_id = year_case_dict[y][i]
        img_fn = os.path.join(thumbnail_dir, case_id, case_id + "_thumbnail.png")
        if os.path.exists(img_fn):
            img = Image.open(img_fn).resize([256, 256])
            row_idx = int(idx / 4)
            col_idx = int(idx % 4)
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), y, fill=(255, 0, 0))
            sv_img_arr[256 * row_idx: 256 * (row_idx + 1), 256 * col_idx:256 * (col_idx + 1), :] = np.array(img)[:, :,
                                                                                                   0:3]
        else:
            print(img_fn)
            # raise Exception("Image file not exist")
    save_to = os.path.join(output_dir, "sample_" + str(i) + ".jpg")
    Image.fromarray(sv_img_arr).save(save_to)

####################################################
output_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/batch_effect/side_by_side"
file_list_txt = "batch_file_names.txt"
fp = open(os.path.join(output_dir, file_list_txt), 'w')
wrt_str = "idx, case_1, case_2, case3\n"
min_length = min([len(ET_109), len(ET_8), len(ET_109)])
for idx in range(min_length):
    sv_img_arr = np.zeros((512, 1536, 3), dtype=np.uint8)
    left_img_fn = os.path.join(thumbnail_dir, ET_45[idx], ET_45[idx] + "_thumbnail.png")
    mid_img_fn = os.path.join(thumbnail_dir, ET_109[idx], ET_109[idx] + "_thumbnail.png")
    right_img_fn = os.path.join(thumbnail_dir, ET_8[idx], ET_8[idx] + "_thumbnail.png")
    if idx == 98:
        print(ET_45[idx])
        print(ET_109[idx])
        print(ET_8[idx])
    if os.path.exists(left_img_fn) and os.path.exists(mid_img_fn) and os.path.exists(right_img_fn):
        left_img = Image.open(left_img_fn).resize([512, 512])
        mid_img = Image.open(mid_img_fn).resize([512, 512])
        right_img = Image.open(right_img_fn).resize([512, 512])
        sv_img_arr[:, 0:512, :] = np.array(left_img)[:, :, 0:3]
        sv_img_arr[:, 512:1024, :] = np.array(mid_img)[:, :, 0:3]
        sv_img_arr[:, 1024:, :] = np.array(right_img)[:, :, 0:3]
        save_to = os.path.join(output_dir, str(idx) + ".jpg")
        Image.fromarray(sv_img_arr).save(save_to)
        wrt_str += str(idx) + "," + ET_45[idx] + "," + ET_109[idx] + "," + ET_8[idx] + "\n"

fp.write(wrt_str)
fp.close()

# Draw intersection matrix
intersection_matrix = np.zeros([len(subset_method1), len(subset_method2)])
for idx1, ss1 in enumerate(subset_method1):
    for idx2, ss2 in enumerate(subset_method2):
        intersection_matrix[idx1, idx2] = len(set(ss1) & set(ss2))

fig, ax = plt.subplots()
ax.matshow(intersection_matrix, cmap=plt.cm.Blues)

for i in range(len(subset_method1)):
    for j in range(len(subset_method2)):
        c = intersection_matrix[i, j]
        ax.text(j, i, str(c), va='center', ha='center')

plt.yticks(range(0, len(subset_method1)), ["ET_45", "ET_109", "ET_8"])
plt.xticks(range(0, len(subset_method2)), ["ICC_AT2", "ICC_Null", "ICC_SSV1"])
plt.title("Exposure time and ICC intersection")
plt.show()

#
intersection_matrix = np.zeros([len(subset_method1), len(subset_method3)])
for idx1, ss1 in enumerate(subset_method1):
    for idx2, ss2 in enumerate(subset_method3):
        intersection_matrix[idx1, idx2] = len(set(ss1) & set(ss2))

fig, ax = plt.subplots()
ax.matshow(intersection_matrix, cmap=plt.cm.Blues)

for i in range(len(subset_method1)):
    for j in range(len(subset_method3)):
        c = intersection_matrix[i, j]
        ax.text(j, i, str(c), va='center', ha='center')

plt.yticks(range(0, len(subset_method1)), ["ET_45", "ET_109", "ET_8"])
plt.xticks(range(0, len(subset_method3)), ["G_2_2", "G_Null"])
plt.title("Exposure time and Gamma intersection")
plt.show()

#
intersection_matrix = np.zeros([len(subset_method3), len(subset_method2)])
for idx1, ss1 in enumerate(subset_method3):
    for idx2, ss2 in enumerate(subset_method2):
        intersection_matrix[idx1, idx2] = len(set(ss1) & set(ss2))

fig, ax = plt.subplots()
ax.matshow(intersection_matrix, cmap=plt.cm.Blues)

for i in range(len(subset_method3)):
    for j in range(len(subset_method2)):
        c = intersection_matrix[i, j]
        ax.text(j, i, str(c), va='center', ha='center')

plt.yticks(range(0, len(subset_method3)), ["G_2_2", "G_Null"])
plt.xticks(range(0, len(subset_method2)), ["ICC_AT2", "ICC_Null", "ICC_SSV1"])
plt.title("Gamma and ICC intersection")
plt.show()

# compare data batch with analysis from R
survival_analysis_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/result_analysis/survival"
All_cases_fn = os.path.join(survival_analysis_dir, "all_cases.csv")
RD0_cases_fn = os.path.join(survival_analysis_dir, "RD0_cases.csv")
RD12_cases_fn = os.path.join(survival_analysis_dir, "RD1.2_cases.csv")

All_cases = []
with open(All_cases_fn) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        All_cases.append(row["x"])

RD0_cases = []
with open(RD0_cases_fn) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        RD0_cases.append(row["x"])

RD12_cases = []
with open(RD12_cases_fn) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        RD12_cases.append(row["x"])

batch1_RD0 = set(ET_45) & set(RD0_cases)
batch2_RD0 = set(ET_8) & set(RD0_cases)
batch3_RD0 = set(ET_109) & set(RD0_cases)
print(len(batch1_RD0))
print(len(batch2_RD0))
print(len(batch3_RD0))

batch1_RD12 = set(ET_45) & set(RD12_cases)
batch2_RD12 = set(ET_8) & set(RD12_cases)
batch3_RD12 = set(ET_109) & set(RD12_cases)
print(len(batch1_RD12))
print(len(batch2_RD12))
print(len(batch3_RD12))

import pandas as pd

df = pd.read_excel(
    "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/result_analysis/All_cases_metadata_12_08_2021.xlsx")

batch1_df = df[df["deidentified_id"].isin(ET_45)]
batch1_save_to = os.path.join(survival_analysis_dir, "batch1_metadata.tsv")
batch1_df.to_csv(batch1_save_to, sep="\t")
# batch1_save_to = os.path.join(survival_analysis_dir, "batch1_metadata.xlsx")
# batch1_df.to_excel(batch1_save_to)

batch2_df = df[df["deidentified_id"].isin(ET_8)]
batch2_save_to = os.path.join(survival_analysis_dir, "batch2_metadata.tsv")
batch2_df.to_csv(batch2_save_to, sep="\t")
# batch2_save_to = os.path.join(survival_analysis_dir, "batch2_metadata.xlsx")
# batch2_df.to_excel(batch2_save_to)

batch3_df = df[df["deidentified_id"].isin(ET_109)]
batch3_save_to = os.path.join(survival_analysis_dir, "batch3_metadata.tsv")
batch3_df.to_csv(batch3_save_to, sep="\t")
# batch3_save_to = os.path.join(survival_analysis_dir, "batch3_metadata.xlsx")
# batch3_df.to_excel(batch3_save_to)

# batch_r12_save_to = os.path.join(survival_analysis_dir, "batch_R12.csv")
# wrt_str = "idx,case_id\n"
# for idx, b_r0 in enumerate(batch1_RD12):
#     c_id = b_r0.split("_")[0]
#     wrt_str += str(idx) + "," + b_r0 + "\n"
# wrt_str = wrt_str[0:-1]
# fp = open(batch_r12_save_to, 'w')
# fp.write(wrt_str)
# fp.close()


case_ids, case_cnt = get_case_ids(ET_45)
print(case_cnt)
case_ids, case_cnt = get_case_ids(ET_8)
print(case_cnt)
case_ids, case_cnt = get_case_ids(ET_109)
print(case_cnt)

case_ids, case_cnt = get_case_ids(batch1_RD0)
print(case_cnt)
case_ids, case_cnt = get_case_ids(batch2_RD0)
print(case_cnt)
case_ids, case_cnt = get_case_ids(batch3_RD0)
print(case_cnt)

case_ids, case_cnt = get_case_ids(batch1_RD12)
print(case_cnt)
case_ids, case_cnt = get_case_ids(batch2_RD12)
print(case_cnt)
case_ids, case_cnt = get_case_ids(batch3_RD12)
print(case_cnt)

print("Done!")
