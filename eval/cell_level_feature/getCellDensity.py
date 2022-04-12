import os
import multiprocessing

roi_csv_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/result_analysis/cell_rois"
cell_prediction_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/result_analysis/roi_cell_classifications"
out_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/result_analysis/cell_density"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
#
# fns = sorted(os.listdir(cell_prediction_dir))
# case_list = []
# for f in fns:
#     case_list.append(f.replace("_cell_features_prediction.txt", ""))
#

def get_list_from_folder_complimentary(dir_whole, dir_sub):
    f_list_1 = os.listdir(dir_whole)
    f_list_2 = os.listdir(dir_sub)
    f_list = [f.replace("_cell_features_prediction.txt","") for f in f_list_1 if f.replace("_cell_features_prediction.txt", "_cell_density.csv") not in f_list_2]
    return f_list


case_list = get_list_from_folder_complimentary(cell_prediction_dir, out_dir)

# for c in case_list:
#     save_to = os.path.join(out_dir, c+"_cell_density.csv")
#     wrt_str = "case_id,roi_area,tumor_density,stroma_density\n"
#     fp = open(save_to, 'w')
#     # read ROI from csv
#     roi_fn = os.path.join(roi_csv_dir, c+"_roi_box.csv")
#     rois = open(roi_fn, 'r').readlines()[1:]
#     # read cell locations
#     cell_prediction_fn = os.path.join(cell_prediction_dir, c+"_cell_features_prediction.txt")
#     cell_predictions = open(cell_prediction_fn, 'r').readlines()[1:]
#     for r in rois:
#         tumor_cell_cnt = 0
#         stroma_cell_cnt = 0
#         tx, ty, bx, by = r.strip().split(",")
#         for cp in cell_predictions:
#             cp_ele = cp.strip().split("\t")
#             cell_x = float(cp_ele[5])/0.25
#             cell_y = float(cp_ele[6])/0.25
#             cell_type = cp_ele[2]
#             if cell_x > float(tx) and cell_x < float(bx) and cell_y > float(ty) and cell_y < float(by):
#                 if cell_type == "Tumor":
#                     tumor_cell_cnt += 1
#                 elif cell_type == "Stroma":
#                     stroma_cell_cnt += 1
#         roi_area = (float(tx)-float(bx))*(float(ty)-float(by))/1000.0
#         tumor_density = float(tumor_cell_cnt)/roi_area
#         stroma_density = float(stroma_cell_cnt)/roi_area
#         cell_density = tumor_density + stroma_density
#         wrt_str += c + "," + str(roi_area) + "," + str(tumor_density) + "," + str(stroma_density) + "\n"
#
#     fp.write(wrt_str)
#     fp.close()

def get_cell_density(c):
    save_to = os.path.join(out_dir, c + "_cell_density.csv")
    wrt_str = "case_id,roi_area,tumor_density,stroma_density\n"
    fp = open(save_to, 'w')
    # read ROI from csv
    roi_fn = os.path.join(roi_csv_dir, c + "_roi_box.csv")
    rois = open(roi_fn, 'r').readlines()[1:]
    # read cell locations
    cell_prediction_fn = os.path.join(cell_prediction_dir, c + "_cell_features_prediction.txt")
    cell_predictions = open(cell_prediction_fn, 'r').readlines()[1:]
    for r in rois:
        tumor_cell_cnt = 0
        stroma_cell_cnt = 0
        tx, ty, bx, by = r.strip().split(",")
        for cp in cell_predictions:
            cp_ele = cp.strip().split("\t")
            cell_x = float(cp_ele[5]) / 0.25
            cell_y = float(cp_ele[6]) / 0.25
            cell_type = cp_ele[2]
            if cell_x > float(tx) and cell_x < float(bx) and cell_y > float(ty) and cell_y < float(by):
                if cell_type == "Tumor":
                    tumor_cell_cnt += 1
                elif cell_type == "Stroma":
                    stroma_cell_cnt += 1
        roi_area = (float(tx) - float(bx)) * (float(ty) - float(by)) / 1000.0
        tumor_density = float(tumor_cell_cnt) / roi_area
        stroma_density = float(stroma_cell_cnt) / roi_area
        cell_density = tumor_density + stroma_density
        wrt_str += c + "," + str(roi_area) + "," + str(tumor_density) + "," + str(stroma_density) + "\n"

    fp.write(wrt_str)
    fp.close()

a_pool = multiprocessing.Pool()
a_pool.map(get_cell_density, case_list)














