import pickle
import os, sys
import numpy as np
import pandas as pd
import multiprocessing

def loadQuPathMeasurements_asDF(txt_file_list):
    df_all = pd.DataFrame()
    for txt_fn in txt_file_list:
        case_df = pd.read_csv(txt_fn, sep='\t')
        df_all = pd.concat([df_all, case_df])
    return df_all

def get_list_from_folder_complimentary(dir_whole, dir_sub):
    f_list_1 = os.listdir(dir_whole)
    f_list_2 = os.listdir(dir_sub)
    f_list = [f for f in f_list_1 if f.replace(".txt", "_prediction.txt") not in f_list_2]
    return f_list

feature_data_root = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/result_analysis/roi_cell_features"
model_root = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/result_analysis/cell_classification_model"
output_root = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/result_analysis/roi_cell_classifications"

class_names = ["Tumor", "Stroma"]

option = 3
# option 1: read case list from txt file:
if option == 1:
    fp_r = open("error_files.txt", 'r')
    err_lines = fp_r.readlines()
    fp_r.close()

    if len(err_lines) > 0:
        txt_fn_list = []
        for el in err_lines:
            txt_fn_list.append(el.strip() + "_cell_features.txt")
        os.remove("error_files.txt")
    else:
        txt_fn_list = sorted(os.listdir(feature_data_root))


# option 2: load case list from listing the folder
if option == 2:
    txt_fn_list = os.listdir(feature_data_root)

# get case list from folder complimentary
if option == 3:
    txt_fn_list = get_list_from_folder_complimentary(feature_data_root, output_root)


# Load trained model from file
pkl_filename = os.path.join(model_root, "cell_classification_model.pkl")
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

#
# fp = open("error_files.txt", 'a')
# for txt_fn in txt_fn_list:
#     # txt_fn = "Wo-1-A5_RIO1338_HE_cell_features.txt"
#     print("processing %s" % txt_fn)
#     case_id = txt_fn.replace("_cell_features.txt", "")
#     data_txt = os.path.join(feature_data_root, txt_fn)
#     df = loadQuPathMeasurements_asDF([data_txt])
#     test_cell_data = np.array(df.iloc[:, 7:])
#     if test_cell_data.shape[1] == 41:
#         test_cell_location = np.array(df.iloc[:, 5:7])
#
#         predictions = pickle_model.predict(test_cell_data)
#         for idx, p in enumerate(predictions):
#             if p == 1:
#                 df.iloc[idx, 2] = "Tumor"
#             else:
#                 df.iloc[idx, 2] = "Stroma"
#
#         # Save to
#         data_out_txt = os.path.join(output_root, txt_fn.replace(".txt", "_prediction.txt"))
#         df.to_csv(data_out_txt, index=None, sep='\t')
#     else:
#         fp.write(case_id+"\n")
#
# fp.close()


def predict_cells(model_case_id_tuple):
    pickle_model, txt_fn = model_case_id_tuple
    print("processing %s" % txt_fn)
    case_id = txt_fn.replace("_cell_features.txt", "")
    data_txt = os.path.join(feature_data_root, txt_fn)
    df = loadQuPathMeasurements_asDF([data_txt])
    test_cell_data = np.array(df.iloc[:, 7:])
    if test_cell_data.shape[1] == 41:
        test_cell_location = np.array(df.iloc[:, 5:7])

        predictions = pickle_model.predict(test_cell_data)
        for idx, p in enumerate(predictions):
            if p == 1:
                df.iloc[idx, 2] = "Tumor"
            else:
                df.iloc[idx, 2] = "Stroma"

        # Save to
        data_out_txt = os.path.join(output_root, txt_fn.replace(".txt", "_prediction.txt"))
        df.to_csv(data_out_txt, index=None, sep='\t')


models = [pickle_model] * len(txt_fn_list)

a_pool = multiprocessing.Pool()
a_pool.map(predict_cells, zip(models, txt_fn_list))

