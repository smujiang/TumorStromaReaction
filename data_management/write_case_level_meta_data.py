import os
from data_manager import WSI_data
import numpy as np

TSR_score_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/result_analysis"
cell_prediction_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/result_analysis/roi_cell_classifications"
cell_density_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/result_analysis/cell_density"
out_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/result_analysis/metadata"
output_template = "case_level_data_demo.ini"

SBOT_case_list = ["OCMC-016","OCMC-017","OCMC-018","OCMC-019","OCMC-020","OCMC-021",\
                  "OCMC-022","OCMC-023","OCMC-024","OCMC-025","OCMC-026","OCMC-027","OCMC-028","OCMC-029","OCMC-030"]

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

fns = sorted(os.listdir(cell_prediction_dir))
case_list = []
for f in fns:
    case_list.append(f.replace("_cell_features_prediction.txt", ""))

# load TSR scores
# HGSOC_tsr = open(os.path.join(TSR_score_dir, "HGSOC_hist_rec_overall.csv"), "r").readlines()[1:]
# SBOT_tsr = open(os.path.join(TSR_score_dir, "SBOT_hist_rec_overall.csv"), "r").readlines()[1:]
# tsr_header = open(os.path.join(TSR_score_dir, "HGSOC_hist_rec.csv"), "r").readlines()[0]
# # HGSOC_tsr = open(os.path.join(TSR_score_dir, "HGSOC_hist_rec.csv"), "r").readlines()[1:]
# # SBOT_tsr = open(os.path.join(TSR_score_dir, "SBOT_hist_rec.csv"), "r").readlines()[1:]
# TSR_score_lines = HGSOC_tsr + SBOT_tsr

TSR_score_lines = open(os.path.join(TSR_score_dir, "all_hist_rec.csv"), "r").readlines()[1:]
tsr_header = open(os.path.join(TSR_score_dir, "all_hist_rec.csv"), "r").readlines()[0]

Header_created = False  # is output csv header created.
all_case_csv = os.path.join(out_dir, "all_case_metadata.csv")
if os.path.exists(all_case_csv):
    os.remove(all_case_csv)
all_csv_fp = open(all_case_csv, 'a')
for c in case_list:
    print("Processing %s" % c)
    cell_feature_prediction_fn = os.path.join(cell_prediction_dir, c+"_cell_features_prediction.txt")
    cell_feature_prediction_lines = open(cell_feature_prediction_fn, "r").readlines()
    feature_names = cell_feature_prediction_lines[0].strip()[cell_feature_prediction_lines[0].index('\t', 50)+1:].replace("\t", ",")
    cell_features = cell_feature_prediction_lines[1:]
    cell_density_fn = os.path.join(cell_density_dir, c+"_cell_density.csv") # duplicate cell type? in the prediction files?
    cell_density_lines = open(cell_density_fn, "r").readlines()
    cell_density_feature_names = cell_density_lines[0].strip()[cell_density_lines[0].index(',')+1:]
    cell_density_features = cell_density_lines[1:]

    TSR_scores = ""
    # get TSR scores
    for tsr in TSR_score_lines:
        if c in tsr:
            TSR_scores = tsr[(tsr.index(",")+1):].strip()
            break
    if len(TSR_scores) > 1:
        # get cell features
        tumor_cell_features = []
        stroma_cell_features = []
        for cp in cell_features:
            cp_ele = cp.strip().split("\t")
            cell_type = cp_ele[2]
            feature_values = []
            for fstr in cp_ele[7:]:
                feature_values.append(float(fstr))
            if "Tumor" == cell_type:
                tumor_cell_features.append(feature_values)
            else:
                stroma_cell_features.append(feature_values)
        tumor_cell_features = np.array(tumor_cell_features)
        stroma_cell_features = np.array(stroma_cell_features)
        tumor_feature_mean = np.mean(tumor_cell_features, axis=0)
        tumor_feature_std = np.std(tumor_cell_features, axis=0)
        stroma_feature_mean = np.mean(stroma_cell_features, axis=0)
        stroma_feature_std = np.std(stroma_cell_features, axis=0)

        cell_density = []
        for den_str in cell_density_features:
            cd = []
            for d in den_str.split(",")[1:]:
                cd.append(float(d))
            cell_density.append(cd)
        cell_density = np.array(cell_density)
        cell_density_mean = np.mean(cell_density, axis=0)
        cell_density_std = np.std(cell_density, axis=0)

        # save to csv files
        if not Header_created:
            header = "scan_request_num,contact_name,PI_name,resolution,block_id,clinic_num,deidentified_id," \
                     "annotator,roi_cnt,tissue_site,tissue_type,fresh_cut,stain_type,histology_type"
            tumor_feature_names_mean = ""
            tumor_feature_names_std = ""
            stroma_feature_names_mean = ""
            stroma_feature_names_std = ""
            for feat_nm in feature_names.strip().split(","):
                tumor_feature_names_mean += "," + "tumor " + feat_nm +"_mean"
                tumor_feature_names_std += "," + "tumor " + feat_nm +"_std"
                stroma_feature_names_mean += "," + "stroma " + feat_nm +"_mean"
                stroma_feature_names_std += "," + "stroma " + feat_nm +"_std"
            header += tumor_feature_names_mean + tumor_feature_names_std + stroma_feature_names_mean + stroma_feature_names_std

            cdf_mean = ""
            cdf_std = ""
            for cdf in cell_density_feature_names.strip().split(","):
                cdf_mean += "," + cdf +"_mean"
                cdf_std += "," + cdf +"_std"
            header += cdf_mean + cdf_std

            tsr_header_names = tsr_header.strip()[tsr_header.index(','):]
            header += tsr_header_names + "\n"
            # write csv header
            all_csv_fp.write(header)
            Header_created = True

        meta_data_line = ""
        meta_data_line += " " # scan_request_num
        meta_data_line += "," + "Jun Jiang" # contact_name
        meta_data_line += "," + "Chen Wang" # PI_name
        meta_data_line += "," + "0.25" # resolution
        meta_data_line += "," + " " # block_id
        meta_data_line += "," + " " # clinic_num
        meta_data_line += "," + c # deidentified_id
        meta_data_line += "," + "Burak Tekin" # annotator
        meta_data_line += "," + "5" # roi_cnt
        meta_data_line += "," + "Primary" # tissue_site
        meta_data_line += "," + "Ovarian" # tissue_type
        meta_data_line += "," + "True" # fresh_cut
        meta_data_line += "," + "H&E" # stain_type
        if c in SBOT_case_list:
            meta_data_line += "," + "SBOT" # histology_type
        else:
            meta_data_line += "," + "HGSOC" # histology_type

        joined_string = ",".join([str(element) for element in tumor_feature_mean])
        meta_data_line += "," + joined_string
        joined_string = ",".join([str(element) for element in tumor_feature_std])
        meta_data_line += "," + joined_string
        joined_string = ",".join([str(element) for element in stroma_feature_mean])
        meta_data_line += "," + joined_string
        joined_string = ",".join([str(element) for element in stroma_feature_std])
        meta_data_line += "," + joined_string
        joined_string = ",".join([str(element) for element in cell_density_mean])
        meta_data_line += "," + joined_string
        joined_string = ",".join([str(element) for element in cell_density_std])
        meta_data_line += "," + joined_string
        meta_data_line += "," + TSR_scores + "\n"
        all_csv_fp.write(meta_data_line)
        '''
        # save to ini files.
        '''
        wsi_data = WSI_data(output_template)
        wsi_data.meta_data['scanning_info']['contact_name'] = "Jun Jiang"
        wsi_data.meta_data['scanning_info']['PI_name'] = "Chen Wang"

        wsi_data.meta_data['sample_info']['deidentified_ID'] = c
        wsi_data.meta_data['sample_info']['annotator'] = 'Burak Tekin'
        wsi_data.meta_data['sample_info']['tissue_site'] = 'Primary'
        wsi_data.meta_data['sample_info']['tissue_type'] = 'Ovarian'
        wsi_data.meta_data['sample_info']['fresh_cut'] = 'True'
        wsi_data.meta_data['sample_info']['stain_type'] = 'H&E'
        wsi_data.meta_data['sample_info']['histology_type'] = 'LGSOC'

        wsi_data.meta_data['cell_level_features']['cell_feature_names'] = feature_names
        wsi_data.meta_data['cell_level_features']['tumor_cell_mean'] = str(tumor_feature_mean)
        wsi_data.meta_data['cell_level_features']['tumor_cell_std'] = str(tumor_feature_std)
        wsi_data.meta_data['cell_level_features']['stroma_cell_mean'] = str(stroma_feature_mean)
        wsi_data.meta_data['cell_level_features']['stroma_cell_std'] = str(stroma_feature_std)
        wsi_data.meta_data['cell_level_features']['cell_density_feature_names'] = cell_density_feature_names
        wsi_data.meta_data['cell_level_features']['cell_density_mean'] = str(cell_density_mean)
        wsi_data.meta_data['cell_level_features']['cell_density_std'] = str(cell_density_std)

        wsi_data.meta_data['tissue_level_features']['TSR_score'] = TSR_scores
        write_to = os.path.join(out_dir, c + "_meta.ini")
        wsi_data.write2file(write_to)

all_csv_fp.close()

print("Done")


