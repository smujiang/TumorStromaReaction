import os


data_from_img = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/result_analysis/BATCH_1_DATA_Jun_Jan3_2022.tsv"
data_from_ehr = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/result_analysis/mayo_clinical_data_Feb12_2018.csv"

match_table_fn = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/result_analysis/clinical_data.csv"
redundant_match_table_fn = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/result_analysis/matched_meta_data.csv"

##################################################################################
if not os.path.exists(redundant_match_table_fn):
    d_img = open(data_from_img, 'r').readlines()
    d_ehr = open(data_from_ehr, 'r').readlines()
    reserved_lines = [d_img[0].replace("\t", ",").replace("\n", ",") + d_ehr[0]]
    for d_i in d_img[1:]:
        ele = d_i.split("\t")
        clinic_num_d_i = ele[5]
        for d_e in d_ehr[1:]:
            ele2 = d_e.split(",")
            clinic_num_d_e = ele2[0]
            if clinic_num_d_i == clinic_num_d_e:
                reserved_lines.append(d_i.replace("\t", ",").replace("\n", ",") + d_e)

    fp = open(redundant_match_table_fn, 'a')
    for i in reserved_lines:
        fp.write(i)
    fp.close()

if not os.path.exists(match_table_fn):
    d_img = open(data_from_img, 'r').readlines()
    d_ehr = open(data_from_ehr, 'r').readlines()
    reserved_lines = [d_ehr[0]]
    clinic_num_d_i_set = set([])
    for d_i in d_img[1:]:
        ele = d_i.split("\t")
        clinic_num_d_i = ele[5]
        clinic_num_d_i_set.add(clinic_num_d_i)
    for d_e in d_ehr[1:]:
        ele2 = d_e.split(",")
        clinic_num_d_e = ele2[0]
        if clinic_num_d_e in clinic_num_d_i_set:
            reserved_lines.append(d_e)

    fp = open(match_table_fn, 'a')
    for i in reserved_lines:
        fp.write(i)
    fp.close()
##################################################################################









