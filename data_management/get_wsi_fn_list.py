import os


# ext = ".svs"
# dir_1 = "\\\\anonymized_dir\\POLE_WSI\\req30984"
# dir_2 = "\\\\anonymized_dir\\POLE_WSI\\req30990"
# dir_3 = "\\\\anonymized_dir\\POLE_WSI"
# dir_4 = "\\\\anonymized_dir\\WSIs"
#
# wsi_fn_list1 = sorted(os.listdir(dir_1))
# wsi_fn_list2 = sorted(os.listdir(dir_2))
# wsi_fn_list3 = sorted(os.listdir(dir_3))
# wsi_fn_list4 = sorted(os.listdir(dir_4))
#
# SBOT_train = []
# HGSOC_train = []
#
#
# def compare_case_list(case_list_src, case_list_dst):
#     ids = [f for f in case_list_src if f not in case_list_dst]
#     return ids
#
# case_list = compare_case_list(score_dir, output_dir)


ext = ".svs"
wsi_dir = "\\\\anonymized_dir\\WSIs\\AI_analysis_on_stromal_reactions"
# wsi_dir = "smb://mfad/researchmn/HCPR/HCPR-GYNECOLOGICALTUMORMICROENVIRONMENT/WSIs/AI_analysis_on_stromal_reactions"
wsi_fn_list = sorted(os.listdir(wsi_dir))

wrt_str = '['
for i in wsi_fn_list:
    if os.path.splitext(i)[1] == ext:
        wrt_str += '"' + i[0:-len(ext)] + '",'
wrt_str = wrt_str[0:-1]
wrt_str += ']'
print(wrt_str)







