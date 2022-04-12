import sys,os


wsi_dir = "/Jun_anonymized_dir/OvaryCancer/WSIs"
seg_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/segmentation"
reaction_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/reaction_prediction_NC"

reaction_score_names = ["Fibrosis", "Cellularity", "Orientation"]

wsi_list = os.listdir(wsi_dir)
ext = ".svs"

case_to_be_finished = set()
print("Check incomplete reaction score estimation.")
for wsi_fn in wsi_list:
    wsi_case_fn = wsi_fn[0:-len(ext)]

    reaction_case_dir = os.path.join(reaction_dir, wsi_case_fn)
    # if os.path.exists(os.path.join(reaction_case_dir, "0001.png")):
    #     print("not_exist")
    if not os.path.exists(reaction_case_dir):
        print("Unprocessed cases:" + wsi_case_fn)
        case_to_be_finished.add(wsi_case_fn)
    else:
        case_reaction_dir_list = os.listdir(reaction_case_dir)
        for rsn in reaction_score_names:
            if rsn not in case_reaction_dir_list:
                print("Incomplete cases:" + wsi_case_fn)
                case_to_be_finished.add(wsi_case_fn)
print("Unprocessed cases: %d" % len(case_to_be_finished))
wrt_str = "["
for c in case_to_be_finished:
    wrt_str += "'" + c + "',"
wrt_str = wrt_str[0:-1]
wrt_str += "]"
print(wrt_str)

print("******************************************")

print("Check incomplete tissue segmentation.")
for wsi_fn in wsi_list:
    wsi_case_fn = wsi_fn[0:-len(ext)]

    seg_case_dir = os.path.join(seg_dir, wsi_case_fn)
    if not os.path.exists(seg_case_dir):
        print("Unprocessed cases:" + wsi_case_fn)




















