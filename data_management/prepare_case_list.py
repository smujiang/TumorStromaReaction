import os
import sys

def write_list_to_csv(save_to_fn, wsi_fn_list, ext=None):
    fp = open(save_to_fn, 'w')
    wrt_str = '['
    for i in wsi_fn_list:
        if ext is not None:
            if os.path.splitext(i)[1] == ext:
                wrt_str += '"' + i[0:-len(ext)] + '",'
        else:
            wrt_str += '"' + i + '",'
    wrt_str = wrt_str[0:-1]
    wrt_str += ']'
    fp.write(wrt_str)


def compare_case_list(case_list_src, case_list_dst):
    ids = [f for f in case_list_src if f not in case_list_dst]
    return ids

def get_wsi_local_list(wsi_dir, fn):
    wsi_fn_list = sorted(os.listdir(wsi_dir))
    write_list_to_csv(fn, wsi_fn_list, ext=".svs")

def get_remote_processed_wsi_list(wsi_dir, fn):
    wsi_fn_list = sorted(os.listdir(wsi_dir))
    write_list_to_csv(fn, wsi_fn_list)


if __name__ == "__main__":
    print(sys.platform)
    fn = "local_all_wsi_list.csv"
    fn_remote = "remote_processed_wsi_list.csv"
    if os.path.exists(fn) and os.path.exists(fn_remote):
        case_list_src = eval(open(fn, 'r').readlines()[0])
        case_list_dst = eval(open(fn_remote, 'r').readlines()[0])
        comp_list = compare_case_list(case_list_src, case_list_dst)

        save_to = "comp.csv"
        write_list_to_csv(save_to, comp_list)
    else:
        if "linux" not in sys.platform.lower():

            wsi_dir = "\\\\anonymized_dir\\WSIs\\AI_analysis_on_stromal_reactions"
            wsi_fn_list = sorted(os.listdir(wsi_dir))
            write_list_to_csv(fn, wsi_fn_list, ext=".svs")

            #TODO: upload to
        else:

            wsi_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/reaction_prediction_NC"
            wsi_fn_list = sorted(os.listdir(wsi_dir))
            write_list_to_csv(fn_remote, wsi_fn_list)


















