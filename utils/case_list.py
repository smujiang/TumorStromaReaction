import os

def Diff(li1, li2):
    return sorted(list(list(set(li1)-set(li2)) + list(set(li2)-set(li1))))

def diff_list(li1, li2):
    return sorted(list(set(li1)-set(li2)))

def CaseID_in_two_folders(folder1, folder2):
    case_id_list1 = sorted(os.listdir(folder1))
    case_id_list2 = sorted(os.listdir(folder2))
    return Diff(case_id_list1, case_id_list2)


if __name__ == "__main__":
    # phase2_wsi_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/WSIs"
    # org_case_ids = os.listdir(phase2_wsi_dir)
    # for c in org_case_ids:
    #     if "wo" not in c:
    #         org_case_ids.remove(c)
    #
    # patch_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/patches"
    # patch_case_ids = os.listdir(patch_dir)
    # for c in patch_case_ids:
    #     if "wo" not in c:
    #         patch_case_ids.remove(c)
    #
    # print("Unprocessed (patch extraction) cases:")
    # print(Diff(org_case_ids, patch_case_ids))
    # print(diff_list(org_case_ids, patch_case_ids))

    # folder2 = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/segmentation"
    # CaseID_in_two_folders(folder1, folder2)


    patch_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/patches"
    case_dirs = sorted(os.listdir(patch_dir))

    wrt_str = '['
    for i in case_dirs:
        if "." not in i:
            wrt_str += '"' + i + '",'
    wrt_str = wrt_str[0:-1]
    wrt_str += ']'
    print(wrt_str)




