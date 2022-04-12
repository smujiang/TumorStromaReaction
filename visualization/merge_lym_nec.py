import os

lym_filePath = "/Jun_anonymized_dir/TIL/heat_map/_HE_lym.txt"
nec_filePath = "/Jun_anonymized_dir/TIL/heat_map/_HE_nec.txt"
output_file = "/Jun_anonymized_dir/TIL/heat_map/_HE_lym_nec.txt"

lym_lines = sorted(open(lym_filePath, 'r').readlines())
nec_lines = sorted(open(nec_filePath, 'r').readlines())

lym_dic = {}
for ll in lym_lines:
    ele_ll = ll.strip().split(" ")
    lym_dic[ele_ll[0] + "," + ele_ll[1]] = ele_ll[2]


nec_dic = {}
for ll in nec_lines:
    ele_ll = ll.strip().split(" ")
    nec_dic[ele_ll[0] + "," + ele_ll[1]] = ele_ll[2]

cnt = 0
wrt_str = ""
for k in lym_dic.keys():
    if k in nec_dic.keys():
        if float(lym_dic[k]) < 0.01 and float(nec_dic[k]) < 0.01:
            pass
        else:
            print(k + "," + lym_dic[k] + "," + nec_dic[k])
            wrt_str += k + "," + lym_dic[k] + "," + nec_dic[k] + "\n"
            cnt += 1
print("Total lin count %d" % cnt)

fp = open(output_file, 'w')
fp.write(wrt_str)
fp.close()
