import pandas as pd
import os

out_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction_results/"
f1_fn = "/Jun_anonymized_dir/OvaryCancer/StromaReaction_results/Area-RGB_hist.csv"
f2_fn = "/Jun_anonymized_dir/OvaryCancer/StromaReaction_results/batch1_metadata.tsv"

f1_lines = open(f1_fn, 'r').readlines()
f2_lines = open(f2_fn, 'r').readlines()
f1_case_num_idx = 0
f2_case_num_idx = 10

print(f1_lines[0].split(",")[f1_case_num_idx])
print(f2_lines[0].split("\t")[f2_case_num_idx])

assert len(f1_lines)==len(f2_lines), "wrong file"
wrt_str = ""
for i in range(len(f1_lines)):
    f1_l = f1_lines[i].split(",")
    f2_l = f2_lines[i].split("\t")
    if i == 0:
        wrt_str += "\t".join(f2_l[0:-1] + f1_l[1:])
    elif f1_l[f1_case_num_idx] == f2_l[f2_case_num_idx]:
        wrt_str += "\t".join(f2_l[0:-1] + f1_l[1:])
    else:
        print(i+1)
        print(f1_l[f1_case_num_idx], f2_l[f2_case_num_idx])
        raise Exception("ID doesn't match error")

out_fn = os.path.join(out_dir, "BATCH_1_DATA.tsv")
fp = open(out_fn, 'w')
fp.write(wrt_str)
fp.close()

print("Done")


