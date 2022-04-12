import os
import pandas as pd

fn = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/result_analysis/All_TSR_patch_statistics.csv"
df = pd.read_csv(fn, sep=',')

quan_low = df.dropna().quantile(0.05, numeric_only=True)
quan_up = df.dropna().quantile(0.95, numeric_only=True)

# metrics = ["Fibrosis"]
metrics = ["Fibrosis", "cellularity", "orientation"]
case_id_list = []
for m in metrics:
    c1 = "test_" + m + "_0_cnt"
    c2 = "test_" + m + "_2_cnt"
    df1_low = df[df[c1] < quan_low[c1]]
    df2_low = df[df[c2] < quan_low[c2]]
    df1_up = df[df[c1] > quan_up[c1]]
    df2_up = df[df[c2] > quan_up[c2]]

    case_id_list += list(df1_low["case_id"])
    case_id_list += list(df2_low["case_id"])
    case_id_list += list(df1_up["case_id"])
    case_id_list += list(df2_up["case_id"])

case_id_list = set(case_id_list)

str_wrt = "["
for c in case_id_list:
    str_wrt += "\"" + c + "\","
str_wrt = str_wrt[:-1]
str_wrt += "]"
print(str_wrt)

print(case_id_list)

# get high and low 10% quantile

# sort score

print("DONE")

















