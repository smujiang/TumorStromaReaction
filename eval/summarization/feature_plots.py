import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

metadata_file = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/result_analysis/metadata/all_case_metadata.csv"
output_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/result_analysis/meata_data_out"

lines = open(metadata_file, 'r').readlines()
header = lines[0]
data_lines = lines[1:]

case_id = []
header_ele = header.strip().split(",")
cell_feature_names = header_ele[14:14+41]
density_feature_names = header_ele[14+41*4:14+41*4+3]
tsr_feature_names = header_ele[14+41*4+3*2:]
print(cell_feature_names)
print(density_feature_names)
print(tsr_feature_names)

num_data_start_column = 14

tumor_mean_feature_array = []
tumor_std_feature_array = []
stroma_mean_feature_array = []
stroma_std_feature_array = []

cell_density_mean_array = []
cell_density_std_array = []
tsr_score_features = []

all_data_array = []
all_data_headers = header_ele[14:]
for dl in data_lines:
    str_ele = dl.strip().split(",")

    case_id.append(str_ele[6])

    ele = []
    data_ele = []
    for idx, e in enumerate(str_ele):
        if idx >= num_data_start_column:
            ele.append(float(e))
            data_ele.append(float(e))
        else:
            ele.append(e)
    all_data_array.append(data_ele)
    # print("%s case" % ele[13])
    start = 14
    end = start + 41
    tumor_mean_feature_array.append(ele[start:end])
    start = end
    end = start + 41
    tumor_std_feature_array.append(ele[start:end])
    start = end
    end = start + 41
    stroma_mean_feature_array.append(ele[start:end])
    start = end
    end = start + 41
    stroma_std_feature_array.append(ele[start:end])

    start = end + 1
    end = start + 2
    cell_density_mean_array.append(ele[start:end])
    start = end + 1
    end = start + 2
    cell_density_std_array.append(ele[start:end])

    start = end
    end = start + 9
    tsr_score_features.append(ele[start:end])


all_data_correlations = np.corrcoef(all_data_array)
import seaborn as sns
g1 = sns.clustermap(all_data_correlations, method="complete", cmap='RdBu', annot=False,
               annot_kws={"size": 7}, vmin=-1, vmax=1, figsize=(20,20),xticklabels=True, yticklabels=True)

save_to = os.path.join(output_dir, "all_clustermap.png")
g1.savefig(save_to)



# HGSOC and SBOT statistics
# HGSOC_rows = [[0, 15], [30, len(all_data_array)]]
# SBOT_rows = [[15, 30]]

HGSOC_rows = [[0, 129], [144, len(all_data_array)]]
SBOT_rows = [[129, 144]]

all_data_array = np.array(all_data_array)
HGSOC_features = []
HGSOC_case_id = []
for s_e in HGSOC_rows:
    HGSOC_features.append(all_data_array[s_e[0]:s_e[1], :])
    HGSOC_case_id += case_id[s_e[0]:s_e[1]]
HGSOC_feature_array = np.concatenate(HGSOC_features, axis=0)

SBOT_features = []
SBOT_case_id = []
for s_e in SBOT_rows:
    SBOT_features.append(all_data_array[s_e[0]:s_e[1], :])
    SBOT_case_id += case_id[s_e[0]:s_e[1]]
SBOT_feature_array = np.concatenate(SBOT_features, axis=0)
for idx, data_name in enumerate(all_data_headers):
    HGSOC_data_column = HGSOC_feature_array[:, idx]
    SBOT_data_column = SBOT_feature_array[:, idx]
    max_ = max(np.concatenate([HGSOC_data_column, SBOT_data_column]).flatten())
    min_ = min(np.concatenate([HGSOC_data_column, SBOT_data_column]).flatten())
    plt.title(data_name)
    fig, axs = plt.subplots(2)
    fig.suptitle(data_name)
    axs[0].hist(HGSOC_data_column, bins=12)
    axs[0].set_xlim([min_, max_])
    axs[0].set_ylabel("HGSOC metric")
    axs[1].hist(SBOT_data_column, bins=12)
    axs[1].set_xlim([min_, max_])
    axs[1].set_ylabel("SBOT metric")
    save_to = os.path.join(output_dir, "statistics_" + data_name.replace("/",  "-") + ".png")
    plt.savefig(save_to)
    print("save %s" % data_name)


SBOT_correlations = np.corrcoef(SBOT_feature_array)
HGSOC_correlations = np.corrcoef(HGSOC_feature_array)
import seaborn as sns
g1 = sns.clustermap(SBOT_correlations, method="complete", cmap='RdBu', annot=True,
               annot_kws={"size": 7}, vmin=-1, vmax=1, figsize=(15,12))

g2 = sns.clustermap(HGSOC_correlations, method="complete", cmap='RdBu', annot=True,
               annot_kws={"size": 7}, vmin=-1, vmax=1, figsize=(15,12))
save_to = os.path.join(output_dir, "SBOT_clustermap.png")
g1.savefig(save_to)
save_to = os.path.join(output_dir, "HGSOC_clustermap.png")
g2.savefig(save_to)

# cell features
plt.figure(figsize=[14, 22], dpi=300)
plt.imshow(np.array(tumor_mean_feature_array), cmap='Blues')
plt.title("tumor feature mean")
ax = plt.gca()
ax.set_xticks(range(0, len(cell_feature_names)))
ax.set_yticks(range(0, len(case_id)))
ax.set_xticklabels(cell_feature_names)
ax.set_yticklabels(case_id)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(6)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(6)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
save_to = os.path.join(output_dir, "tumor_mean_feature_array.png")
plt.savefig(save_to)


plt.figure(figsize=[14, 22], dpi=300)
plt.imshow(np.array(tumor_std_feature_array), cmap='Blues')
plt.title("tumor feature std")
ax = plt.gca()
ax.set_xticks(range(0, len(cell_feature_names)))
ax.set_yticks(range(0, len(case_id)))
ax.set_xticklabels(cell_feature_names)
ax.set_yticklabels(case_id)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(6)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(6)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
save_to = os.path.join(output_dir, "tumor_std_feature_array.png")
plt.savefig(save_to)

plt.figure(figsize=[14, 22], dpi=300)
plt.imshow(np.array(stroma_mean_feature_array), cmap='Blues')
plt.title("stroma feature mean")
ax = plt.gca()
ax.set_xticks(range(0, len(cell_feature_names)))
ax.set_yticks(range(0, len(case_id)))
ax.set_xticklabels(cell_feature_names)
ax.set_yticklabels(case_id)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(6)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(6)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
save_to = os.path.join(output_dir, "stroma_mean_feature_array.png")
plt.savefig(save_to)

plt.figure(figsize=[14, 22], dpi=300)
plt.imshow(np.array(stroma_std_feature_array), cmap='Blues')
plt.title("stroma feature std")
ax = plt.gca()
ax.set_xticks(range(0, len(cell_feature_names)))
ax.set_yticks(range(0, len(case_id)))
ax.set_xticklabels(cell_feature_names)
ax.set_yticklabels(case_id)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(6)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(6)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
save_to = os.path.join(output_dir, "stroma_std_feature_array.png")
plt.savefig(save_to)


# cell density feature plot
plt.figure(figsize=[6, 12], dpi=300)
plt.imshow(np.array(cell_density_mean_array), cmap='Blues')
plt.title("cell density feature mean")
ax = plt.gca()
ax.set_xticks(range(0, len(density_feature_names[1:])))
ax.set_yticks(range(0, len(case_id)))
ax.set_xticklabels(density_feature_names[1:])
ax.set_yticklabels(case_id)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(6)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(6)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
save_to = os.path.join(output_dir, "cell_density_mean_array.png")
plt.savefig(save_to)

plt.figure(figsize=[6, 12], dpi=300)
plt.imshow(np.array(cell_density_std_array), cmap='Blues')
plt.title("cell density feature std")
ax = plt.gca()
ax.set_xticks(range(0, len(density_feature_names[1:])))
ax.set_yticks(range(0, len(case_id)))
ax.set_xticklabels(density_feature_names[1:])
ax.set_yticklabels(case_id)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(6)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(6)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
save_to = os.path.join(output_dir, "cell_density_std_array.png")
plt.savefig(save_to)

# TSR score plot
plt.figure(figsize=[6, 12], dpi=300)
plt.imshow(np.array(tsr_score_features), cmap='Blues')
plt.title("TSR distribution")
ax = plt.gca()
ax.set_xticks(range(0, len(tsr_feature_names)))
ax.set_yticks(range(0, len(case_id)))
ax.set_xticklabels(tsr_feature_names)
ax.set_yticklabels(case_id)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(6)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(6)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
save_to = os.path.join(output_dir, "tsr_score_features.png")
plt.savefig(save_to)







