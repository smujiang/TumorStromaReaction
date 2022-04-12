import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


All_TSR_patch_statistics_fn = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/result_analysis/All_TSR_patch_statistics.csv"

df = pd.read_csv(All_TSR_patch_statistics_fn, sep=',')
SBOT_cases = df.loc[df["case_id"].str.contains("OCMC", case=False)]

SBOT_TSR_r = np.array(SBOT_cases.iloc[:, 14:23])

HGSOC_cases = df.loc[~df["case_id"].str.contains("OCMC", case=False)]

HGSOC_TSR_r = np.array(HGSOC_cases.iloc[:, 14:23])


#####################################################################

#####################################################################
fib_values = np.array([])
fib_content = []
fib_diff = []

for s in range(0, 3):
    fib_values = np.concatenate([fib_values, HGSOC_TSR_r[:, s]])
    fib_content = np.concatenate([fib_content, [str(s)]*len(HGSOC_TSR_r[:, s])])
    fib_diff = np.concatenate([fib_diff, ["HGSOC"]*len(HGSOC_TSR_r[:, s])])
    fib_values = np.concatenate([fib_values, SBOT_TSR_r[:, s]])
    fib_content = np.concatenate([fib_content, [str(s)]*len(SBOT_TSR_r[:, s])])
    fib_diff = np.concatenate([fib_diff, ["SBOT"]*len(SBOT_TSR_r[:, s])])

fib_vio = pd.DataFrame({"metrics": fib_values,
                         "content": fib_content,
                         "diff": fib_diff})
fig = plt.figure(0, figsize=(5, 5))
ax = sns.violinplot(x='content', y='metrics', hue='diff', data=fib_vio, split=True)
ax.set_title("Fibrosis", fontsize=14)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:], loc='top right')  # hide legend title
ax.xaxis.label.set_visible(False)  # hide x label
ax.yaxis.label.set_visible(False)  # hide y label
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.show()
# sv_nm = os.path.join(sv_dir, "vio_plot_fib.png")
# fig.savefig(sv_nm, bbox_inches='tight')
# plt.close(fig)

#####################################################################

#####################################################################
cel_values = np.array([])
cel_content = []
cel_diff = []

for s in range(3, 6):
    cel_values = np.concatenate([cel_values, HGSOC_TSR_r[:, s]])
    cel_content = np.concatenate([cel_content, [str(s-3)]*len(HGSOC_TSR_r[:, s])])
    cel_diff = np.concatenate([cel_diff, ["HGSOC"]*len(HGSOC_TSR_r[:, s])])
    cel_values = np.concatenate([cel_values, SBOT_TSR_r[:, s]])
    cel_content = np.concatenate([cel_content, [str(s-3)]*len(SBOT_TSR_r[:, s])])
    cel_diff = np.concatenate([cel_diff, ["SBOT"]*len(SBOT_TSR_r[:, s])])

cel_vio = pd.DataFrame({"metrics": cel_values,
                         "content": cel_content,
                         "diff": cel_diff})
fig = plt.figure(0, figsize=(5, 5))
ax = sns.violinplot(x='content', y='metrics', hue='diff', data=cel_vio, split=True)
ax.set_title("Cellularity", fontsize=14)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:], loc='upper center')  # hide legend title
ax.xaxis.label.set_visible(False)  # hide x label
ax.yaxis.label.set_visible(False)  # hide y label
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.show()

# sv_nm = os.path.join(sv_dir, "vio_plot_cel.png")
# fig.savefig(sv_nm, bbox_inches='tight')
# plt.close(fig)


#####################################################################

#####################################################################
ori_values = np.array([])
ori_content = []
ori_diff = []

for s in range(6, 9):
    ori_values = np.concatenate([ori_values, HGSOC_TSR_r[:, s]])
    ori_content = np.concatenate([ori_content, [str(s-6)]*len(HGSOC_TSR_r[:, s])])
    ori_diff = np.concatenate([ori_diff, ["HGSOC"]*len(HGSOC_TSR_r[:, s-6])])
    ori_values = np.concatenate([ori_values, SBOT_TSR_r[:, s]])
    ori_content = np.concatenate([ori_content, [str(s-6)]*len(SBOT_TSR_r[:, s])])
    ori_diff = np.concatenate([ori_diff, ["SBOT"]*len(SBOT_TSR_r[:, s])])

ori_vio = pd.DataFrame({"metrics": ori_values,
                         "content": ori_content,
                         "diff": ori_diff})
fig = plt.figure(0, figsize=(5, 5))
ax = sns.violinplot(x='content', y='metrics', hue='diff', data=ori_vio, split=True)
ax.set_title("Orientation", fontsize=14)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:], loc='top right')  # hide legend title
ax.xaxis.label.set_visible(False)  # hide x label
ax.yaxis.label.set_visible(False)  # hide y label
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.show()
# sv_nm = os.path.join(sv_dir, "vio_plot_ori.png")
# fig.savefig(sv_nm, bbox_inches='tight')
# plt.close(fig)




print("Done")



