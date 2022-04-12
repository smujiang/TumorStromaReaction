import os
import numpy as np
from PIL import Image
from sklearn.metrics import jaccard_score, roc_curve, precision_recall_fscore_support, average_precision_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

eval_log_fn = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/TumorStroma_MaskRCNN/ROI_Masks_out_train_val_test_split/test_out/eval_log_el.csv"

df = pd.read_csv(eval_log_fn, sep=',')
df_sel = df.iloc[:, 1:4]
labels = df_sel.head()
all_metrics = np.array(df_sel)

fig, ax = plt.subplots()
ax.set_title('Segmentation Evaluation Metrics', fontsize=14)
ax.boxplot(all_metrics, notch=True)
ax.set_xticklabels(labels, fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.show()

print("Averaged Jaccard similarity: %.4f" % np.mean(all_metrics[:,0]))
print("Averaged average_precision_score scores: %.4f" % np.mean(all_metrics[:,1]))
print("Averaged F1 scores: %.4f" % np.mean(all_metrics[:,2]))

from scipy import stats

# plot fit
xx = np.arange(0.0, 1.05, 0.1)
yy = xx

fig = plt.figure()
ax = fig.gca()
x = all_metrics[:, 0]
y = all_metrics[:, 1]
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
ax.scatter(x, y, marker='o', edgecolors='k', c='w', s=18)
y_fit = xx*slope + intercept
ax.plot(xx, y_fit, 'r--', linewidth=3)
ax.plot(xx, yy, 'k--', linewidth=2)
ax.set_xlabel('IoU', fontsize=16)
ax.set_ylabel('AP', fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
str_r = r'R^2=' + "{:.3f}".format(r_value**2)
ax.text(0.5, 0.4, str_r, fontsize=14, color='r')
# str_p = r'p=' + "{:.3e}".format(p_value)
# ax.text(0.5, 0.35, str_p, fontsize=14, color='r')
str_s = r'Slop=' + "{:.3f}".format(slope)
ax.text(0.5, 0.35, str_s, fontsize=14, color='r')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.grid(True)
plt.show()


fig = plt.figure()
ax = fig.gca()
x = all_metrics[:, 0]
y = all_metrics[:, 2]
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
ax.scatter(x, y, marker='o', edgecolors='k', c='w', s=18)
y_fit = xx*slope + intercept
ax.plot(xx, y_fit, 'r--', linewidth=3)
ax.plot(xx, yy, 'k--', linewidth=2)
ax.set_xlabel('IoU', fontsize=16)
ax.set_ylabel('DSC', fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
str_r = r'R^2=' + "{:.3f}".format(r_value**2)
ax.text(0.5, 0.4, str_r, fontsize=14, color='r')
# str_p = r'p=' + "{:.3e}".format(p_value)
# ax.text(0.5, 0.35, str_p, fontsize=14, color='r')
str_s = r'Slop=' + "{:.3f}".format(slope)
ax.text(0.5, 0.35, str_s, fontsize=14, color='r')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.grid(True)
plt.show()

fig = plt.figure()
ax = fig.gca()
x = all_metrics[:, 1]
y = all_metrics[:, 2]
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
ax.scatter(x, y, marker='o', edgecolors='k', c='w', s=18)
y_fit = xx*slope + intercept
ax.plot(xx, y_fit, 'r--', linewidth=3)
ax.plot(xx, yy, 'k--', linewidth=2)
ax.set_xlabel('AP', fontsize=16)
ax.set_ylabel('DSC', fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
str_r = r'R^2=' + "{:.3f}".format(r_value**2)
ax.text(0.6, 0.4, str_r, fontsize=14, color='r')
# str_p = r'p=' + "{:.3e}".format(p_value)
# ax.text(0.5, 0.35, str_p, fontsize=14, color='r')
str_s = r'Slop=' + "{:.3f}".format(slope)
ax.text(0.6, 0.35, str_s, fontsize=14, color='r')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.grid(True)
plt.show()



print("Done")

