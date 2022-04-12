from tensorflow.keras.applications import VGG16
import os
import glob
from PIL import Image
import numpy as np
import openslide
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import jaccard_score, roc_curve, precision_recall_fscore_support, average_precision_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn
import matplotlib.pyplot as plt

def plot_confusion_matrix(data, labels, title, output_filename):
    """Plot confusion matrix using heatmap.

    Args:
        data (list of list): List of lists with confusion matrix data.
        labels (list): Labels which will be plotted across x and y axis.
        output_filename (str): Path to output file.

    """
    seaborn.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))

    plt.title(title)

    seaborn.set(font_scale=1.4)
    ax = seaborn.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'})

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set(ylabel="True Label", xlabel="Predicted Label")

    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == '__main__':
    img_rows = 256
    img_cols = 256
    img_channel = 3
    img_rescale_rate = 2

    thumbnail_downsample = 128
    cmap = plt.get_cmap("jet")
    IMG_SHAPE = (img_rows, img_cols, img_channel)
    num_classes = 3

    all_class_list = ["Fibrosis", "Cellularity", "Orientation"]
    labels = ["0", "1", "2"]
    # all_model_list = ["Fibrosis_25-0.1366.hdf5", "Cellularity_18-0.1784.hdf5", "Orientation_27-0.2193.hdf5"]
    # all_model_list = ["Fibrosis_10-0.1976.hdf5", "Cellularity_16-0.0911.hdf5", "Orientation_06-0.1932.hdf5"]
    all_model_list = ["Fibrosis_27-0.2070.hdf5", "Cellularity_26-0.1913.hdf5", "Orientation_35-0.1652.hdf5"]

    # Load our data
    shuffled_testing_csv_file = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/PatchSampling/testing_five_cases.csv"
    WSI_root = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/WSIs"
    output_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/VGG16_Classification"
    eval_out_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/VGG16_Classification_eval"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(eval_out_dir):
        os.makedirs(eval_out_dir)

    test_df = pd.read_csv(shuffled_testing_csv_file, header=0)
    f_list = pd.Series.get(test_df, "img_fn").tolist()

    # Load our model
    model_weights_path = '/Jun_anonymized_dir/OvaryCancer/StromaReaction/model'
    VGG16_MODEL = VGG16(include_top=True, weights=None, input_tensor=None, input_shape=IMG_SHAPE,
                        pooling=None, classes=num_classes, classifier_activation='softmax')
    print(VGG16_MODEL.summary())

    for idx, m in enumerate(all_class_list):
        print("Evaluating %s on testing data" % m)
        ground_truth_score = pd.Series.get(test_df, m).tolist()

        model_ckpt = os.path.join(model_weights_path, m, all_model_list[idx])
        VGG16_MODEL.load_weights(model_ckpt)
        # cnt = 100

        eval_out_csv = os.path.join(output_dir, "eval_testing_dataset_" + m + ".csv")
        csv_fp = open(eval_out_csv, 'w')
        eval_results = "location_x,location_y,width,height,score0,score1,score2\n"

        predicted_score_list = []
        predicted_score_int_list = []
        location_list = []
        test_x = np.empty((1, img_rows, img_cols, img_channel), dtype=np.float32)
        for f in f_list:
            test_x[0, :, :, :] = np.array(Image.open(f, 'r')).astype(np.float32) / 255.0
            out = VGG16_MODEL.predict(test_x)
            # get score for heat-map
            # TODO: calculate score from three probability
            max_val = max(out[0])
            max_idx = list(out[0]).index(max_val)

            predicted_score_list.append(out[0])
            predicted_score_int_list.append(max_idx)
            # score = 0
            # for val_idx, val in enumerate(out[0]):
            #     score += val_idx * val
            # score_list.append(score)

            # get locations from image file name
            img_fn = os.path.split(f)[1]
            ele = img_fn.split("_")
            loc_x = int(ele[-2])
            loc_y = int(ele[-1][0:-4])
            location_list.append([loc_x, loc_y])
            eval_results += str(loc_x) + "," + str(loc_y) + "," + str(img_cols * img_rescale_rate) \
                            + "," + str(img_rows * img_rescale_rate) + "," + str(out[0][0]) + "," + str(out[0][1]) \
                            + "," + str(out[0][2]) + "\n"
            # cnt -= 1
            # if cnt == 0:
            #     break
        csv_fp.write(eval_results)
        csv_fp.close()

        precision, recall, f, s = precision_recall_fscore_support(ground_truth_score, predicted_score_list, average="micro")
        # precision, recall, f, s = precision_recall_fscore_support(ground_truth_score, predicted_score_list, average="macro")
        print("%s: \n \t\tprecision:%.4f, recall:%.4f, F1_score:%.4f" % (m, precision, recall, f))


        cm = confusion_matrix(ground_truth_score, predicted_score_list, normalize='true')
        title = all_class_list[idx] + " Normalized Confusion Matrix"
        output_filename = os.path.join(eval_out_dir, all_class_list[idx] + "_normalized.jpg")
        plot_confusion_matrix(cm, labels, title, output_filename)
        print(cm)

