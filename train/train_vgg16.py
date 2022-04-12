import random
import os

header = "Fibrosis,Cellularity,Orientation,img_fn\n"
# csv_file = "/anonymized_dir/Dataset/OvaryCancer/StromaReactionAnnotation_pro/all_samples.csv"
# shuffled_training_csv_file = "/anonymized_dir/Dataset/OvaryCancer/StromaReactionAnnotation_pro/training_five_cases.csv"
# shuffled_validation_csv_file = "/anonymized_dir/Dataset/OvaryCancer/StromaReactionAnnotation_pro/training_five_cases.csv"

# csv_file = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/PatchSampling/all_samples.csv"
# shuffled_training_csv_file = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/PatchSampling/training_five_cases.csv"
# shuffled_validation_csv_file = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/PatchSampling/validation_five_cases.csv"

# train with SBOT cases as negative controls
shuffled_training_csv_file = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/PatchSampling/training_with_SBOT_cases.csv"
shuffled_validation_csv_file = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/PatchSampling/validation_with_SBOT_cases.csv"



training_cnt = len(open(shuffled_training_csv_file, 'r').readlines()) - 1
validate_cnt = len(open(shuffled_validation_csv_file, 'r').readlines()) - 1

'''
###################### create data generator directly from image folder############################
'''
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import time
import tensorflow.keras.backend as K
import pandas as pd
import tensorflow as tf
import numpy as np
# import staintools

IMG_SHAPE = (256, 256, 3)
num_classes = 3   # score from 1 to 3

all_label_list = ["Fibrosis", "Cellularity", "Orientation"]
names = header.strip().split(",")
train_df = pd.read_csv(shuffled_training_csv_file, header=0)
train_file_list = pd.Series.get(train_df, "img_fn").tolist()
# train_Fibrosis_scores_list = pd.Series.get(train_df, "Fibrosis").tolist()
# train_Cellularity_scores_list = pd.Series.get(train_df, "Cellularity").tolist()
# train_Orientation_scores_list = pd.Series.get(train_df, "Orientation").tolist()

val_df = pd.read_csv(shuffled_validation_csv_file, header=0)
val_file_list = pd.Series.get(val_df, "img_fn").tolist()
# val_Fibrosis_scores_list = pd.Series.get(val_df, "Fibrosis").tolist()
# val_Cellularity_scores_list = pd.Series.get(val_df, "Cellularity").tolist()
# val_Orientation_scores_list = pd.Series.get(val_df, "Orientation").tolist()

# augmentor = staintools.StainAugmentor(method='vahadane', sigma1=0.2, sigma2=0.2)

def decode_example_fd(filename, label):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32) / 255.
    image = tf.reshape(image, IMG_SHAPE)
    # data augmentation
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    label = tf.cast(tf.one_hot(label, num_classes), tf.int64)
    return image, label


def WSI_data_generator_fd(file_list, label_list, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((file_list, label_list))
    dataset = dataset.map(decode_example_fd, num_parallel_calls=5)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    return dataset


epochs = 50
patience = 2
bs = 32
trained_models_path = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/model"
log_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/log"

# VGG16_MODEL = VGG16(include_top=True, weights=None, input_tensor=None, input_shape=IMG_SHAPE,
#                     pooling=None, classes=num_classes, classifier_activation='softmax')
VGG16_MODEL = VGG16(include_top=True, weights=None, input_tensor=None, input_shape=IMG_SHAPE,
                    pooling=None, classes=num_classes)

for m in all_label_list:
    print("Training " + m + " predictive model")
    # Callbacks

    if not os.path.exists(os.path.join(log_dir, m)):
        os.makedirs(os.path.join(log_dir, m))
    latest_check_point = tf.train.latest_checkpoint(os.path.join(log_dir, m))
    if latest_check_point is not None:
        VGG16_MODEL.load_weights(latest_check_point)
    tensor_board = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(log_dir, m), histogram_freq=0, write_graph=True,
                                                  write_images=True)

    model_names = os.path.join(trained_models_path, m, m+'_{epoch:02d}-{val_loss:.4f}.hdf5')
    if not os.path.exists(os.path.join(trained_models_path, m)):
        os.makedirs(os.path.join(trained_models_path, m))
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)

    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)
    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

    VGG16_MODEL.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=1e-5), metrics=['accuracy'])

    train_scores_list = pd.Series.get(train_df, m).tolist()
    val_scores_list = pd.Series.get(val_df, m).tolist()

    train_ds = WSI_data_generator_fd(train_file_list, train_scores_list, batch_size=bs)
    val_ds = WSI_data_generator_fd(val_file_list, val_scores_list, batch_size=bs)

    # opt = tf.keras.optimizers.Adam(0.1)
    # iterator = iter(train_ds)
    # ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=VGG16_MODEL, iterator=iterator)
    # manager = tf.train.CheckpointManager(ckpt, os.path.join(trained_models_path, m), max_to_keep=3)
    # VGG16_MODEL.save_weights()

    t1 = time.time()
    history = VGG16_MODEL.fit(train_ds,
                              epochs=epochs,
                              steps_per_epoch=int(training_cnt / bs),
                              validation_steps=int(validate_cnt / bs),
                              validation_data=val_ds,
                              callbacks=callbacks)

    t2 = time.time() - t1
    K.clear_session()
