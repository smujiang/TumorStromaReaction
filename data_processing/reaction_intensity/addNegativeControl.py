import os
import random
from PIL import Image


original_training = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/PatchSampling/training_five_cases.csv"
original_validation = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/PatchSampling/validation_five_cases.csv"

train_lines = open(original_training).readlines()
training_cnt = len(train_lines)
print(training_cnt)

val_lines = open(original_validation).readlines()
val_cnt = len(val_lines)
print(val_cnt)

SBOT_dir = "/Jun_anonymized_dir/OvaryCancer/auto_enc_patches_256"
SBOT_cases_train = ["OCMC-016", "OCMC-018", "OCMC-020", "OCMC-022", "OCMC-024", "OCMC-026"]
# SBOT_cases_train = ["OCMC-016"]

wrt_str_train_list = []
wrt_str_val_list = []
for case in SBOT_cases_train:
    SBOT_imgs = os.listdir(os.path.join(SBOT_dir, case))
    # randomly pick 1405 from each case
    imgs = random.sample(SBOT_imgs, 1405)
    for i in imgs[0:1000]:
        wrt_str_train_list.append("0,0,0," + os.path.join(SBOT_dir, case, i) + "\n")
        # img_1 = Image.open(os.path.join(SBOT_dir, case, i))
        #
        # img_2 = Image.open(val_lines[1].split(",")[3][:-1])
        #
        # print(img_1.width)
        # print(img_2.width)

    for i in imgs[1000:]:
        wrt_str_val_list.append("0,0,0," + os.path.join(SBOT_dir, case, i) + "\n")


train_lines = train_lines + wrt_str_train_list
print(len(train_lines))

new_train_csv = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/PatchSampling/training_with_SBOT_cases.csv"
fp = open(new_train_csv, 'w')
fp.write(train_lines[0])
a = train_lines[1:]
random.Random().shuffle(a)
for l in a:
    fp.write(l)
fp.close()

val_lines = val_lines + wrt_str_val_list
print(len(val_lines))

new_val_csv = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/PatchSampling/validation_with_SBOT_cases.csv"
fp = open(new_val_csv, 'w')
fp.write(val_lines[0])
a = val_lines[1:]
random.Random().shuffle(a)
for l in a:
    fp.write(l)
fp.close()









