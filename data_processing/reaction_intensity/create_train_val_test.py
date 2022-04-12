import random
import os
from PIL import Image

header = "Fibrosis,Cellularity,Orientation,img_fn\n"
# csv_file = "/anonymized_dir/Dataset/OvaryCancer/StromaReactionAnnotation_pro/all_samples.csv"
# shuffled_training_csv_file = "/anonymized_dir/Dataset/OvaryCancer/StromaReactionAnnotation_pro/training_five_cases.csv"
# shuffled_validation_csv_file = "/anonymized_dir/Dataset/OvaryCancer/StromaReactionAnnotation_pro/training_five_cases.csv"

csv_file = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/PatchSampling/all_samples.csv"
shuffled_training_csv_file = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/PatchSampling/training_five_cases.csv"
shuffled_validation_csv_file = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/PatchSampling/validation_five_cases.csv"
shuffled_testing_csv_file = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/PatchSampling/testing_five_cases.csv"

train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

lines = open(csv_file).readlines()[1:]
img_cnt = len(lines)
train_cnt = int(img_cnt * train_ratio)
val_cnt = int(img_cnt * val_ratio)

random.shuffle(lines)
training_lines = lines[0:train_cnt]
validate_lines = lines[train_cnt:train_cnt+val_cnt]
testing_lines = lines[train_cnt+val_cnt:]

fp = open(shuffled_training_csv_file, 'w')
fp.write(header)
fp.writelines(training_lines)
fp.close()

fp = open(shuffled_validation_csv_file, 'w')
fp.write(header)
fp.writelines(validate_lines)
fp.close()

fp = open(shuffled_testing_csv_file, 'w')
fp.write(header)
fp.writelines(testing_lines)
fp.close()

