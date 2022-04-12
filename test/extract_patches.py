from wsitools.patch_extraction.patch_extractor import ExtractorParameters, PatchExtractor
from wsitools.tissue_detection.tissue_detector import TissueDetector  # import dependent packages
import os
import time
import argparse
import sys
sys.path.insert(1, "./utils")
import load_configuration

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config",
                    dest='conf_fn',
                    required=True,
                    help="file name for configuration")
args = parser.parse_args()

conf = load_configuration.PatchExtractionConfig(args.conf_fn)
log_dir = conf.log_dir
output_root = conf.output_dir
WSI_root = conf.WSI_dir
wsi_ext = conf.wsi_ext
patch_size = conf.patch_size
stride = conf.stride
rescale_to = conf.rescale_to

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if conf.case_list.lower() == 'all':
    testing_cases = []
    for file in os.listdir(WSI_root):
        if file.endswith(wsi_ext):
            testing_cases.append(file)
else:
    testing_cases = eval(conf.case_list)

tissue_detector = TissueDetector("LAB_Threshold", threshold=85)
parameters = ExtractorParameters(output_root, log_dir=log_dir, save_format='.jpg', patch_size=patch_size, stride=stride,
                                 sample_cnt=-1, extract_layer=0, patch_filter_by_area=0.5, patch_rescale_to=rescale_to)
patch_extractor = PatchExtractor(tissue_detector, parameters, feature_map=None, annotations=None)

for case in testing_cases:
    wsi_fn = os.path.join(WSI_root, case + wsi_ext)
    if os.path.exists(wsi_fn):
        print("Extracting from %s" % case + wsi_ext)
        start = time.time()

        patch_num = patch_extractor.extract(wsi_fn)
        output_dir = os.path.join(output_root, case)
        print("%d Patches have been save to %s" % (patch_num, output_dir))
        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Time elapsed:{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    else:
        print("Can't locate WSI for patch extraction for case: %s" % case)
