#!/usr/bin/bash


config_file="./test_config1.ini"

# 1. extract patches from WSI
python ./test/extract_patches.py --config=$config_file

# 2. tumor stroma segmentation
python ./test/segmenation.py --config=$config_file

# 3. get thumbnail level segmentation results and then get tumor-stroma entangling area
python ./test/get_TSR_entangling.py

# 4. stroma reaction intensity estimation
python ./test/reaction_prediction.py --config=$config_file

# 5. get top 5 ROIs
python ./eval/cell_level_feature/get_rois.py

# 6. get cell features
# create QuPath project, and import images into it
# currently, cell detection and feature extraction rely on QuPath, need to install QuPath (Command line) first.
QuPath-0.2.3 script -s ./eval/cell_level_feature/create_projects.groovy

export proj_path=/Jun_anonymized_dir/OvaryCancer/StromaReaction/QuPathProj
for i in $(ls ${proj_path});
  do
# load ROIs into QuPath project, project folders are defined in last step
    QuPath-0.2.3 script -s -p ${proj_path}/${i}/${i}.qpproj ./eval/cell_level_feature/load_ROI2QuPath.groovy;
# detect cells and extract cell features
    QuPath-0.2.3 script -s -p ${proj_path}/${i}/${i}.qpproj ./eval/cell_level_feature/getROICellFeatures.groovy;
done
# load ROIs into QuPath project, project folders are defined in last step
#QuPath-0.2.3 script -s -p /project/WSI_Profiling/req30984/job_0.qpproj ./eval/cell_level_feature/load_ROI2QuPath.groovy
## detect cells and extract cell features
#QuPath-0.2.3 script -s -p /project/WSI_Profiling/req30984/job_0.qpproj ./eval/cell_level_feature/getROICellFeatures.groovy

# 7. classify cells
python ./eval/cell_level_feature/cell_prediction.py

# 8. cell density
python ./eval/cell_level_feature/getCellDensity.py

# 9. summarize all data
python ./eval/summarization/all_case_proportion_ratio_plus.py

# 10. write case level metadata
python ./data_management/write_case_level_meta_data.py






