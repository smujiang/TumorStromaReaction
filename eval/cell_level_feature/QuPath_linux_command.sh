#!/usr/bin/bash
# linux
# create projects
#/Jun_anonymized_dir/QuPath-0.2.3/bin/QuPath-0.2.3 script -s ./create_projects.groovy
/Jun_anonymized_dir/QuPath-0.2.3/bin/QuPath-0.2.3 script -s ./GetCellFeatures.groovy

# load ROIs to QuPath
#java -Djava.awt.headless=true -jar /Jun_anonymized_dir/QuPath-0.2.3/lib/app/qupath-0.2.3.jar script -s -p /Jun_anonymized_dir/OvaryCancer/StromaReaction/QuPathProj/WSIs_0/WSIs_0.qpproj /Jun_anonymized_dir/OvaryCancer/StromaReaction/eval/cell_level_feature/load_ROI2QuPath.groovy

#/Jun_anonymized_dir/QuPath-0.2.3/bin/QuPath-0.2.3 script -s -p /Jun_anonymized_dir/OvaryCancer/StromaReaction/QuPathProj/WSIs_0/WSIs_0.qpproj /Jun_anonymized_dir/OvaryCancer/StromaReaction/eval/cell_level_feature/load_ROI2QuPath.groovy