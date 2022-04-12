## Data processing
Pathologists were invited to use polygons to label homogenous regions.
For each region, stroma-reaction intensity scores measuring fibrosis, cellularity and polilarity were assigned.  
According to task, annotations were parsed separately from QuPath with corresponding groovy scripts 

### for tumor-stroma segmentation
There will be 

### for stroma reaction intensity estimation

##Training Phase
### Tumor-stroma segmentation

### Reaction intensity estimation

## Performance Evaluation



##Testing Phase (Deploy as web application )
1. upload a WSI to sever
2. process the uploaded image
3. return the results
4. visualization


##Testing Phase (Deploy as local application )
1. Patch extraction
3. return the results
4. visualization


# Pipeline steps
Get tissue level features: 
1. get tumor-stroma segmentation
2. predict tumor-stroma reaction (TSR)
3. Save thumbnails for further processing
4. Summarize TSR scores in entangling area (./summarization/all_case_propotion_ratio_plus.py)

Get cell-level features: (including average cell morphology & color feature and cell density) 
1. After we get the tumor-stroma segmentation, we use "get_rois.py" to automatically get ROIs from the original whole slide images. The code will generate .csv files to save selected ROIs.
2. Run QuPath scripts. Install the command line version of QuPath (can be on linux server), and scripts can be run & managed within ./QuPath_commandline_scripts.bash. The scripts will do the following:
   1) create QuPath projects (main code in "create_projects.groovy")
   2) load ROIs (main code in "load_rio2QuPath.groovy")
   3) get cell features (main code in "getROICellFeatures.groovy")
The code will create QuPath project, import WSIs to the project and load those ROIs from .csv files to the QuPath project.
Then get cellular features from cells insides of ROIs. The features will be saved to .txt files. Please note, the code will set image type to H&E first, otherwise the dimension and feature names may be different.
3. Transfer those .txt files to infodev, and use pre-trained model (in the first paper with Dr. Wang Chen) to do cell classification (cell_prediction.py).
4. Read classification results and ROI .csv files to calculate cell density. (getCellDensity.py)

Save case level features:
1. code in (data_management/write_case_level_meta_data.py)
   a. Save all features to standard meta file for each case
   b. save all cases into one .csv file.


Evaluation:
1. which are HGSOC and SBOT cases?
2. which HGSOC and SBOT cases are used for training?  (TS segmentation? TSR estimation? Cell classification? respectively)
3. Metadata representation? 
   Histogram differences in HGSOC and SBOT?
   Batch effect?
   association map
   mutual-correlation matrix? (group by data driven)
















