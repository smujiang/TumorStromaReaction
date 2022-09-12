## Background
Similar to a process of normal wound healing, the proliferation of tumor cells create wound to adjacent tissues and 
tumor-stroma reaction (TSR) in cancer has been associated with increased extracellular matrix and production of 
growth factors to facilitate recovery growth of injured tissues. In ovarian cancers(OC), histopathological examination of 
tumor-stroma reaction is critical to differentiate low-grade serous carcinoma from serous borderline serous tumor (SBOT), with the latter lacking tumor triggered stroma reaction. More importantly, tumor-stroma reaction has been reported to facilitate tumorigenesis and associated with prognostic differences in many solid cancers such as cholangiocarcinoma, pancreatic cancer, melanoma, and OC.
However, paucity of study aims to overcome subjective bias or automate TSR evaluation for enabling association analysis to a large cohort. 
In this work, we proposed to use deep learning methods to computationally quantify TSR in whole slide images. The workflow is shown in Figure 1.
The predicted TSR scores were used to establish associations between tissue-level features, prognosis, and molecular pathways of high-grade serous ovarian carcinoma (HGSOC).
![Workflow](./img/Figure_1.png)

More details can be found in [our paper](https://www.frontiersin.org/articles/10.3389/fmed.2022.994467/full) published on Frontiers in Medicine.
### cite our work
```
author = {Jiang Jun, Tekin Burak, Yuan Lin, Armasu Sebastian, Winham Stacey J., Goode Ellen L., Liu Hongfang, Huang Yajue, Guo Ruifeng, Wang Chen},
title =  {{Computational tumor stroma reaction evaluation led to novel prognosis-associated fibrosis and molecular signature discoveries in high-grade serous ovarian carcinoma}},
journal  ={Frontiers in Medicine},
volume ={9},
doi  = {10.3389/fmed.2022.994467},
year  = {2022}
}
```
## Data Processing
Pathologists were invited to use polygons to label homogenous regions. Annotation criteria can be found in our paper.
For each region, stroma-reaction intensity scores measuring fibrosis, cellularity and orientation were assigned.  
According to task, annotations were parsed separately from QuPath with corresponding groovy scripts 
Please refer to ./data_processing/*.groovy folder for details.

## Testing a H&E whole slide image
Please refer to the bash script in ./test_WSI.sh. You will be able to know the pipeline steps written in python code.
















