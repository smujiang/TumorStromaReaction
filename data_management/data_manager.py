import os
from configparser import ConfigParser


class WSI_data:
    def __init__(self, data_fn):
        self.meta_data = ConfigParser()
        self.meta_data.read(data_fn)
        self.scanning_info = Scanning_info()
        self.scanning_info.scan_request_num = self.meta_data.get('scanning_info', 'scan_request_num')
        self.scanning_info.contact_name = self.meta_data.get('scanning_info', 'contact_name')
        self.scanning_info.PI_name = self.meta_data.get('scanning_info', 'PI_name')
        self.scanning_info.resolution = self.meta_data.get('scanning_info', 'resolution')

        self.sample_info = Sample_info()
        self.sample_info.block_ID = self.meta_data.get('sample_info', 'block_ID')
        self.sample_info.clinic_num = self.meta_data.get('sample_info', 'clinic_num')
        self.sample_info.deidentified_ID = self.meta_data.get('sample_info', 'deidentified_ID')
        self.sample_info.annotator = self.meta_data.get('sample_info', 'annotator')
        self.sample_info.ROI_cnt = self.meta_data.getint('sample_info', 'ROI_cnt')
        self.sample_info.tissue_site = self.meta_data.get('sample_info', 'tissue_site')
        self.sample_info.tissue_type = self.meta_data.get('sample_info', 'tissue_type')
        self.sample_info.fresh_cut = self.meta_data.get('sample_info', 'fresh_cut')
        self.sample_info.stain_type = self.meta_data.get('sample_info', 'stain_type')
        self.sample_info.histology_type = self.meta_data.get('sample_info', 'histology_type')

        self.cell_level_features = Cell_level_features()
        self.cell_level_features.cell_feature_names = self.meta_data.get('cell_level_features', 'cell_feature_names')
        self.cell_level_features.tumor_cell_mean = self.meta_data.get('cell_level_features', 'tumor_cell_mean')
        self.cell_level_features.stroma_cell_mean = self.meta_data.get('cell_level_features', 'stroma_cell_mean')
        self.cell_level_features.tumor_cell_std = self.meta_data.get('cell_level_features', 'tumor_cell_std')
        self.cell_level_features.stroma_cell_std = self.meta_data.get('cell_level_features', 'stroma_cell_std')
        self.cell_level_features.cell_density_mean = self.meta_data.get('cell_level_features', 'cell_density_mean')
        self.cell_level_features.cell_density_std = self.meta_data.get('cell_level_features', 'cell_density_std')

        self.tissue_level_features = Tissue_level_features()
        self.tissue_level_features.TSR_score = self.meta_data.get("tissue_level_features", 'TSR_score')

    def write2file(self, out_fn):
        # write values to file
        # print("write values to file")
        with open(out_fn, 'w') as configfile:    # save
            self.meta_data.write(configfile)

class Scanning_info:
    def __init__(self):
        self.scan_request_num = ""
        self.contact_name = ""
        self.PI_name = ""
        self.resolution = 0.25

class Sample_info:
    def __init__(self):
        self.block_ID = ""
        self.clinic_num = ""
        self.deidentified_ID = ""
        self.annotator = ""
        self.ROI_cnt = 5
        self.tissue_site = "ovarian"
        self.tissue_type = "primary"
        self.fresh_cut = "True"
        self.stain_type = "H&E"
        self.histology_type = "LGSOC"

class Cell_level_features:
    def __init__(self):
        self.feature_names = ["", "cell_spatial_density"]
        self.tumor_cell_mean = [0.23, 0.121]
        self.stroma_cell_mean = [0.23, 0.121]
        self.tumor_cell_std = [0.23, 0.121]
        self.stroma_cell_std = [0.23, 0.121]

class Tissue_level_features:
    def __init__(self):
        self.TSR_score = []


if __name__== "__main__":
    print("start to test")


