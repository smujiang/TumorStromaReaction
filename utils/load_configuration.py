from configparser import ConfigParser


class PatchExtractionConfig:
    def __init__(self, config_fn):
        config = ConfigParser()
        config.read(config_fn)
        self.output_dir = config.get('patch_extraction', 'output_dir')
        self.patch_size = config.getint('patch_extraction', 'patch_size')
        self.stride = config.getint('patch_extraction', 'sampling_stride')
        self.rescale_to = config.getint('patch_extraction', 'patch_rescale_to')
        self.log_dir = config.get('patch_extraction', 'log_dir')
        self.WSI_dir = config.get('patch_extraction', 'WSI_dir')
        self.wsi_ext = config.get('patch_extraction', 'wsi_ext')
        self.case_list = config.get('patch_extraction', 'case_list')

class TestSegmentationConfig:
    def __init__(self, config_fn):
        config = ConfigParser()
        config.read(config_fn)
        self.output_dir = config.get('tumor_stroma_segmentation', 'output_dir')
        self.model_dir = config.get('tumor_stroma_segmentation', 'model_dir')
        # self.model_fn = config.getint('tumor_stroma_segmentation', 'model_fn')
        self.class_text_list = config.get('tumor_stroma_segmentation', 'class_text_list')
        self.color_list = config.get('tumor_stroma_segmentation', 'color_list')
        self.patch_dir = config.get('tumor_stroma_segmentation', 'patch_dir')
        self.case_list = config.get('tumor_stroma_segmentation', 'case_list')


class CombinedTest:
    def __init__(self, config_fn):
        config = ConfigParser()
        config.read(config_fn)
        self.output_dir = config.get('combined_test', 'output_dir')
        self.patch_dir = config.get('combined_test', 'patch_dir')
        self.all_class_list = config.get('combined_test', 'all_class_list')
        self.labels = config.get('combined_test', 'labels')
        self.IMG_SHAPE = config.get('combined_test', 'IMG_SHAPE')
        self.all_model_list = config.get('combined_test', 'all_model_list')
        self.model_weights_path = config.get('combined_test', 'model_weights_path')
        self.case_list = config.get('combined_test', 'case_list')
        self.segmentation_output_dir = config.get('tumor_stroma_segmentation', 'output_dir')
        self.seg_mask_thumb_dir = config.get('combined_test', 'seg_mask_thumb_dir') # for lite version
        self.tumor_stroma_color = config.get('tumor_stroma_segmentation', 'color_list')
        self.stroma_reaction_score_colors = config.get('combined_test', 'stroma_reaction_score_colors')

# class StromaReactionConfig:
#     def __init__(self, config_fn):
#         config = ConfigParser()
#         config.read(config_fn)
#         self.output_dir = config.get('stroma_reaction_intensity_estimation', 'output_dir')
#         self.model_dir = config.getint('stroma_reaction_intensity_estimation', 'model_dir')
#
#
# class TestStromaReactionConfig:
#     def __init__(self, config_fn):
#         config = ConfigParser()
#         config.read(config_fn)
#         self.output_dir = config.get('stroma_reaction_intensity_estimation', 'output_dir')
#         self.model_fn = config.getint('stroma_reaction_intensity_estimation', 'model_fn')
#

