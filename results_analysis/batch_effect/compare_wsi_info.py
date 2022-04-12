from openslide import OpenSlide
import os

wsi_dir = "/Jun_anonymized_dir/OvaryCancer/WSIs"

wsi_f1 = "deidentified_case1.svs"
wsi_f2 = "deidentified_case2.svs"
wsi_f3 = "deidentified_case3.svs"


wsi_obj1 = OpenSlide(os.path.join(wsi_dir, wsi_f1))
wsi_obj2 = OpenSlide(os.path.join(wsi_dir, wsi_f2))
wsi_obj3 = OpenSlide(os.path.join(wsi_dir, wsi_f3))

prop_wsi_1 = wsi_obj1.properties
prop_wsi_2 = wsi_obj2.properties
prop_wsi_3 = wsi_obj3.properties

print(prop_wsi_1)
print(prop_wsi_2)
print(prop_wsi_3)

wsi_1_prop_dict = dict(prop_wsi_1)
wsi_2_prop_dict = dict(prop_wsi_2)
wsi_3_prop_dict = dict(prop_wsi_3)

print("Done")







