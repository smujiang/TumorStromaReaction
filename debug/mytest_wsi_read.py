import openslide
import os

wsi_dir = "/Jun_anonymized_dir/OvaryCancer/WSIs"
wsi_fn = "deidentified_case.svs"
wsi_obj = openslide.open_slide(os.path.join(wsi_dir, wsi_fn))
img = wsi_obj.read_region((1000,1000), 0, (2000, 2000))
thumb = wsi_obj.get_thumbnail([1000, 1000]).convert("RGB")
print(img)

