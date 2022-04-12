import os

wsi_dir = "\\\\anonymized_dir\\POLE_WSI\\req30990"
ext = '.svs'
wrt_str = '['
wsi_list = os.listdir(wsi_dir)


cnt = 0
for wsi in wsi_list:
    wrt_str += '"' + wsi[0:-len(ext)] + '",'
    cnt += 1
wrt_str = wrt_str[0:-1]
wrt_str += ']'
print("Get %d cases in total" %cnt)
print(wrt_str)






















