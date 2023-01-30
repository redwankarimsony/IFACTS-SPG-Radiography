from glob import glob
import shutil
import os.path as osp


all_label_files = glob("../Annotation/Annotations_T1-T5_150_identities/*.txt")
all_kays = [x.split("/")[-1].split(".")[0]+"." for x in all_label_files]

print(all_kays)
target_dir = "dummy_dataset"
all_img_path = glob("/home/sonymd/Downloads/Chest Radiograph Files/Chest_Radiograph/*/*.*")

for file_key in all_kays:
    for img_path in all_img_path:
        if file_key in img_path:
            shutil.copy(img_path, osp.join(target_dir, img_path.split("/")[-1]))





