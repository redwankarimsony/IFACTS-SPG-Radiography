import glob
import os
import itertools


import numpy as np


def scan_images(img_dir, exts=["jpg", "png", "jpeg", "bmp"]):
    print(img_dir, f"{img_dir}/*.{exts[0]}")
    img_files = []
    for ext in exts:
        img_files.extend(glob.glob(f"{img_dir}/*.{ext}"))
        img_files.extend(glob.glob(f"{img_dir}/*.{ext.upper()}"))
    

    IDs = set([x.split("_")[-4] for x in img_files])
    print(f"IDs: {len(IDs)}, Files: {len(img_files)}")
    return img_files, IDs

def get_label(x):
    return  (x[0].split("_")[-4] == x[1].split("_")[-4]) * 1.


if __name__ == "__main__":
    img_dir = "Annotation/images_annotation_1"
    img_files, IDs = scan_images(img_dir=img_dir)
    combs = itertools.combinations(img_files, 2)
    with open("Annotation/match_pairs.txt", "w") as fp:
        for comb in combs:
            print(comb[0].split("/")[-1], "\t", comb[1].split("/")[-1], "\t", get_label(x=comb), file=fp)


    texts = np.loadtxt("Annotation/match_pairs.txt", dtype= "str")
    print(texts)
    

