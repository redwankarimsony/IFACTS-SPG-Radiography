# config/config.py
# Created by sonymd at 2/5/23
# Mail: sonymd@msu.edu
# GitHub: www.github.com/redwankarimsony
# Graduate Researcher, iPRoBe Lab, CSE, MSU

class cfg:
    def __init__(self):
        self.experiment_name = 'base_experiment'
        self.img_dir= "/home/sonymd/Desktop/IFACTS/IFACTS-SPG-Radiography/Annotation/images_annotation_2"
        self.annot_dir= "/home/sonymd/Desktop/IFACTS/IFACTS-SPG-Radiography/Annotation/Three_ROI_280_identities"
        self.width= 224
        self.height= 224
        self.valid_ratio=30
