# config/config.py
# Created by sonymd at 2/5/23
# Mail: sonymd@msu.edu
# GitHub: www.github.com/redwankarimsony
# Graduate Researcher, iPRoBe Lab, CSE, MSU
import os


class configClass:
    def __init__(self):
        self.experiment_name = 'base_experiment'
        self.results_dir = f"results/{self.experiment_name}"
        self.img_dir = "/home/sonymd/Desktop/IFACTS/IFACTS-SPG-Radiography/Annotation/images_annotation_2"
        self.annot_dir = "/home/sonymd/Desktop/IFACTS/IFACTS-SPG-Radiography/Annotation/Three_ROI_280_identities"
        self.width = 512
        self.height = 512
        self.valid_ratio = 0.5
        self.box_preset = 0    #[0=(T1-T5 segment only),
                        # 1= (Whole Vertebral Column Only)
                        # 2= (T1-T5) with Clavicle
                        # 3 =(Consider Whole Xray)]



        self.make_results_dir()

    def make_results_dir(self):
        try:
            os.makedirs(self.results_dir, exist_ok=False)
        except Exception as e:
            print(e)


cfg = configClass()

