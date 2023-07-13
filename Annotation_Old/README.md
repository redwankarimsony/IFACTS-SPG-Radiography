- [images_annotation_1](images_annotation_1)
  Matching radiographs from 100 identies. Both Forensic case data (Antemortem -> Postmortem matches) and NIJ (Antemortem 1 -> Antemortem 2 matches) are included. Some identities have more than 2 associated images meaning more than 1 match between images can be made
- [labels_annotation_1_2022-12-14-04-09-39](labels_annotation_1_2022-12-14-04-09-39)
  .txt files coding a rectangle bounding the area of observer interest for matching features between the 255 images included images_annotation_1
- [images_annotation_1_cropped](images_annotation_1)
  Matching radiographs from 100 identies. Both Forensic case data (Antemortem -> Postmortem matches) and NIJ (Antemortem 1 -> Antemortem 2 matches) are included. Images are finely cropped discarding the unnecesary part around and cropped to near square aspect ratio. 
- [100_identities](100_identities)
  First file is annotation_T1-T5. This includes.txt files coding a rectangle bounding box around the bodies of the 1st thoracic (T1) through 5th thoracic vertebrae (T5). The bounding box runs from the top of the T1 vertebrae to the botton of the T5 vertebrae. The images used in this annotation were pulled from the "images_annotation_1" file, not the cropped file. Second file is "annotation_T1-T4" which includes .txt files coding a rectangle bounding box around the bodies of the 1st thoracic (T1) through 4th thoracic vertebrae (T4). The bounding box runs from the top of the T1 vertebrae to the botton of the T4 vertebrae. The images used in this annotation were pulled from the "images_annotation_1" file, not the cropped file.
- [Annotations_T1-T5_130_identities](Annotations_T1-T5_130_identities)
  This includes.txt files coding a rectangle bounding box around the bodies of the 1st thoracic (T1) through 5th thoracic vertebrae (T5). The bounding box runs from the top of the T1 vertebrae to the botton of the T5 vertebrae. The images used in this annotation were pulled from the "images_annotation_2" file and include 130 identities.
- [Annotations_T1-T5_150_identities](Annotations_T1-T5_150_identities)
  This includes.txt files coding a rectangle bounding box around the bodies of the 1st thoracic (T1) through 5th thoracic vertebrae (T5). The bounding box runs from the top of the T1 vertebrae to the botton of the T5 vertebrae. The images used in this annotation were pulled from the "images_annotation_2" file and include 150 identities.
- [Annotatations_T1-T5_280_identities](Annotatations_T1-T5_280_identities)
  This includes.txt files coding a rectangle bounding box around the bodies of the 1st thoracic (T1) through 5th thoracic vertebrae (T5). The bounding box runs from the top of the T1 vertebrae to the botton of the T5 vertebrae. The images used in this annotation were pulled from the "images_annotation_2" file and include 280 identities.
- [Annotations_ALV](Annotations_ALV)
New data added by Lex.
- [Test_images_radqual](Test_images_radqual)
This folder includes the test annotations and images with radiographic quality coding added at the end of the file name.
0 = 0%-25% of vertebral column visible
1 = 25%-50% of vertebral column visible
2 = 50%-75% of vertebral column visible
3 = 75%-100% of vertebral column visible
