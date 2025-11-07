# Automatic Comparative Chest Radiography Using Deep Neural Networks  
**IEEE Access, Vol. 13, Jan 2025, pp. 4398–4410**  
📘 DOI: [10.1109/ACCESS.2025.3525579](https://doi.org/10.1109/ACCESS.2025.3525579)  
🔗 [IEEE Xplore Link](https://ieeexplore.ieee.org/document/10820523)  
🧑‍💻 [Code & Dataset Repository](https://github.com/iPRoBe-lab/MSUFAL-Radiography-Dataset)  

---

## 🧠 Abstract  
Comparative medical radiography involves matching antemortem (AM) and postmortem (PM) radiographs for personal identification in forensic investigations.  
In this paper, we introduce a **deep neural network–based system** for radiographic identification using AM and PM chest X-rays.  
A dataset of **5,165 anonymized chest radiographs** from **760 individuals** was compiled from the NIH Chest X-ray dataset and MSU Forensic Anthropology Laboratory (MSUFAL).  
Three expert-annotated **regions of interest (ROIs)** — thoracic vertebrae (T1–T5), clavicles, and complete vertebral column — were explored using **ResNet**, **DenseNet**, and **EfficientNet** architectures.  
The results show:  
- T1–T5 ROI yields the best recognition performance.  
- EfficientNets outperform ResNets and DenseNets.  
- An **ensemble of models** across ROIs achieves the highest accuracy.  
This work introduces the first systematic study of ROIs in forensic radiographic identification and releases a new expert-annotated dataset to advance the field.  

---

## 📈 Key Contributions  
- Developed a **deep learning–based radiographic identification system** for AM–PM chest X-ray matching.  
- Compiled a **unique expert-annotated dataset** of 5,165 radiographs representing 760 individuals.  
- Evaluated **three network families (ResNet, DenseNet, EfficientNet)** across multiple ROIs.  
- Achieved **97.81% Rank-1 accuracy** using an ensemble of ROI-specific models.  
- Released the **MSUFAL Radiography Dataset** and source code for reproducibility and future research.  

---

## 🧩 Method Overview  
1. **Data Curation:**  
   - Combined radiographs from NIH Chest X-ray and MSUFAL datasets.  
   - Manual annotation of 3 ROIs by forensic experts.  

2. **Model Training:**  
   - Used CNN families (ResNet, DenseNet, EfficientNet) on each ROI.  
   - Evaluated both **closed-set** and **open-set** identification.  

3. **Ensemble Fusion:**  
   - Combined predictions from ROI-based models for higher reliability.  

4. **Explainability:**  
   - Used **GradCAM++** to visualize model focus regions and ensure interpretability.  

---

## 🧮 Results Summary  
| Task | Best Model | ROI | Rank-1 Accuracy |
|------|-------------|-----|----------------|
| Closed-set AM Identification | DenseNet-161 | T1–T5 | 87.51% |
| Closed-set PM Identification | EfficientNet-B7 | Whole Image | 92.38% |
| Ensemble (All ROIs) | EfficientNet-B7 + DenseNet-161 + ROI Models | Combined | **97.81%** |

- Verification evaluated via **True Match Rate (TMR)** and **ROC curves**.  
- Models trained on AM data generalized well to PM radiographs.  

---

## 🧠 Explainability  
GradCAM++ saliency visualizations confirmed that ROI-trained models focus on **bony features** (e.g., clavicles and vertebrae), which are critical for forensic identification.  
The ensemble effectively integrates local and global information, improving robustness against soft tissue variation and PM imaging artifacts.  

---

## 📊 Dataset & Code  
- **Dataset:** 5,165 radiographs (AM + PM)  
- **Individuals:** 760  
- **Sources:** NIH Chest X-ray, MSUFAL forensic cases  
- **Released at:**  
  - 🧾 Dataset: [github.com/iPRoBe-lab/MSUFAL-Radiography-Dataset](https://github.com/iPRoBe-lab/MSUFAL-Radiography-Dataset)  
  - 💻 Code: [github.com/iPRoBe-lab/ComparativeRadiography](https://github.com/iPRoBe-lab/ComparativeRadiography)  

---

## 👩‍🔬 Authors  
| Name | Affiliation |
|------|--------------|
| **Redwan Sony** | Dept. of Computer Science and Engineering, Michigan State University, USA |
| **Carolyn V. Isaac** | Dept. of Anthropology, Michigan State University, USA |
| **Alexis Vanbaarle** | Dept. of Anthropology, Michigan State University, USA |
| **Clara J. Devota** | School of Medicine, Wayne State University, USA |
| **Todd Fenton** | Dept. of Anthropology, Michigan State University, USA |
| **Joseph T. Hefner** | Dept. of Anthropology, Michigan State University, USA |
| **Arun Ross** | Dept. of Computer Science and Engineering, Michigan State University, USA |

---

### 📖 Citation
```bibtex
@ARTICLE{10820523,
  author={Sony, Redwan and Isaac, Carolyn V. and Vanbaarle, Alexis and Devota, Clara J. and Fenton, Todd and Hefner, Joseph T. and Ross, Arun},
  journal={IEEE Access}, 
  title={Automatic Comparative Chest Radiography Using Deep Neural Networks}, 
  year={2025},
  volume={13},
  number={},
  pages={4398-4410},
  keywords={Radiography;Forensics;Biological system modeling;Biometrics;Diagnostic radiography;Training;Probes;X-ray imaging;Feature extraction;Accuracy;Anthropology;deep neural networks;radiographic identification;region of interest},
  doi={10.1109/ACCESS.2025.3525579}}

---

## 🧭 Keywords  
`Comparative Radiography`, `Forensic Identification`, `Chest X-ray`, `Deep Neural Networks`, `Biometrics`, `Explainability`, `Ensemble Learning`

---

## 🧩 Acknowledgments  
This research was supported by the **MSU Strategic Projects Grant**.  
The authors thank the forensic experts who assisted in data annotation and validation.

---

## 📫 Contact  
**Redwan Sony**  
Ph.D. Student, iPRoBe Lab, Michigan State University  
📧 redwankarimsony@msu.edu  
🌐 [GitHub: redwankarimsony](https://github.com/redwankarimsony)  
🌐 [HomePage](https://redwankarimsony.github.io)  

---

> © 2025 IEEE. Open Access publication under Creative Commons License.  
> For non-commercial research use only.





# <center>IFACTS-SPG-RADIOGRAPHY</center>

**How to Run Experiments**

1. Make a new config file in the `configs` directory appropriately
2. Update the configuration import in the `train_base.py` file to make things smoother.
3. Run your bash script witht the appropriate commands. Care should be taken to select the gpus concurrent processes delay time etc in the bash script. 
4. After all the experiment have finished running, run the `summarize.py` file with the appropriate experiment name to summarize that experiment only in the `experiments` directory. 
5. Use the script `analysis_with_cmc.py` to generate the cumulative matching curves for the models and box presets. 
6. Use the script `analysis_with_grad_cam.py` to generate the gradCAM analysis of the classification systems. 

