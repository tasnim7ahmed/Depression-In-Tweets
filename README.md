# Depression-In-Tweets

 ## Prerequisites
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

### Getting started


- Install PyTorch and other dependencies.

  please run the command `pip install -r requirements.txt`.

### Dataset
Mental health research through data-driven methods has been hindered by a lack of standard typology and scarcity of adequate data. To address the issue, the authors of the DEPTWEET dataset will provide the dataset freely of charge to researchers in order to promote mental health research. [Send an email to the corresponding author for obtaining the dataset](mailto:mohsinulkabir@iut-dhaka.edu).

- Go to the `Scripts` directory:
```bash
cd Scripts
```
- Place the dataset in the `../Dataset/` folder and run the command: 
```bash
python dataset.py
```

### Training and Evaluation
- Go to the `Scripts` directory:
```bash
cd Scripts
```

- Train stand-alone model (BERT-base-uncased) with GPU support:
```bash
python train.py
```
- Train stand-alone model (BERT-base-uncased) with CPU only:
```bash
python train.py --device CPU
```

- Train stand-alone SVM or LSTM model:
```bash
Codes are available at ./Notebooks folder.
```
Information regarding other training parameters can be found at `Scripts/common.py` file.

Fine-tuned models will be saved at `../Models/` folder.\
Evaluation output files will be saved at `../Output/` folder.\
Figures will be saved at `../Figures/` folder.


### Citation
If you use this code for your research, please cite our [paper](https://www.sciencedirect.com/science/article/abs/pii/S0747563222003235).
```
@article{KABIR2022107503,
title = {DEPTWEET: A typology for social media texts to detect depression severities},
journal = {Computers in Human Behavior},
pages = {107503},
year = {2022},
issn = {0747-5632},
doi = {https://doi.org/10.1016/j.chb.2022.107503},
url = {https://www.sciencedirect.com/science/article/pii/S0747563222003235},
author = {Mohsinul Kabir and Tasnim Ahmed and Md. Bakhtiar Hasan and Md Tahmid Rahman Laskar and Tarun Kumar Joarder and Hasan Mahmud and Kamrul Hasan},
keywords = {Social media, Mental health, Depression severity, Dataset},
abstract = {Mental health research through data-driven methods has been hindered by a lack of standard typology and scarcity of adequate data. In this study, we leverage the clinical articulation of depression to build a typology for social media texts for detecting the severity of depression. It emulates the standard clinical assessment procedure Diagnostic and Statistical Manual of Mental Disorders (DSM-5) and Patient Health Questionnaire (PHQ-9) to encompass subtle indications of depressive disorders from tweets. Along with the typology, we present a new dataset of 40191 tweets labeled by expert annotators. Each tweet is labeled as ‘non-depressed’ or ‘depressed’. Moreover, three severity levels are considered for ‘depressed’ tweets: (1) mild, (2) moderate, and (3) severe. An associated confidence score is provided with each label to validate the quality of annotation. We examine the quality of the dataset via representing summary statistics while setting strong baseline results using attention-based models like BERT and DistilBERT. Finally, we extensively address the limitations of the study to provide directions for further research.}
}

```
