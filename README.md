
# JIS 2023
Research Artifact Repository: Coffee Plant Leaf Disease Detection for Digital Agriculture


![https://doi.org/10.5281/zenodo.10056491)](https://zenodo.org/badge/DOI/10.5281/zenodo.10056491.svg)


## Repository Description
The Research Artifact Repository "JIS 2023", associated to "Coffee Plant Leaf Disease Detection for Digital Agriculture" article serves as a centralized and organized source of all data, codes, models, and results generated during the scientific research process dedicated to detecting diseases in coffee plant leaves, aiming to drive digital agriculture. This repository constitutes a comprehensive collection of resources that were utilized, developed, and refined throughout the research process, providing a detailed insight into the research methodology and the outcomes achieved.


## Repository Contents

#### 1. Input Data
- Haralick Features file extracted from JMUBEN and JMUBEN2 datasets.

#### 2. Codes and Algorithms
- Codes for data exploration;
- Image preprocessing algorithms;
- Implementation of Machine Learning and Artificial Intelligence algorithms for coffee leaf disease detection.

#### 3. Machine Learning Models
- Trained Multi-layer Perceptron (MLP) Artificial Neural Networks (ANN) and Convolucional Neural Network (CNN) models for detecting specific diseases in coffee leaves;
- Training logs.

#### 4. Results and Evaluations
- CSV metrics files.

#### 4. Performance reports of the models
- Graphs and visualizations of the obtained results;
- Comparison of different algorithms and approaches used in the research.


## Preparing for Initial Execution After Download
Before you can start using the resources within the "Coffee Plant Leaf Disease Detection for Digital Agriculture" repository, it's essential to download the repository and perform certain installations and configurations. This guide will walk you through the process step by step.

### Step 1: Downloading the Repository
1. Navigate to the repository's web page on GitHub;
2. Click on the "Download" or "Code" button, then select "Download ZIP";
3. Once the ZIP file is downloaded, extract its contents to a location of your choice on your computer.

### Step 2: Installing Required Software:**
Ensure that you have the following software installed on your system:
- **Python:** Check if Python is installed by opening a terminal or command prompt and typing `python --version` or `python3 --version`. If not installed, download and install Python from the official website (https://www.python.org/downloads/).
- **Package Manager (pip):** Pip usually comes pre-installed with Python. You can verify its presence by typing `pip --version` in the terminal or command prompt. If not available, you can install it by following the instructions on the official website (https://pip.pypa.io/en/stable/installing/).

### Step 3: Setting Up Virtual Environment (Optional but Recommended)**
It's good practice to work within a virtual environment to manage dependencies. To set up a virtual environment, follow these commands in the terminal or command prompt:
```bash
# Install virtualenv (if not already installed)
pip install virtualenv

# Create a new virtual environment
virtualenv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS and Linux:
source venv/bin/activate
```

### Step 4: Installing Dependencies
Navigate to the repository folder using the terminal or command prompt and install the required packages using the following command:
```bash
pip install -r requirements.txt
```

### Step 5: Setting Up Datasets and Folder Renaming
It is crucial to download and unzip the three datasets used in this research. Once downloaded and unzipped, these datasets should be placed in `datasets` folder. In special, subfolders of JMUBEN and JMUBEN2 datasets must be placed in `datasets/jmuben/original` folder.

1. [JMUBEN](https://data.mendeley.com/datasets/t2r6rszp5c/1) and [JMUBEN2](https://data.mendeley.com/datasets/tgv3zb82nd/1) datasets;
2. [BRACOL](https://data.mendeley.com/datasets/yy2k5y8mxg/1) dataset (if do not work, try download [here](https://drive.google.com/file/d/15YHebAGrx1Vhv8-naave-R5o3Uo70jsm/view));
3. [Plant Patologies](https://data.mendeley.com/datasets/vfxf4trtcg/5) dataset;
4. [RoCoLe](https://data.mendeley.com/datasets/c5yvn32dzg/2) dataset.

Additionally, it is important to note that datasets root folders within the repository might need to be renamed as per the instructions below.

1. BRACOL: rename to `bracol`;
2. Plant Patologies: rename to `plant_patologies`;
3. RoCoLe: rename to `rocole`.

Renaming these folders correctly ensures that the code can access the necessary files and resources seamlessly. Careful attention to dataset installation, folder allocation, and renaming is essential to guarantee the smooth operation of the code during its initial execution.

### Step 6: First-time Execution
By following these steps, you have successfully downloaded the repository, prepared your system, and are ready to explore and execute the research code for coffee plant leaf disease detection in the context of digital agriculture.
    
## Authors

- [@laisdib](https://github.com/laisdib)
- [@elloa](https://github.com/elloa)
