# CS182_proj
# Movie Recommendation System
This project implements and evaluates several movie recommendation models, including traditional collaborative filtering methods and more advanced deep learning approaches.

## Project Structure
CS182_PROJ-MAIN/
│s
├── data/
│   ├── ... (processed data)
│
├── origin/
│   ├── ... (the baseline model code)
│
├── torch_ver/
│   ├── __pycache__/
│   ├── command.md
│   ├── comparison.py
│   ├── data_process.py
│   ├── evaluate.py
│   ├── MMOE_evaluate.py
│   ├── MMOE_train.py
│   ├── MMOE.py
│   ├── model_comparison.py
│   ├── model.py
│   ├── origin_model.py
│   ├── origin_train.py
│   ├── train_comparison.py
│   └── train.py
│
├── .gitignore
├── CFModel.py
├── Content_Based_and_Collaborative_Filtering_Models.ipynb
├── Data_Processing.ipynb
├── Deep_learning_Model.ipynb
├── environment.yml
├── README.md
├── requirements.txt
└── SVD_Model.ipynb

## Models
This repository includes the following models:

CFModel (baseline)

IndependentTimeModel

UserTimeModel

UMTimeModel

TwoStageMMoEModel (MMOE)

## Getting Started
1. Installation
You can set up the environment using either pip with requirements.txt or conda with environment.yml.

2. Data Processing
You can preprocess the data using the Data_Processing.ipynb notebook.

## Training and Evaluating different models:
First, specify the model to train in train_comparison.py, then run comparison.py to train, evaluate and compare different models and plot the performances. The model comparison plots will be saved in data/analysis_plots.

## Other useful info
The four ipynb files and CFModel.py are all the original author's code. We used Deep_Learning_Model as our baseline. The origin folder contains the original code modified to the torch version.  

torch_ver is our own code, which includes four models: IndependentTimeModel, UserTimeModel, UMTimeModel, and TwoStageMMoEModel (MMOE).  

The two networks in MMOE are LSTM and UMTimeModel.  

Both the models and data are available on GitHub in the same format, located in the data directory. The link is as follows:  

https://github.com/Askeladd-James/CS182_proj