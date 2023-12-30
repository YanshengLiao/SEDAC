# SEDAC: A CVAE-based Data Augmentation Method for Security Bug Report Identification

## Introduction

This repository provides source code of **SEDAC**.

**SEDAC** is a **SE**curity bug report identification 
framework composed of **D**istilBERT **A**nd 
**C**onditional variation autoencoder (CVAE).
It cosists of four stages: 

Stage 1: **Text Representation**  
Stage 2: **CVAE Model Training**  
Stage 3: **Synthesize Bug Report Vectors**  
Stage 4: **SBR Identification**  

## Environment
- Python package:
  - pandas==1.5.2
  - numpy==1.23.4
  - torch==2.0.0
  - scikit-learn==1.2.0

## Usage
There are 3 python files in total: `CVAE_model.py` is 
the structure of CVAE model, `cvae_synthesis.py` is
the synthesis process of four Apache projects, and
`cvae_synthesis_for_Chromium` is the synthesis process of
Chromium project. Since the csv file of Chromium is too
large to be read, it is divided into several chunks in 
case of memory leak.  

For `Stage 1: Text Representation`, `project` can be assigned
diffenrent csv files, including `Camel.csv`,`Derby.csv`,
and `Wicket.csv`.
```
project = pd.read_csv("Ambari.csv")
```
`pipeline()` is a easy way to use different models for inference.
We use `feature-extraction` here to extract the hidden states 
from the base transformer, which can be used as features in downstream tasks.
`model` and `tokenizer` can be assigned different BERT
variants for experiments.
```
feature_extraction = pipeline('feature-extraction', model="distilbert-base-uncased",
                              tokenizer="distilbert-base-uncased", truncation = True, max_length = 512)
```
For `Stage 4: SBR Identification`, `classifier` can be assigned
different machine learning algorithms for experiments.
```
classifier = LogisticRegression()
```
