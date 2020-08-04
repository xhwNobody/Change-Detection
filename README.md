# Change-Detection
This repository provides a PyTorch implementation of A Deep Learning Method for Change Detection in Synthetic Aperture Radar Images.
This is a paper link:https://www.onacademic.com/detail/journal_1000042300093999_1cf9.html

## 1.Inference
1.python make_test_sample.py:By giving a SAR image, it will make all samples. 

2.python model/inference.py:To get result.

## 2.Dataset
The dataset contains three SAR images,including 'Ottawa','FarmlandC' and 'FarmlandD'.

## 3.How to Train

1.python pre_pseudo_label_SFCM.py:To make prepseudo labels.It is not convenient to upload too many files,you can get the preprocessed data through my link.

2.python choose_correct_sample.py:Choose the right samples.

3.python train.py
