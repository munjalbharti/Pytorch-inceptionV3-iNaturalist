# Pytorch InceptionV3 model pretrained on iNaturalist dataset


This repository contains the inceptionV3 model pretrained on iNaturalist dataset. The model is converted from tensorflow implmentation of work
Large Scale Fine-Grained Categorization and Domain-Specific Transfer Learning (https://github.com/richardaecn/cvpr18-inaturalist-transfer)


Model architecture is defined in myinception.py

Trained pytorch can be found here https://drive.google.com/file/d/1VHcS2o0aYtr1MkYQP_uRgEGcM9bIcM_E/view

Note that the classifier (last fully connected layer) is trained on CUB (200 classes). However the feature extractor (all previous layers) is trained on iNaturalist. 

## Results on CUB Classification
|    Pretrained Model            | CUB-200  |
| -------------            |:-------------:| 
| iNat           |   89.2  |







