# Pytorch InceptionV3 model pretrained on iNaturalist dataset


This repository contains the inceptionV3 model pretrained on iNaturalist dataset. The model is converted from tensorflow implementation of work
Large Scale Fine-Grained Categorization and Domain-Specific Transfer Learning (https://github.com/richardaecn/cvpr18-inaturalist-transfer)


Model architecture is defined in myinception.py
The converted pretrained model can be found here https://drive.google.com/file/d/1VHcS2o0aYtr1MkYQP_uRgEGcM9bIcM_E/view


##Training on CUB dataset:
To train the network on CUB dataset run the following command:

python train_inception_classifier.py --cuda --save_dir='<PATH_WHERE_MODEL_SAVED>'

Note that this will only train the classifier (last fully connected layer) on CUB dataset (200 classes). However the feature extractor (all previous layers) is taken from pretrained iNaturalist model. 

##Testing on CUB dataset:

 python test_inception_classifier.py --cuda --load='<PATH_WHERE_MODEL_SAVED>/cub_inception_30.pth'

## Results on CUB Classification
|    Pretrained Model            | CUB-200  |
| -------------            |:-------------:| 
| iNat           |   89.1  |







