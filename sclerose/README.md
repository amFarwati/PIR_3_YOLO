# Test de U-net sur la sclérose en plaque

## extract.py
Ce script permet d'extraire les images (flair) et les segmentations à partir des fichiers .nii. Il est possible de choisir le nombre de tranches à extraire. Un filtre de correction N4 permet également de corriger les inhomogénéités des données pour avoir de meilleurs résultats après l'entrainement.

## moveTest.py
Ce script permet de déplacer une partie (typiquement 20%) des données dans un autre dossier pour constituer le training set.

## trained_model
Ce dossier devrait contenir les modèles entrainées. Il contient actuellement le meilleur modèle que nous avons obtenu (dice score de 0.96).

## 01_UNET_TF2_test.ipynb
Ce notebook permet d'importer et de testr un modèle U-net sur des données non appris durant l'entrainement. Il a été originalement écrit par Thomas GRENIER.

## 02_UNET_TF2_train.ipynb
Ce notebook permet d'entrainer un modèle U-net. Il a été originalement écrit par Thomas GRENIER. Les modifications apportées permettent d'entrainer le modèle sur les données de la sclérose en plaque. 