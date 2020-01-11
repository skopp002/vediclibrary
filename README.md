# vediclibrary
Foundation for Preservation of Knowledge Vedic Library

Dataset Collection:

After placing a cropped image that needs to be added to the dataset, use createVowelDataset.py to generate the inverse binarized and reshaped image which can be added to the collection. 




Classfication:

1. GeetaDevanagariCustomDataset.py to iterate quickly with custom dataset. It uses completely handbuilt dataset and not use UCI repository at all.
2. GeetaDevanagari.py which augments UCI dataset with the custom dataset created. The "path" variable defined in prepareTestTrainData function needs to point to the Train data from UCI dataset


Youtube Presentation of the project: https://www.youtube.com/watch?v=2h_k0kMbb90&list=PLKyD-cTJmDwFwMaXRh3eymbaYL2wScXGf
