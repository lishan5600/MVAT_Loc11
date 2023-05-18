# ProteinSubcellularLocation
## Environment


  - python 3.6
  - tensorflow 1.12.0

## Dataset
BaiduDisk Link
 link：https://pan.baidu.com/s/1KrDFA8B7jubTA83FIMHpYA
 Code：yl43 
  
## Usage

  * **ImageSeparation**  
     This part is responsible for image channel separation.The main function is processAll.m  
  * **feature extraction**  
     This part is responsible for feature extraction.
     * Multi Label/ -The five convolutional neural networks for mutli-label images feature extraction.
     * Single Label/-The five convolutional neural networks for single-label images feature extraction.
     * traditional/-Used to extract Haralick texture,LBP,LTP and LQP features.The main function is getAllFeature.m
  * **feature selection**  
    Using mrmr for feature selection.
  * **classification**  
     * svmMutli.py-The classifier of multi-label
     * svmSingle.py-The classifier of single-label

## Applicable situation
  Multi-label part of the algorithm is suitable for multi-label problems, and single-label part is suitable for multi-classification problems.
  This algorithm can used to predict protein subcelluar location.
