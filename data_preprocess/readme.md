# Instruction for Data Pre-processing

## FoD500
 We can directly download from [Defocus-Net](https://github.com/dvl-tum/defocus-net). The first 400 stacks are the training
 set and the last 100 stacks is the test set. 
 
## DDFF-12
We need to first modify the ```data_pth``` and ```out_pth``` in 
```data_preprocess/my_refocus.py ``` and then run 
```
python data_preprocess/my_refocus.py 
```
to first get the focal stack images. Please make sure the path variables is correct, according to your data location. Next, run 
```
python data_preprocess/pack_to_h5.py --img_folder <OUTPUT FORM MY_REFOCUS> --dpth_folder <PATH_TO_DEPTH> --outfile <PATH_TO H5 OUTPUT>
```
This code will generate a new ```.h5``` which includes the training and validation set. For test set, we can directly download
from the official [DDFF-12](https://hazirbas.com/datasets/ddff12scene/) webiste. The reason we do not use their pre-processed training file
is because we want to further split the original training set into training and validation set.  

## Mobile Depth
please first download from [Mobile Depth](https://www.supasorn.com/dffdownload.html), and then modify the path variables in  ```data_preprocess/reorganize_mobileDFF.py``` and run it. 
```
python data_preprocess/reorganize_mobileDFF.py 
```
This code will re-organize this dataset, by extract the aligned stacks and put a ```.txt``` file in each scene folder that 
indicates the focal distance of each frame. 

## Acknowledgement
Part of the code for DDFF-12 pre-processing is adopted from [DDFF-toolbox](https://github.com/hazirbas/ddff-toolbox)