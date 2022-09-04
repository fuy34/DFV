# DFVDFF: Deep Depth from Focus with Differential Focus Volume

This is a PyTorch implementation of our CVPR-2022 paper:

[Deep Depth from Focus with Differential Focus Volume](https://arxiv.org/pdf/2112.01712.pdf)

[Fengting Yang](http://fuy34.github.io/), [Xiaolei Huang](http://faculty.ist.psu.edu/suh972/), and 
[Zihan Zhou](https://zihan-z.github.io/) 

Please feel free to contact [Fengting Yang](http://fuy34.github.io/) (fuy34bkup@gmail.com) if you have any questions.

## Prerequisites
The code is developed and tested with
- Python 3.6
- Pytorch 1.0.0 (w/ Cuda 10.0) and 1.6.0 (w/ Cuda 10.2)
- More details are available in ```requirements.txt```

## Data Preparation
### Download 
The data used in our experiment are from [FoD500](https://github.com/dvl-tum/defocus-net),  [DDFF-12](https://hazirbas.com/datasets/ddff12scene/),
and [Mobile Depth](https://www.supasorn.com/dffdownload.html). 

* For FoD500, we directly use the original raw data without pre-processing. 
* For DDFF12, we provide our pre-processed data [here](https://pennstateoffice365-my.sharepoint.com/:u:/g/personal/fuy34_psu_edu/ERBeMZVm8UhNnQNIg1zXe6IBfLVpTxJtYuPymgU1TqjAbQ?e=g9u9kX). If you prefer to generate the training and validation split by yourself.
 Please download the ```Lightfield (24.5GB)``` and ```Depth (57.9MB)``` from the website, and 
follow the instruction in the next section to prepare the train and validation set. The DDFF-12 test set
is only needed if you wish to submit your test result to the [leaderboard](https://competitions.codalab.org/competitions/17807#learn_the_details). You can directly use 
the pre-processed test set at the [ddff-pytorch](https://github.com/soyers/ddff-pytorch) repository.
* For Mobile Depth, we need to reorganize the files. Please follow the steps shown in the next section. Note that no ground truth is provided in this dataset, and we only
use it for qualitative evaluation. 

### Pre-processing
For FoD500 dataset, no data pre-processing is needed. 

For DDFF-12 dataset, please first modify the ```data_pth``` and ```out_pth``` in 
```data_preprocess/my_refocus.py ``` and then run 
```
python data_preprocess/my_refocus.py 
```
to get the focal stack images. The path variables must be corrected, according to your data location. Next, run 
```
python data_preprocess/pack_to_h5.py --img_folder <OUTPUT FORM MY_REFOCUS> --dpth_folder <PATH_TO_DEPTH> --outfile <PATH_TO H5 OUTPUT>
```
This will generate a ```.h5``` file for training and validation. The reason we do not use the official training and validation split
is that some stacks in their validation set are actually from the same scene included in their training set. We wish no scene 
overlapping between training and validation set for a more accurate validation.  

For Mobile depth dataset, please modify the path variables in  ```data_preprocess/reorganize_mobileDFF.py``` and then run it. 
```
python data_preprocess/reorganize_mobileDFF.py 
```

## Training
Given the DDFF-12 h5.file in ```<DDFF12_PTH>```, and FoD data folder in ```<FOD_PTH>```, please run 
```
CUDA_VISIBLE_DEVICES=0 python train.py --stack_num 5 --batchsize 20 --DDFF12_pth <DDFF12_PTH> --FoD_pth <FOD_PTH> --savemodel <DUMP_PTH>  --use_diff 0/1
```
to train the model.  ```--use_diff 0``` refers to the simple focus volume model (Ours-FV), and ```--use_diff 1``` corresponds to
 the differential focus volume model (Ours-DFV). We have shared [Our-FV](https://drive.google.com/file/d/1oF0MZC3zBY-HRlXOYDlHqiTJ_KgPfEQP/view?usp=sharing)
and [Our-DFV](https://drive.google.com/file/d/1kKJlZybv4Kbpn7Xa2f2K25VErOQyind8/view?usp=sharing) checkpoint pre-trained on the FoD500 and DDFF-12 training set. 
Please note this is not the final model for our DDFF-12 submission which we also include the DDFF-12 validation set in the training.  

## Evaluation
### DDFF-12
To evaluate on the DDFF-12 validation set, run
```
python eval_DDFF12.py --stack_num 5 --loadmodel <CKPT_PTH> --data_path  <DDFF12_PTH> --use_diff 0/1
```
The number generate at the end shows the metrics 
```mse```,	```rms```, ```log_rms```, ```abs_rel```, ```sq_rel```,	```a1```, ```a2```, ```a3```, ```Bump.```, ```avgUnc.``` in order.
Please check the [DDFF-12 dataset](https://arxiv.org/pdf/1704.01085.pdf) paper for their meaning, except ```avgUnc.``` 
which is introduced by us to evaluate the network uncertainty to its prediction. 

Also if you are not using our pre-trained checkpoint, please comment the following lines in ```eval_DDFF12.py```. We add these lines at the paper submssion for the reviewer to better reproduce our results. 
https://github.com/fuy34/DFV/blob/374420792ffde65bc68db101e25cdc5b6cbf0990/eval_DDFF12.py#L37-L41

For website submission or visualization on the test set. 
```
python DDFF12_submisson.py --data_path <TEST_SET_PTH> --loadmodel <CKPT_PTH> --use_diff 0/1 --outdir <DUMP_PTH>
```
### FoD500
To generate test results, run 
```
python FoD_test.py --data_path <FOD_PTH> --loadmodel <CKPT_PTH> --use_diff 0/1 --outdir <FOD_DUMP_PTH>
```
The code will also provide the ```avgUnc.``` result on FoD500. Next, the evaluation results can be generated by running
```
python eval_FoD500.py --res_path <FOD_DUMP_PTH>
```

### Mobile depth
To generate qualitative results, run 
```
python eval_mobile_Depth.py --data_path <FOD_PTH> --loadmodel <CKPT_PTH> --use_diff 0/1 --outdir <FOD_DUMP_PTH>
```

## Acknowledgement
Parts of the code are developed from [HSM](http://vision.middlebury.edu/stereo/submit3/), 
[DDFF-pytorch](https://github.com/soyers/ddff-pytorch), [DDFF-toolbox](https://github.com/hazirbas/ddff-toolbox) and [DefocusNet](https://github.com/dvl-tum/defocus-net).

