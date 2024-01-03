# E-UNet
This is a official implementation for our paper: "[Novel Enhanced UNet for Change Detection Using Multimodal Remote Sensing Image](https://ieeexplore.ieee.org/document/10287366)" has been published on <font size=5>**IEEE  GEOSCIENCE AND REMOTE SENSING LETTERS**</font> by Zhiyong Lv, Haitao Huang, Weiwei Sun, Tao Lei, Jón Atli Benediktsson, and Junhuai Li.  

 ## Requirements
>python=3.8
pytorch=1.9  
opencv-python=4.6.0.66  
scikit-image=0.18.1  
scikit-learn=0.24.1  

## Usage

### P2P
Use CycleGan.py to convert the image

### Train
1. Load the train and train(val) data path  
python train.py  

### Test
1. Load the model path  
2. Load the test data path  
python test.py

## Citation
If you find our work useful for your research, please consider citing our paper:  
``` 
@ARTICLE{10287366,
  author={Lv, Zhiyong and Huang, Haitao and Sun, Weiwei and Lei, Tao and Benediktsson, Jón Atli and Li, Junhuai},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Novel Enhanced UNet for Change Detection Using Multimodal Remote Sensing Image}, 
  year={2023},
  volume={20},
  number={},
  pages={1-5},
  doi={10.1109/LGRS.2023.3325439}}
```
## Contact us 
If you have any problme when running the code, please do not hesitate to contact us. Thanks.  
E-mail: Lvzhiyong_fly@hotmail.com, hht_zsl@outlook.com
Date: Mar 17, 2023  
