# RPCANet
**[WACV2024]** Official implementation of paper "RPCANet: Deep Unfolding RPCA Based Infrared Small Target Detection".



## Abstract
Deep learning (DL) networks have achieved remarkable performance in infrared small target detection (ISTD). However, these structures exhibit a deficiency in interpretability and are widely regarded as black boxes, as they disregard domain knowledge in ISTD. To alleviate this issue, this work proposes an interpretable deep network for detecting infrared dim targets, dubbed RPCANet. Specifically, our approach formulates the ISTD task as sparse target extraction, low-rank background estimation, and image reconstruction in a relaxed Robust Principle Component Analysis (RPCA) model. By unfolding the iterative optimization updating steps into a deep-learning framework, time-consuming and complex matrix calculations are replaced by theory-guided neural networks. RPCANet detects targets with clear interpretability and preserves the intrinsic image feature, instead of directly transforming the detection task into a matrix decomposition problem. Extensive experiments substantiate the effectiveness of our deep unfolding framework and demonstrate its trustworthy results, surpassing baseline methods in both qualitative and quantitative evaluations.

## Network Architecture
![overall](https://github.com/fengyiwu98/RPCANet/assets/115853729/03e0efcf-3c81-472e-a591-48a5d79e54ab)
![network_architecture](https://github.com/fengyiwu98/RPCANet/assets/115853729/85d53cf7-48ca-4099-ba05-943cb9811193)


## Requirements
- **Python 3.8**
- **Windows10, Ubuntu18.04 or higher**
- **NVDIA GeForce RTX 3090**
- **pytorch 1.8.0 or higher**
- **More details from requirements.txt** 

## Datasets

**We used NUDT-SIRST, IRSTD-1K and sirst-aug for both training and test. Three datasets can be found and downloaded in: [NUDT-SIRST](https://github.com/YeRen123455/Infrared-Small-Target-Detection), [IRSTD-1K](https://github.com/RuiZhang97/ISNet), [SIRST-Aug](https://github.com/Tianfang-Zhang/AGPCNet). And you can train the model using the dataset we have arranged according to our code settings: [Google Drive](https://drive.google.com/file/d/1rs6ORtekqHmuPEPhyq61iPPVOxx2QF7B/view?usp=drive_link).**
 
**Please first download these datasets and place the 3 datasets to the folder `./datasets/`. More results will be released soon!** 



* **Our project has the following structure:**
```
├──./datasets/
│    ├── NUDT-SIRST/ & IRSTD-1k/ & sirst_aug/
│    │    ├── trainval
│    │    │    ├── images
│    │    │    │    ├── 000002.png
│    │    │    │    ├── 000004.png
│    │    │    │    ├── ...
│    │    │    ├── masks
│    │    │    │    ├── 000002.png
│    │    │    │    ├── 000004.png
│    │    │    │    ├── ...
│    │    ├── test
│    │    │    ├── images
│    │    │    │    ├── 000001.png
│    │    │    │    ├── 000003.png
│    │    │    │    ├── ...
│    │    │    ├── masks
│    │    │    │    ├── 000001.png
│    │    │    │    ├── 000003.png
│    │    │    │    ├── ...
```
<br>

## Commands for Training
* **Run** `run_0.py` **to perform network training:**
```bash
$ python run_0.py
```

## Commands for Evaluate your own results
* **Run** `t_models.py` **to generate file of the format .mat and .png:**
```bash
$ python t_models.py
```
* **The file generated will be saved to** `./results/` **that has the following structure**:
```
├──./results/
│    ├── [dataset_name]
│    │   ├── img
│    │   │    ├── 000000.png
│    │   │    ├── 000001.png
│    │   │    ├── ...
│    │   ├── mat
│    │   │    ├── 000000.mat
│    │   │    ├── 000001.mat
│    │   │    ├── ...
```
* **Run** `cal_from_mask.py` **for direct evaluation**:
```bash
$ python cal_from_mask.py
```

## Citation
```
@InProceedings{Wu_2024_WACV,
    author    = {Wu, Fengyi and Zhang, Tianfang and Li, Lei and Huang, Yian and Peng, Zhenming},
    title     = {RPCANet: Deep Unfolding RPCA Based Infrared Small Target Detection},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {4809-4818}
}
```
