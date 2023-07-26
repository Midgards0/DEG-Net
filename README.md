# Diversity-enhancing Generative Network
Hi, this is the core code of our recent work "Diversity-enhancing Generative Network for Few-shot Hypothesis Adaptation" (ICML2023, <https://arxiv.org/abs/2307.05948>). This work is done by

- Ruijiang Dong (HKBU),<dongruijiang@pku.edu.cn>
- Haoang Chi (NUDT), <haoangchi618@gmail.com>
- Dr. Feng Liu (UOM), <feng.liu1@unimelb.edu.au>
- Dr. Tongliang Liu (USYD), <tongliang.liu@sydney.edu.au>
- Dr. Mingming Gong (UOM), <mingming.gong@unimelb.edu.au>
- Dr. Gang Niu (RIKEN), <gang.niu.ml@gmail.com>
- Dr. Masashi Sugiyama (RIKEN), <sugi@k.u-tokyo.ac.jp>
- Dr. Bo Han (HKBU), <bhanml@comp.hkbu.edu.hk>

# Software version
Torch version is 1.7.1. Python version is 3.7.6. CUDA version is 11.0.

These python files, of cause, require some basic scientific computing python packages, e.g., numpy. I recommend users to install python via Anaconda (python 3.7.6), which can be downloaded from <https://www.anaconda.com/distribution/#download-section> . If you have installed Anaconda, then you do not need to worry about these basic packages.

After you install anaconda and pytorch (gpu), you can run codes successfully.

# Dataset
Please manually download the datasets [VisDA-C](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) from the official websites, and modify the path of images in each '.txt' under the folder './object/data/'. 

# DEG-Net


Please feel free to test the TOHAN method by running train_source.py and train_target.py.

Specifically, 
Train model on the source domain
```
cd object/
CUDA_VISIBLE_DEVICES=0 python train_source.py
```
Adaptation to other target domains
```
CUDA_VISIBLE_DEVICES=0 python train_target.py
```
in your terminal (using the first GPU device).

It can be seen that test power on the target task will be significantly improved after introducing related tasks.
We will also update our pre-trained models soon.

# Citation
If you are using this code for your own researching, please consider citing
```
@inproceedings{dong2023diversity,
  title={Diversity-enhancing Generative Network for Few-shot Hypothesis Adaptation},
  author={Dong, Ruijiang and Liu, Feng and Chi, Haoang and Liu, Tongliang and Gong, Mingming and Niu, Gang and Sugiyama, Masashi and Han, Bo},
  booktitle={ICML},
  year={2023}
}
```
# Acknowledgment
RJD and BH were supported by NSFC Young Scientists Fund No. 62006202 and Guangdong Basic and Applied Basic Research Foundation No. 2022A1515011652, and HKBU CSD Departmental Incentive Grant. BH was also supported by RIKEN Collaborative Research Fund. MS was supported by JST CREST Grant Number JPMJCR18A2.
