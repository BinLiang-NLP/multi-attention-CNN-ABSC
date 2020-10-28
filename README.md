# multi-channels-CNN-ABSC

# Introduction
The core code of our proposed method for refining target embeddings.

This repository was used in our paper:  
  
**“Context-aware Embedding for Targeted Aspect-based Sentiment Analysis”**  
Bin Liang, Jiachen Du, Ruifeng Xu<sup>*</sup>, Binyang Li, Hejiao Huang. *Proceedings of ACL 2019*
  
Please cite our paper and kindly give a star for this repository if you use this code. 

## Requirements

* Python 3.6 / 3.7
* numpy >= 1.13.3
* PyTorch >= 1.0.0

## Usage

### Training
* Train with command, optional arguments could be found in [train.py](/train.py)
* Run refining target: ```./run.sh```


## Model

An overall architecture of our proposed framework is as follow:

<img src="/assets/model.png" width = "40%" />

## Citation

The BibTex of the citation is as follow:

```bibtex
@article{bin2017aspect,
  title={Aspect-based sentiment analysis based on multi-attention CNN},
  author={Bin, Liang and Quan, Liu and Jin, Xu and Qian, Zhou and Peng, Zhang},
  journal={Journal of Computer Research and Development},
  volume={54},
  number={8},
  pages={1724},
  year={2017}
}
```


