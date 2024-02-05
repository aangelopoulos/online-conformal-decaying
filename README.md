# Online Conformal Prediction with Decaying Step Sizes
Anastasios N. Angelopoulos, Rina Foygel Barber, and Stephen Bates
<p align="center">
    <a style="text-decoration:none !important;" href="https://arxiv.org/abs/2402.01139" alt="arXiv"><img src="https://img.shields.io/badge/paper-arXiv-red" /></a>
    <a style="text-decoration:none !important;" href="https://opensource.org/licenses/MIT" alt="License"><img src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
    <a style="text-decoration:none !important;" href="http://hits.dwyl.com/aangelopoulos/online-conformal-decaying" alt="arXiv"><img src="https://hits.dwyl.com/aangelopoulos/online-conformal-decaying.svg?style=flat-square" /></a>
</p>

This is a lightweight codebase used to reproduce the experiments in our paper.
To run it, you will need the following dependencies:
```
jupyter
numpy
matplotlib
seaborn
pandas
gdown
```

There are two relevant notebooks: `elec2.ipynb` and `imagenet-smallest-sets.ipynb`, which implement the according experiments in our paper.

Our main method is implemented as the only function in `core.py`.
