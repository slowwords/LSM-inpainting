# LSM-inpainting
<img src="./imgs/CelebA.png" alt="result" style="zoom:80%;" />

**Image inpainting based on localized step modulation**<br>

_Zhiwen Wang_<br>

## Introduction

The PyTorch implementation of my personal experiment: image inpainting based on localized step modulation.

### Prerequistes

- Python >= 3.6
- PyTorch >= 1.0

## Getting Started

### Installation

- Clone this repository:

```
git clone https://github.com/slowwords/LSM-inpainting.git
cd LSM-inpainting
```

- Install PyTorch and dependencies from [http://pytorch.org](http://pytorch.org/)
- Install python requirements:

```
pip install -r requirements.txt
```
### Datasets

**Image Dataset.** We evaluate the proposed method on the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [Paris StreetView](https://github.com/pathak22/context-encoder) datasets, which are widely adopted in the literature.

### Training

To train the model, you run the following code.

```
python train.py \
  --image_path [path to image directory] \
  --mask_mode [comod_mask or user_mask, if user_mask, mask_path should be given] \
  --data_mode [centercrop or resize] \
  --datasets_name
```

### Testing

To test the model, you run the following code.

```
python test.py \
  --pre_trained [path to checkpoints] \
  --image_path [path to image directory] \
  --mask_mode [comod_mask or user_mask, if user_mask, mask_path should be given] \
  --test_iters [iters for testing] \
  --result_path [path to results]
```

## Citation

This work has not yet been published, so stay tuned for our other work if possible!

[**Paper**](https://link.springer.com/article/10.1007/s00371-023-03045-z)

```
@article{Wang_2023_TVCJ,
  title={Dynamic context-driven progressive image inpainting with auxiliary generative units},
  author={Wang, Zhiwen and Li, Kai and Peng, Jinjia},
  journal={The Visual Computer},
  pages={1--16},
  year={2023},
  publisher={Springer}
}
```
