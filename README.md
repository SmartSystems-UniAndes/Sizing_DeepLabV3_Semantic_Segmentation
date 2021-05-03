# DeepLabV3 Transfer Learning for PV Panels Segmentation

## About

This repository contains a case study for the work developed by Malof, J. et al. in *Mapping solar array location, size,
and capacity using deep learning and overhead imagery* [1]. Here is shown the transfer learning process for 
photovoltaic panels segmentation with the DeepLab V3 architecture trained for ImageNet.

## Work Environment

See [SETUP.md](SETUP.md).

## Main

To use the repo run the main.py file that has the following arguments:

- **train_dataset**: Path to train dataset (default: 'dataset/train').
- **test_dataset**: Path to test dataset (default: 'dataset/test').
- **train_model**: Set True to train a new model (default: False).
- **im_resize**: Pixels to resize the images (default: 500).
- **batch_size**: Batch size for training process (default: 8).
- **backbone**: Model backbone (default: 'RESNET101').
- **optimizer**: Optimizer for training process (default: 'Adam').
- **lr**: Learning rate for training process (default: 0.001).
- **epochs**: Number of epochs for training process (default: 50).
- **trained_model**: Path to trained model (default: 'trained_models/deeplab_v3_RESNET101_model.pt').
- **output_results**: Path to save the inference outputs (default: 'results/').

Example:

```sh
$ python main.py --train False
```

## Citing Work

```BibTeX
@article{malof2019mapping,
  title={Mapping solar array location, size, and capacity using deep learning and overhead imagery},
  author={Malof, Jordan M and Li, Boning and Huang, Bohao and Bradbury, Kyle and Stretslov, Artem},
  journal={arXiv preprint arXiv:1902.10895},
  year={2019}
}
```

## References

[1] Malof, J. M., Li, B., Huang, B., Bradbury, K., & Stretslov, A. (2019).
[Mapping solar array location, size, and capacity using deep learning and overhead imagery](https://arxiv.org/ftp/arxiv/papers/1902/1902.10895.pdf). arXiv preprint arXiv:1902.10895.