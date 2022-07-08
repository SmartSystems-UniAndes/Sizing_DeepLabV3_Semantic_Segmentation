# DeepLabV3 Transfer Learning for PV Panels Segmentation

## About

This repository contains a case study for the work developed by Malof, J. et al. in *Mapping solar array location, size,
and capacity using deep learning and overhead imagery* [1]. Here is shown the transfer learning process for 
photovoltaic panels segmentation with the DeepLab V3 architecture trained for ImageNet.

## How it works?

Run the file main.py. If you want tho check the transfer learning process, see the segmentation_model/deep_lab_v3.py file.
To download the dataset and the trained model, go to this link: https://drive.google.com/drive/folders/1HF9cHprGdPsjziL-NKyU9dU2R6fUEUGu?usp=sharing

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
@article{gaviria_machine_2022,
	title = {Machine learning in photovoltaic systems: A review},
	issn = {0960-1481},
	url = {https://www.sciencedirect.com/science/article/pii/S0960148122009454},
	doi = {10.1016/j.renene.2022.06.105},
	shorttitle = {Machine learning in photovoltaic systems},
	abstract = {This paper presents a review of up-to-date Machine Learning ({ML}) techniques applied to photovoltaic ({PV}) systems, with a special focus on deep learning. It examines the use of {ML} applied to control, islanding detection, management, fault detection and diagnosis, forecasting irradiance and power generation, sizing, and site adaptation in {PV} systems. The contribution of this work is three fold: first, we review more than 100 research articles, most of them from the last five years, that applied state-of-the-art {ML} techniques in {PV} systems; second, we review resources where researchers can find open data-sets, source code, and simulation environments that can be used to test {ML} algorithms; third, we provide a case study for each of one of the topics with open-source code and data to facilitate researchers interested in learning about these topics to introduce themselves to implementations of up-to-date {ML} techniques applied to {PV} systems. Also, we provide some directions, insights, and possibilities for future development.},
	journaltitle = {Renewable Energy},
	shortjournal = {Renewable Energy},
	author = {Gaviria, Jorge Felipe and Narváez, Gabriel and Guillen, Camilo and Giraldo, Luis Felipe and Bressan, Michael},
	urldate = {2022-07-03},
	date = {2022-07-01},
	langid = {english},
	keywords = {Deep learning, Machine learning, Neural networks, Photovoltaic systems, Reinforcement learning, Review},
	file = {ScienceDirect Snapshot:C\:\\Users\\jfgf1\\Zotero\\storage\\G96H46L2\\S0960148122009454.html:text/html},
},

@article{malof2019mapping,
  title={Mapping solar array location, size, and capacity using deep learning and overhead imagery},
  author={Malof, Jordan M and Li, Boning and Huang, Bohao and Bradbury, Kyle and Stretslov, Artem},
  journal={arXiv preprint arXiv:1902.10895},
  year={2019}
}
```

## References
[1] Jorge Felipe Gaviria, Gabriel Narváez, Camilo Guillen, Luis Felipe Giraldo, and Michael Bressan. Machine learning in photovoltaic systems: A review. ISSN 0960-1481. doi: 10.1016/j.renene.2022.06.105. URL https://www.sciencedirect.com/science/article/pii/S0960148122009454?via%3Dihub

[2] Malof, J. M., Li, B., Huang, B., Bradbury, K., & Stretslov, A. (2019).
[Mapping solar array location, size, and capacity using deep learning and overhead imagery](https://arxiv.org/ftp/arxiv/papers/1902/1902.10895.pdf). arXiv preprint arXiv:1902.10895.

## Licenses

### Software
The software is licensed under an [MIT License](https://opensource.org/licenses/MIT). A copy of the license has been included in the repository and can be found [here](https://github.com/SmartSystems-UniAndes/PV_MPPT_Control_Based_on_Reinforcement_Learning/blob/main/LICENSE-MIT.txt).
