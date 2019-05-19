# CycleGAN for transfering characteristics from a set of images to a new image


This is the project for the subject CS-E4890 - Deep Learning, while I was doing my exchange program at Aalto University.
It was in [this paper](https://arxiv.org/pdf/1703.10593.pdf).

## Getting started

### Installing

- Clone this repo:
```
git clone https://github.com/davicorreiajr/cyclegan.git
cd cyclegan
```
- Install the dependencies:
```
pip install -r requirements.txt
```

### Training

- You need to choose or create your dataset.
There is a list of dataset availables in `./scripts/download_dataset.sh`; let's say you've chosen `vangogh2photo`.

- Then, you need to create the folders:
```
mkdir /datasets
mkdir /datasets/vangogh2photo
mkdir /checkpoints
mkdir /checkpoints/cyclegan_simple
```

- And then download the dataset:
```
bash ./scripts/download_dataset.sh vangogh2photo
```

- Finally, run the training script:
```
python train.py
```
Also in the `train.py` file you can find and change the training options.

### Testing

- Create the folders:
```
mkdir results
mkdir results/cyclegan_simple
mkdir datasets/test
```

- Upload the pictures you want to apply the style to `./datasets/test`;

- Run the test script:
```
python test.py
```
Also in the `test.py` file you can find and change the testing options.

### Running in Google Colab

Since I didn't have GPUs, I needed to use Google Colab. If you also want want to use, you can copy the notebook `colab.ipynb` and run on Colab. It is going to clone the repository and it already has the cells with the folder creations and also with the training/testing commands.

## Folder structure

- `/data`: responsible of loading the images; train will use `both_directions_dataset.py` and test will use `single_direction_dataset.py`. They basically get the paths to the images and load them when requested by the models;
- `/datasets`: where the datasets from training and testing are. For training, your dataset should have `trainA` and `trainB` folder, where the desired target should be in `trainB`;
- `/models`: it contains the files responsible for the construction of the generators and discriminators. The `cycle_gan_model.py` has the general architecture of the model, while `networks.py` takes care of the implements of its architecture;
- `/scripts`: bash files to download pretrained models (from the paper) and datasets;
- `/util`: some helper methods.

## Copyright & credits

This project was based in the paper `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks`.


## Troubles & suggestions

Please, if you find any problem or have some sugestion, don't hesitate to open an issue or even a pull request.
