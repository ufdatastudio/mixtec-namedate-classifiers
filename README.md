
# MIXTEC-CLASSIFIERS

## Prerequisites

Before you begin, make sure that you have the following installed:

- Python (preferably version 3.7 or higher)
- Git
- GPU drivers (if using GPU for training)
- Huggingface credentials (for dataset access)
- Git lfs: You might need this to ensure that the images downloaded from huggingface are not just references but actual files.

## Steps

### 1. Setting up the Image Dataset

To clone the dataset from Huggingface, use the following command. You will be prompted to provide your Huggingface credentials:

```bash
./setup_dataset.sh
```

This will download the dataset to your local system.

### 2. Setting up the Python Environment

To set up the required Python environment, use the provided `environment.yml` file. Note that you may need to adjust the file for the GPU version of PyTorch installed on your system. You can modify the `environment.yml` to match your installed CUDA version if necessary.

To create the environment, run:

```bash
conda env create -f environment.yml
```

Once the environment is created, activate it with:

```bash
conda activate name-date
```

### 3. Perform Data Augmentation

Once the environment is set up, you can perform data augmentation on the images in the dataset. To do so, run the `augment_images.py` script and specify the folder path to your images dataset as a command-line argument:

```bash
python augment_images.py <path_to_image_dataset>
```

This will perform data augmentation on the images and save the results in the same directory.

### 4. Train the Model

To train the classification model on the augmented dataset, run the `classification.py` script:

```bash
python classification.py
```

This script will:

- Train the models on the image dataset.
- Save the trained models in the `models` folder.
- Export accuracy and loss curves.
- Write precision, recall, and F1 score to a text file.

## Sample Commands

Here is a summary of the commands you can run:

1. Set up the dataset:

```bash
./setup_dataset.sh
```

2. Create and activate the Python environment:

```bash
conda env create -f environment.yml
conda activate <your_environment_name>
```

3. Perform data augmentation:

```bash
python augment_images.py <path_to_your_image_dataset>
```

4. Train the model:

```bash
python classification.py
```
