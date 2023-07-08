# Brain Tumor Classification with CNN

This repository contains code for training a Convolutional Neural Network (CNN) model to classify brain tumor images into three categories: benign, malign, and normal. The model is built using Keras library and trained on a dataset containing images of brain tumors.

## Requirements

- Python 3.x
- Keras
- TensorFlow
- NumPy

## Dataset

The dataset used for training the model should be organized into three folders: `benign`, `malign`, and `normal`. These folders should be placed in the same directory as the script file. Each folder should contain the respective category of brain tumor images.

## Model Architecture

The CNN model architecture consists of several layers, including convolutional layers, max pooling layers, batch normalization layers, dropout layers, and dense layers. Here is an overview of the model's architecture:

```
Input -> Conv2D -> MaxPooling2D -> BatchNormalization -> Conv2D -> MaxPooling2D -> BatchNormalization -> Conv2D -> MaxPooling2D -> BatchNormalization -> Conv2D -> MaxPooling2D -> BatchNormalization -> Dropout -> Flatten -> Dense -> Dropout -> Dense -> Output
```

The model is compiled with the Adam optimizer, categorical cross-entropy loss function, and accuracy as the evaluation metric.

## Data Augmentation

To improve the model's performance and generalization, data augmentation is applied to the training data. The `ImageDataGenerator` class from Keras is used to perform data augmentation. The following augmentations are applied during training:

- Rescaling the pixel values to a range of 0 to 1
- Random shear transformation
- Random zooming
- Horizontal flipping

## Usage

1. Ensure that the required dependencies are installed.
2. Organize your dataset into three folders: `benign`, `malign`, and `normal`. Place these folders in the same directory as the script file.
3. Run the script. The model will be trained using the provided dataset and saved to disk.
4. The trained model will be saved in two files: `model.json` (contains the model architecture) and `model.h5` (contains the model weights).

## Training

During training, the model is trained using the data augmentation applied to the training dataset. The training data is loaded using the `flow_from_directory` function, which automatically detects the class labels based on the folder structure. The model is trained for a fixed number of epochs, with a specified batch size.

## Saving the Model

After training, the model is saved to disk in two files: `model.json` and `model.h5`. The JSON file contains the model architecture, while the H5 file contains the model weights. These files can be later loaded to make predictions or further fine-tune the model.

## Example Dataset Structure

The dataset structure should follow the example below:

```
- TrainingDataset
  - benign
    - benign_image1.jpg
    - benign_image2.jpg
    - ...
  - malign
    - malign_image1.jpg
    - malign_image2.jpg
    - ...
  - normal
    - normal_image1.jpg
    - normal_image2.jpg
    - ...
```

## License

This project is licensed under the Apache License 

## Acknowledgments

- The code in this repository was inspired by various examples and tutorials on CNN image classification.
- The dataset used for training the model is not provided here, but you can use your own dataset with the appropriate folder structure.

Feel free to contribute to this project by creating pull requests or opening issues.