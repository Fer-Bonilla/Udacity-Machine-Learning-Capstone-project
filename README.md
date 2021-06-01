# Machine Learning Capstone project

Capstone projetc for the Udacity's Machine Learning Nanodegree

Project sections:

- Problem understanding
- Data used
- Pytorch Model Architecture
- Implementation
- Project structure
- Model performance
- Using the model

## Problem understanding

The project goal is to build a Dog Breeds Classifier app that uses an image classification deep learning model to perform the Dog breed identification. To achieve this goal is necessary review different models based on CNN architectures (Convolutional Neuronal Networks) mainly. The project development includes to try models built from scratch and model using transfer learning approach on pretrained image classification architectures. The main tool used to develop the project is Pytorch and the models available in the torchvision.models library. 

## Project requirements

Project start with a data exploration to review the image data organization and classes defined. Data formats need to be transformed to fit the pretrained model’s requirements building data loaders function that process de images and add data augmentation to improve the training process. Different libraries and frameworks are suggested to use for the task required in the app. Face detection, Dog’s detection, and Dog Breed detection. The app to be developed must me perform these tasks:

 - If a dog is detected in the image, return the predicted breed.
 - If a human is detected in the image, return the resembling dog breed.
 - If neither is detected in the image, provide output that indicates an error.

Some models are suggested by the Udacity Team, but other models will be considered and tested to achieve best app performance.



## Data used

Dataset used for training and testing are 

dog dataset: 
human_dataset:

Datasets ara avaible for donwload from:





## Pytorch Models Architecture

Test model



Models built


### Base model (Small)

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 112, 112]             896
         MaxPool2d-2           [-1, 32, 56, 56]               0
            Conv2d-3           [-1, 64, 28, 28]          18,496
         MaxPool2d-4           [-1, 64, 14, 14]               0
            Conv2d-5           [-1, 32, 14, 14]          18,464
         MaxPool2d-6             [-1, 32, 7, 7]               0
           Dropout-7                 [-1, 1568]               0
            Linear-8                  [-1, 512]         803,328
       BatchNorm1d-9                  [-1, 512]           1,024
          Dropout-10                  [-1, 512]               0
           Linear-11                  [-1, 133]          68,229
================================================================
Total params: 910,437
Trainable params: 910,437
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 4.39
Params size (MB): 3.47
Estimated Total Size (MB): 8.44
----------------------------------------------------------------

### Medium size model

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1        [-1, 128, 112, 112]           3,584
       BatchNorm2d-2        [-1, 128, 112, 112]             256
         MaxPool2d-3          [-1, 128, 56, 56]               0
            Conv2d-4           [-1, 64, 56, 56]          73,792
       BatchNorm2d-5           [-1, 64, 56, 56]             128
            Conv2d-6           [-1, 32, 56, 56]          18,464
         MaxPool2d-7           [-1, 32, 28, 28]               0
            Conv2d-8           [-1, 64, 28, 28]          18,496
         MaxPool2d-9           [-1, 64, 14, 14]               0
           Conv2d-10           [-1, 32, 14, 14]          18,464
        MaxPool2d-11             [-1, 32, 7, 7]               0
          Dropout-12                 [-1, 1568]               0
           Linear-13                  [-1, 784]       1,230,096
          Dropout-14                  [-1, 784]               0
           Linear-15                  [-1, 382]         299,870
          Dropout-16                  [-1, 382]               0
           Linear-17                  [-1, 133]          50,939
      BatchNorm1d-18                  [-1, 133]             266
================================================================
Total params: 1,714,355
Trainable params: 1,714,355
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 32.15
Params size (MB): 6.54
Estimated Total Size (MB): 39.27
----------------------------------------------------------------

### Big size model

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 224, 224]           1,792
       BatchNorm2d-2         [-1, 64, 224, 224]             128
            Conv2d-3        [-1, 128, 224, 224]          73,856
       BatchNorm2d-4        [-1, 128, 224, 224]             256
            Conv2d-5        [-1, 128, 224, 224]         147,584
         MaxPool2d-6        [-1, 128, 112, 112]               0
            Conv2d-7        [-1, 256, 112, 112]         295,168
       BatchNorm2d-8        [-1, 256, 112, 112]             512
            Conv2d-9        [-1, 256, 112, 112]         590,080
        MaxPool2d-10          [-1, 256, 56, 56]               0
           Conv2d-11          [-1, 128, 56, 56]         295,040
        MaxPool2d-12          [-1, 128, 28, 28]               0
           Conv2d-13           [-1, 64, 28, 28]          73,792
        MaxPool2d-14           [-1, 64, 14, 14]               0
           Conv2d-15           [-1, 32, 14, 14]          18,464
        MaxPool2d-16             [-1, 32, 7, 7]               0
           Linear-17                  [-1, 784]       1,230,096
          Dropout-18                  [-1, 784]               0
           Linear-19                  [-1, 392]         307,720
          Dropout-20                  [-1, 392]               0
           Linear-21                  [-1, 133]          52,269
      BatchNorm1d-22                  [-1, 133]             266
================================================================
Total params: 3,087,023
Trainable params: 3,087,023
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 292.26
Params size (MB): 11.78
Estimated Total Size (MB): 304.61
----------------------------------------------------------------

### Results


|    Model    | Test Loss | Parameters number | Epochs | Test Accuracy |
|:-----------:|:---------:|:-----------------:|--------|:-------------:|
| Small size  |  3.232469 |           246,799 |   50   | 21% (183/836) |
| Medium size |  3.131659 |         1,714,355 |   50   | 25% (215/836) |
| Big size    |  3.917502 |         3,087,023 |   20   | 11% ( 94/836) |



Transfer Learning Model:


![Pytorch model](https://github.com/Fer-Bonilla/Udacity-Machine-Learning-plagiarism-detection-model/blob/main/notebook_ims/network_architecture.png)


## Implementation

**Pytorch BinaryClassifier Model**

  ```Python
import torch
import torch.nn.functional as F
import torch.nn as nn

## TODO: Complete this classifier
class BinaryClassifier(nn.Module):
    """
    Define a neural network that performs binary classification.
    The network should accept your number of features as input, and produce 
    a single sigmoid value, that can be rounded to a label: 0 or 1, as output.
    
    Notes on training:
    To train a binary classifier in PyTorch, use BCELoss.
    BCELoss is binary cross entropy loss, documentation: https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss
    """

    ## TODO: Define the init function, the input params are required (for loading code in train.py to work)
    def __init__(self, input_features, hidden_dim, output_dim):
        """
        Initialize the model by setting up linear layers.
        Use the input parameters to help define the layers of your model.
        :param input_features: the number of input features in your training/test data
        :param hidden_dim: helps define the number of nodes in the hidden layer(s)
        :param output_dim: the number of outputs you want to produce
        """
        super(BinaryClassifier, self).__init__()

        # define any initial layers, here
        
        # Add a fully connected layer
        self.fc1 = nn.Linear(input_features, hidden_dim)
        
        # Add a fully connected layer  
        self.fc2 = nn.Linear(hidden_dim, int(hidden_dim/2))
        
        # Add a fully connected layer
        self.fc3 = nn.Linear(int(hidden_dim/2), output_dim)
        
        # Add dropout to avoid overfitting
        self.drop = nn.Dropout(0.25)
        
        # add a  element-wise function sigmoid
        self.sigmoid = nn.Sigmoid()

    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        """
        Perform a forward pass of our model on input features, x.
        :param x: A batch of input features of size (batch_size, input_features)
        :return: A single, sigmoid-activated value as output
        """
        
        # define the feedforward behavior
        
        # Add a fully connected layer with Relu activation
        x = torch.relu(self.fc1(x))
        
        # Add a dropout to avoid overfitting
        x = self.drop(x)
        
        # Add a fully connected layer with Relu activation function
        x = torch.relu(self.fc2(x))
        
        # Generate single, sigmoid-activated value as output
        x = torch.sigmoid(self.fc3(x))
        
        return x
  ```

**Pytorch training function**


## Model performance

The project structure is based on the Udacity's project template:

Training Loss 
```
Epoch: 10, Loss: 0.6487539325441632
Epoch: 20, Loss: 0.613213564668383
Epoch: 30, Loss: 0.5324599317141941
Epoch: 40, Loss: 0.4539638161659241
Epoch: 50, Loss: 0.3536339061600821
Epoch: 60, Loss: 0.301504128745624
Epoch: 70, Loss: 0.28896574676036835
Epoch: 80, Loss: 0.2756422853895596
Epoch: 90, Loss: 0.2657772238765444
Epoch: 100, Loss: 0.22072330649409974
```

Accuracy
  ```
  1.0
  ```

## App model



## Using the model

  1. Execute the 2_Plagiarism_Feature_Engineering.ipynb (Load the data and write into plagirims_data directory to create train and test data)
  2. Execute 3_Training_a_Model.ipynb script to train and deploy the model
  3. Delete all the resources from Sagemaker

## Author 
Fernando Bonilla [linkedin](https://www.linkedin.com/in/fer-bonilla/)
