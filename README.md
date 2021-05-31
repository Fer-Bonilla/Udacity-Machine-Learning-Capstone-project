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

Base model (Small)

Medium model

Big model


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
