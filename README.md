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

| Layer (type)                                                     | Output Shape       | Param #   |
|------------------------------------------------------------------|:-------------------|:----------|
| Conv2d-1                                                         | [-1, 32, 112, 112] |       896 |
| MaxPool2d-2                                                      | [-1, 32, 56, 56]   |         0 |
| Conv2d-3                                                         | [-1, 64, 28, 28]   |    18,496 |
| MaxPool2d-4                                                      | [-1, 64, 14, 14]   |         0 |
| Conv2d-5                                                         | [-1, 32, 14, 14]   |    18,464 |
| MaxPool2d-6                                                      | [-1, 32, 7, 7]     |         0 |
| Dropout-7                                                        | [-1, 1568]         |         0 |
| Linear-8                                                         | [-1, 133]          |   208,677 |
| BatchNorm1d-9                                                    | [-1, 133]          |       266 |


 - Total params: 910,437
 - Trainable params: 910,437
 - Non-trainable params: 0

### Medium size model

| Layer (type)                                                     | Output Shape        | Param #   |
|------------------------------------------------------------------|:--------------------|:----------|
| Conv2d-1                                                         | [-1, 128, 112, 112] |     3,584 |
| BatchNorm2d-2                                                    | [-1, 128, 112, 112] |       256 |
| MaxPool2d-3                                                      | [-1, 128, 56, 56]   |         0 |
| Conv2d-4                                                         | [-1, 64, 56, 56]    |    73,792 |
| BatchNorm2d-5                                                    | [-1, 64, 56, 56]    |       128 |
| Conv2d-6                                                         | [-1, 32, 56, 56]    |    18,464 |
| MaxPool2d-7                                                      | [-1, 32, 28, 28]    |         0 |
| Conv2d-8                                                         | [-1, 64, 28, 28]    |    18,496 |
| MaxPool2d-9                                                      | [-1, 64, 14, 14]    | 0         |
| Conv2d-10                                                        | [-1, 32, 14, 14]    | 18,464    |
| MaxPool2d-11                                                     | [-1, 32, 7, 7]      | 0         |
| Dropout-12                                                       | [-1, 1568]          | 0         |
| Linear-13                                                        | [-1, 784]           | 1,230,096 |
| Dropout-14                                                       | [-1, 784]           | 0         |
| Linear-15                                                        | [-1, 382]           | 299,870   |
| Dropout-16                                                       | [-1, 382]           | 0         |
| Linear-17                                                        | [-1, 133]           | 50,939    |
| BatchNorm1d-18                                                   | [-1, 133]           | 266       |


 - Total params: 1,714,355
 - Trainable params: 1,714,355
 - Non-trainable params: 0

### Big size model

| Layer (type)                                                     | Output Shape        | Param #   |
|------------------------------------------------------------------|---------------------|-----------|
| Conv2d-1                                                         | [-1, 64, 224, 224]  | 1,792     |
| BatchNorm2d-2                                                    | [-1, 64, 224, 224]  | 128       |
| Conv2d-3                                                         | [-1, 128, 224, 224] | 73,856    |
| BatchNorm2d-4                                                    | [-1, 128, 224, 224] | 256       |
| Conv2d-5                                                         | [-1, 128, 224, 224] | 147,584   |
| MaxPool2d-6                                                      | [-1, 128, 112, 112] | 0         |
| Conv2d-7                                                         | [-1, 256, 112, 112] | 295,168   |
| BatchNorm2d-8                                                    | [-1, 256, 112, 112] | 512       |
| Conv2d-9                                                         | [-1, 256, 112, 112] | 590,080   |
| MaxPool2d-10                                                     | [-1, 256, 56, 56]   | 0         |
| Conv2d-11                                                        | [-1, 128, 56, 56]   | 295,040   |
| MaxPool2d-12                                                     | [-1, 128, 28, 28]   | 0         |
| Conv2d-13                                                        | [-1, 64, 28, 28]    | 73,792    |
| MaxPool2d-14                                                     | [-1, 64, 14, 14]    | 0         |
| Conv2d-15                                                        | [-1, 32, 14, 14]    | 18,464    |
| MaxPool2d-16                                                     | [-1, 32, 7, 7]      | 0         |
| Linear-17                                                        | [-1, 784]           | 1,230,096 |
| Dropout-18                                                       | [-1, 784]           | 0         |
| Linear-19                                                        | [-1, 392]           | 307,720   |
| Dropout-20                                                       | [-1, 392]           | 0         |
| Linear-21                                                        | [-1, 133]           | 52,269    |
| BatchNorm1d-22                                                   | [-1, 133]           | 266       |
 
 
 - Total params: 3,087,023
 - Trainable params: 3,087,023
 - Non-trainable params: 0

### Results


|    Model    | Test Loss | Parameters number | Epochs | Test Accuracy |
|:-----------:|:---------:|:-----------------:|--------|:-------------:|
| Small size  |  3.232469 |           246,799 |   50   | 21% (183/836) |
| Medium size |  3.131659 |         1,714,355 |   50   | 25% (215/836) |
| Big size    |  3.917502 |         3,087,023 |   20   | 11% ( 94/836) |



## Transfer Learning Model:
resnet152

- Total params: 59,261,125
 - Trainable params: 59,261,125
 - Non-trainable params: 0


![Pytorch model](https://github.com/Fer-Bonilla/Udacity-Machine-Learning-Capstone-project/blob/main/resnet152.png)


## Implementation

**Pytorch BinaryClassifier Model**




**Pytorch training function**


## Model performance

The model trained using transfer learning with resnet152 pre-trained model achieved around 85% accuracy. I tried with some additional images, for example, two ShihZu images, and the model doesn't work well. and with dogs with mixed bred the model was confused. That means that there is a lot of improvement because 85% accuracy sounds ok but in practice still far from a good performance. Is clear that using a pre-trained model offers a fast and better way to train models for a specific task like dog breed detection. As I suggested when chose the pre-trained model, There more state-of-the-art models with more than 600 million parameters.

Model performance:

Test Loss: 0.471680
Test Accuracy: 86% (723/836)

## App model

  ```Python 
  def run_app(img_path):

      '''
      Define a function to execute inference process to detect faces and fogd bred
      predicted ImageNet class for image at specified path

      Args:
          img_path: path to an image

      Returns:
          Index corresponding to RESNET-120 model's prediction
      '''    

      ## handle cases for a human face, dog, and neither

      # First look for human face 
      if face_detector(img_path):
          print("Face Detected")
          image = Image.open(img_path)
          plt.imshow(image)
          plt.show()
          print(f"Looks like: \n The {predict_breed_transfer(img_path)}\n")

      # If not face detectec check for dog breed
      elif dog_detector(img_path):
          print("Dog Detected")
          image = Image.open(img_path)
          plt.imshow(image)
          plt.show()
          print(f"Predicted breed:... \n{predict_breed_transfer(img_path)}\n")

      # Not face or dog detected
      else:
          print("Couldn't detect dogs or faces.\n")
          image = Image.open(img_path)
          plt.imshow(image)
          plt.show()
  ```

## Using the model

  1. Execute the eda.ipynb to visualize the Exploratory Data Analysis
  2. Execute dog_app.ipynb for all the training and testis model.

## Possible improvements

 - As was reviewed before, state-of-the-art models offer higher accuracy for the image classification tasks, can also provide better performance for the transfer learning task. These models require more resources for the training process.
 - Aditional optimizations can be used to improve performance, like hyperparameter tuning and network pruning to accelerate inference time response.
 - Increasing data with more sources using web scrapping and data augmentation using GAN networks to enrich the training process can provide better input for the training and validation process.
 - New models based on a generative approach can be used to improve the transfer learning process.

## Author 
Fernando Bonilla [linkedin](https://www.linkedin.com/in/fer-bonilla/)
