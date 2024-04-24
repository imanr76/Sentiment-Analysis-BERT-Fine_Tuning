# Fine-tuning a BERT LLM Model for Sentiment Analysis

## 1. Project Description
This project focuses on fine-tuning a BERT model for sentiment analysis of the product reviews. The pretrained model is an Albert model from HuggingFace with pretraining checkpoint of 'albert-base-v1'. 

To perform the fine-tuning, a linear layer is added on top of the Albert model and the pooling layer of the Albert model is deleted. Furthermore, the ... layer of the Albert model is frozen so that they are not updated during trianing. This reduces the computation load needed for traning the model. 

The dataset used is the Amazon product reviews dataset publicly available from the following S3 location: 

"s3://dlai-practical-data-science/data/raw/womens_clothing_ecommerce_reviews.csv"

A sample of the traiend model could be found in the models directory. The trained model showed a 87.1% accuracy on the test set which could be improved by hyperparameter tuning. 

The specific details of the trained model are as follows and the data used could be found in the data directory: 

- Size of the embedding vector for each token : 20

- Number of head layers : 1
- Learning rate for trianing the model : 0.0001
- Number of epochs to run : 30
- Threshold for positive and negative labels : 0.5
- Number of batches to use for each parameter update : 16

## 2. Tech Stack
 - Python
 - Pytorch
 - AWS CLI
 - AWS Python SDK
 - Sagemaker Python SDK

## 3. How to run the project: 
Before running this project. please consider the following points: 
- Install the project packages using the requirements.txt file.
- Make sure you have AWS CLI installed on your machine.
- To process the data, train the model and then make some inferences from the model, run the <b>main.py</b> script from within the src directory. 
<b>NOTE: you must run the main script from within the src directory, many of the scripts use relative paths which could lead to errors</b>

A list of the input parameters could be viewed in the main.py script.

## 4. Project File Struture:

- src: Contains project codes and scripts. 
    - <b>main.py</b>: Preprocesses the data, trains a model, saves the model and then uses the last trained model to make some predictions. 
    - <b>data_preparation.py</b>: Preprocesses the data based on the inputs. 
    - <b>inferecne.py</b>: Could be used to make predictions using the last trained model. 
    - <b>training.py</b>: Script for training the model. 
    
- <b>models</b>: Model artifacts are saved in this directory. Each trained model is saved as a subdirectory. Each subdirectory contains the saved Pytorch model object, Information about the training, such as losses and accuracies during epochs and a JSON file containing the classification report of the model on the test set. 

- <b>data</b>: Contains the raw data and processed data for training and inference of the model. 

- <b>requirements.txt</b>: The requirements file containing the required packages for using the project.    

