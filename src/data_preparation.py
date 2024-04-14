import transformers 
from transformers import AlbertTokenizer
import pandas as pd
import torch
from torch.utils.data import Dataset
import subprocess
import os
from sklearn.model_selection import train_test_split

#------------------------------------------------------------------------------
# Function Definitions
def rating_to_sentiment(rating):
    """
    Parameters
    ----------
    rating : int
        The star rating of the item from 1 to 5.

    Returns
    -------
    sentiment : int
        Implied sentiment of the star rating, assumes ratings between 1 and 3 (inclusive) to be
        negative (0) and rating more than 3 to be positive (1).
    """
    
    if rating in {1, 2, 3}:
        return 0
    else:
        return 1

class dataset(Dataset):
    """
    Pytorch Dataset class for converting the pandas dataframes into datasets
    """
    def __init__(self,data):
        self.data = data
        self.length = (len(self.data[0]) - 1) // 2
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        sentiment = self.data[index, -1]
        input_ids = self.data[index, :self.length]
        attention_mask = self.data[index, self.length:-1]
        return input_ids, attention_mask, sentiment


def process_data(max_len = 500, train_size = 0.8, validation_size = 0.15, test_size = 0.05):
    """
    Downloads the data from the S# bucket on AWS, transforms it, balances it, creates a vocbulary from the
    review texts, converts the text into sequences of indices, divides the data into 
    training, validation and test sets and save them as PyTorch datsets. 

    Parameters
    ----------
    max_len : int, optional
        Maximum review text sequence length. The default is 500.
    train_size : int, optional
        Fraction of training data of all data. The default is 0.8.
    validation_size : int, optional
        Fraction of validation data of all data. The default is 0.15.
    test_size : int, optional
        Fraction of test data of all data. The default is 0.05.

    Returns
    -------
    None.

    """
    #------------------------------------------------------------------------------
    # Reading and transforming the dataset
    
    # Creating the necessary directories and downloading the data
    if 'data' not in os.listdir("./.."):
        os.mkdir("./../data")
    
    if 'raw_data' not in os.listdir("./../data"):
        os.mkdir("./../data/raw_data")
    
    if "womens_clothing_ecommerce_reviews.csv" not in os.listdir("./../data/raw_data"):
        subprocess.run("aws s3 cp s3://dlai-practical-data-science/data/raw/womens_clothing_ecommerce_reviews.csv ./../data/raw_data",\
                       shell=True)
    
    
    # Reading the data
    data = pd.read_csv("./../data/raw_data/womens_clothing_ecommerce_reviews.csv")
    # data = data.sample(frac = 0.6)
    # Keeping the useful columns
    data_transformed =  data[["Review Text", "Rating", "Class Name"]].copy()
    # Renaming the columns for convenience
    data_transformed.rename(columns = {"Review Text":'review', "Rating":"rating", "Class Name":"product_category"}, inplace = True)
    # dropping the rows wth empty cells 
    data_transformed.dropna(inplace = True)
    # Removing the data for product categories with less than 10 reviews
    data_transformed  = data_transformed.groupby("product_category").filter(lambda review: len(review) > 10)
    # Converting the star rating to sentiment and dropping the rating column as it is not needed anymore
    data_transformed["sentiment"] = data_transformed["rating"].apply(lambda rating: rating_to_sentiment(rating))
    data_transformed.drop(columns = "rating", inplace = True)
    # Saving the transformed dataset
    data_transformed.to_csv("./../data/raw_data/womens_clothing_ecommerce_reviews_transformed.csv", index = False)
    
    
    #------------------------------------------------------------------------------
    # Balancing the dataset
    
    # Balancing the dataset based on the sentiments so we have the same number of reviews for both sentiments
    data_transformed_grouped_for_balance = data_transformed.groupby(["sentiment"])[["review","sentiment", "product_category"]]
    data_transformed_balanced = data_transformed_grouped_for_balance.apply(lambda x: \
                                    x.sample(data_transformed.groupby(["sentiment"]).size().min()))\
                                    .reset_index(drop = True)# Saving the balanced dataset
    # Saving the balanced dataset
    data_transformed_balanced.to_csv("./../data/raw_data/womens_clothing_ecommerce_reviews_balanced.csv", index = False)
    
    # Creating the required directories to save the data
    if "training" not in os.listdir("./../data"):
        os.mkdir("./../data/training")
    
    if "validation" not in os.listdir("./../data"):
        os.mkdir("./../data/validation")
    
    if "test" not in os.listdir("./../data"):
        os.mkdir("./../data/test")
    
    # Dividing the data into train, validation and test sets
    training_data, temp_data = train_test_split(data_transformed_balanced, test_size = 1 - train_size, random_state = 10)
    validation_data, test_data = train_test_split(temp_data, test_size = test_size / (validation_size + test_size), random_state = 10)
    
    # Saving the train, validation and test datasets
    training_data.to_csv("./../data/training/womens_clothing_ecommerce_reviews_balanced_training.csv", index = False)
    validation_data.to_csv("./../data/validation/womens_clothing_ecommerce_reviews_balanced_validation", index = False)
    test_data.to_csv("./../data/test/womens_clothing_ecommerce_reviews_balanced_test", index = False)
    #------------------------------------------------------------------------------
    # Preprocessing the data for the NLP task
    

    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1')
    # Tokenizing, padding and truncating the review texts and creating a tensor including the tokenized review text, 
    # attention masks and the sentiment of the review as the last element of the tensor for training set
    training_data_reviews = tokenizer(list(training_data['review'].values), return_tensors="pt", padding = 'max_length', truncation = True, max_length  = max_len)
    training_data_sentiments = torch.tensor(training_data['sentiment'].values).view(-1 ,1)
    training_data_tensor = torch.cat((training_data_reviews["input_ids"], training_data_reviews["attention_mask"], training_data_sentiments), dim = 1)
    # Tokenizing, padding and truncating the review texts and creating a tensor including the tokenized review text, 
    # attention masks and the sentiment of the review as the last element of the tensor for validation set
    validation_data_reviews = tokenizer(list(validation_data['review'].values), return_tensors="pt", padding = 'max_length', truncation = True, max_length  = max_len)
    validation_data_sentiments = torch.tensor(validation_data['sentiment'].values).view(-1 ,1)
    validation_data_tensor = torch.cat((validation_data_reviews["input_ids"], validation_data_reviews["attention_mask"], validation_data_sentiments), dim = 1)
    # Tokenizing, padding and truncating the review texts and creating a tensor including the tokenized review text, 
    # attention masks and the sentiment of the review as the last element of the tensor for test set
    test_data_reviews = tokenizer(list(test_data['review'].values), return_tensors="pt", padding = 'max_length', truncation = True, max_length  = max_len)
    test_data_sentiments = torch.tensor(test_data['sentiment'].values).view(-1 ,1)
    test_data_tensor = torch.cat((test_data_reviews["input_ids"], test_data_reviews["attention_mask"], test_data_sentiments), dim = 1)
    
    
    # Creating torch Datasets
    train_dataset = dataset(training_data_tensor)
    validaton_dataset = dataset(validation_data_tensor)
    test_dataset = dataset(test_data_tensor)
    
    # Saving the torch Datasets for future sse and reference
    torch.save(train_dataset, "./../data/training/training_dataset.pth")
    torch.save(validaton_dataset, "./../data/validation/validation_dataset.pth")
    torch.save(test_dataset, "./../data/test/test_dataset.pth")



#------------------------------------------------------------------------------
# Running the script directly
if __name__ == "__main__":
    
    # Maximum review text sequence length
    max_len = 200
    # Fraction of training data of all data
    train_size = 0.6
    # Fraction of validation data of all data
    validation_size = 0.2
    # Fraction of test data of all data
    test_size = 0.2
    
    # Preprocessing the data
    process_data(max_len,  train_size, validation_size, test_size)






































