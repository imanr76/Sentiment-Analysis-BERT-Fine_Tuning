import transformers
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch
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

def process_reviews(review, tokenizer, max_len, pad = True):
    """
    Performs the following tasks on each review text:
        - cleaning the text
        - tokenizing the text
        - lemmantizing the text
        - converting the tokens into indices 
        - padding and truncating the review based on max_len passed to the function

    Parameters
    ----------
    review : str
        The product review text.
    tokenizer : obj
        Tokenizer object for tokenization of the text.
    lemmatizer : obj
        Lemmantizer onject for Llmmantization of the text.
    vocabulary : obj
        Vocabulary object correspoding tokens and indices.
    max_len : int
        Maximum allowed length of a product review.
    pad : boolean
        Either to pad the input or not.

    Returns
    -------
    review_processed : list
        A list of indices.
    """
    review_tokenized = tokenizer(review)
    if pad and len(review_processed) < max_len:
        review_processed.extend([0] * (max_len - len(review_processed)))
    elif len(review_processed) > max_len:
        review_processed = review_processed[:max_len]
    return review_processed

def convert_to_tensor(dataframe):
    """
    Converts the dataframe values into a list of tensors and appending the sentiment for each
    review to the review tensor.

    Parameters
    ----------
    dataframe : pandas DataFrame
        Dataframe whose data to be converted.

    Returns
    -------
    combined_tensor : list[torch.tensor]
        A list of torch tensors containing the indices of each review and the sentiment 
        as the last element.
    """
    
    # Converting the dataset values to lists
    review_processed_values = dataframe['review_processed'].tolist()
    sentiment_values = dataframe['sentiment'].tolist()
    #Converting dataset values to tensors
    review_processed_tensor = torch.tensor(review_processed_values)
    sentiment_tensor = torch.tensor(sentiment_values)
    # Appending the sentiment to the review indices tensor as the last element
    sentiment_tensor = sentiment_tensor.unsqueeze(1)
    combined_tensor = torch.cat((review_processed_tensor, sentiment_tensor), dim=1)
    return combined_tensor
    

class dataset(Dataset):
    """
    Pytorch Dataset class for converting the pandas dataframes into datasets
    """
    def __init__(self,data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

print(tokenizer(["Hello world", "Hello i i i i i i "], return_tensors="pt", padding=True, truncation=True))