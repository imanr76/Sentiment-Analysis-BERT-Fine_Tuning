import torch
import os 
from training import sentiment_analyzer 
from transformers import AlbertTokenizer

def predict(model, reviews, threshold, max_len, tokenizer = AlbertTokenizer.from_pretrained("albert-base-v1")):
    """
    Accepts some reviews nd returns their corresponding sentiments. 

    Parameters
    ----------
    model : obj
        PyTorch model.
    tokenizer : obj
        Word tokenizer.
    vocabulary : obj
        Vocabulary object.
    lemmatizer : obj
        Lemmantizer object.
    reviews : TYPE
         A list of reviews whose sentiment we want to predict.
    threshold : float
        Threshold for determining positive and negative values based on the model output.
    max_len : int
        Maximum allowed length of the input sequence.

    Returns
    -------
    predictions : TYPE
        DESCRIPTION.

    """
    with torch.no_grad():
        model.eval()
        model.to("cpu")
        predictions = []
        for review in reviews:
            review_processed = tokenizer(review, return_tensors="pt", padding = False, truncation = True, max_length  = max_len)
            prediction = torch.where(model.sigmoid(model(review_processed["input_ids"], review_processed["attention_mask"])) >= threshold, torch.tensor(1), torch.tensor(0))
            print(model.sigmoid(model(review_processed["input_ids"], review_processed["attention_mask"])))
            if prediction.item() == 1:
                predictions.append("Positive")
            else:
                predictions.append("Negative")
            print(f"Review : '{review}'\nPredicted Sentiment : {predictions[-1]}\n")
    return predictions

#------------------------------------------------------------------------------
# Running the script directly
if __name__ == "__main__":
    
    
    # Setting the threshold for positive and negative labels
    threshold = 0.5
    max_len = 500
    model = torch.load(os.path.join("./models", os.listdir("./models")[-1]))
    
    reviews =  ["i love it", "it was too loose" ,"This is a nice computer, I love using it and i will recommend you to buy it as soon as possible"]
    
    predictions = predict(model, reviews, threshold, max_len)