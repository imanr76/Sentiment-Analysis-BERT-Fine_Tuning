

# Importing the required libraries
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import data_preparation
from data_preparation import dataset
import time
from datetime import datetime
import transformers
from transformers import AlbertModel
import os
import json
#------------------------------------------------------------------------------
# Function Definitions


# Defining the Model
class sentiment_analyzer(nn.Module):
    
    # Initializing the model
    def __init__(self, albert_model, dropout = 0):
        super().__init__()
        self.training_info = []
        albert_model.pooler = None
        self.albert = albert_model
        for name, param in self.albert.named_parameters():
            if "encoder" not in name:
                param.requires_grad = False
        # Linear layer. Expects inputs of shape (batch_size, 768)
        self.linear1 = nn.Linear(768, 768)
        self.linear2 = nn.Linear(768, 768)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear3 = nn.Linear(768, 1)
        # The sigmoid layer
        self.sigmoid = nn.Sigmoid()
    
    # Defining the forward pass
    def forward(self, input_ids, attention_mask):
        
        # Running the input sequence through the embedding layer and lstm layer
        x = self.albert(input_ids = input_ids, attention_mask = attention_mask)
        x = x[0][:, 0]
        # x = self.linear1(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.linear2(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        x = self.linear3(x)
        return x.view(-1,1)
    
    def train_(self, train_dataloader, validation_dataset, epochs, loss_func, optimizer, device = torch.device("cpu"),threshold = 0.5, batch_size = 128):
        """
        Runs the training loop for of the model and trains the model based on the input parameters.

        Parameters
        ----------
        train_dataloader : obj
            PyTorch DataLoader object for training data.
        validation_dataset : obj
            PyTorch Database object for validation evaulation.
        epochs : int
            DESCRIPTION.
        loss_func : obj
            Loss function to use.
        optimizer : obj
            Optimizer to use.

        Returns
        -------
        None
        """
        # Saving and defining the required parameters and variables
        self.loss_func = loss_func
        num_samples = len(train_dataloader.dataset)
        num_batch = len(train_dataloader)
        loss_train_list = []
        accuracy_train_list = []
        loss_validation_list = []
        accuracy_validation_list = []
        
        validation_loader = DataLoader(validation_dataset, batch_size = batch_size, shuffle = True) 

        # Running the training loop
        for epoch in range(epochs):    
            correct_sentiments = 0
            epoch_loss = 0

            for review, attention_mask, sentiment_real in train_dataloader:
                # print(f"batch {batch} of {num_batch}")
                review = review.to(device)
                attention_mask = attention_mask.to(device)
                sentiment_real = sentiment_real.to(device)
                sentiment_real = sentiment_real.to(torch.float)
                sentiment_real = sentiment_real.view(-1, 1)
                sentiment_pred = self.forward(review, attention_mask)
                sentiment_pred = self.sigmoid(sentiment_pred)
                sentiment_pred_args = torch.where(sentiment_pred >= threshold, torch.tensor(1).to(device), torch.tensor(0).to(device))
                correct_sentiments += torch.sum(sentiment_pred_args == sentiment_real).item()
                optimizer.zero_grad()
                loss_train = self.loss_func(sentiment_pred, sentiment_real)
                loss_train.backward()
                epoch_loss += loss_train.item()
                optimizer.step()
            
            with torch.no_grad():
                correct_sentiments_validation = 0
                run_loss_validation = 0
                for validation_review, validation_attention_mask, validation_sentiment_real in validation_loader:
                    
                    validation_review = validation_review.to(device)
                    validation_attention_mask = validation_attention_mask.to(device)
                    validation_sentiment_real = validation_sentiment_real.to(device)
                    validation_sentiment_real = validation_sentiment_real.view(-1, 1)
                    validation_sentiment_real = validation_sentiment_real.to(torch.float)
                
                    validation_preds = self.forward(validation_review, validation_attention_mask)
                    run_loss_validation += self.loss_func(validation_preds, validation_sentiment_real).item()
                    accuracy_validation_args = torch.where(validation_preds >= threshold, torch.tensor(1).to(device), torch.tensor(0).to(device))
                    correct_sentiments_validation += torch.sum(accuracy_validation_args == validation_sentiment_real).item()
                    
            accuracy_validation = correct_sentiments_validation / len(validation_dataset)
            loss_validatoion = run_loss_validation / len(validation_loader)
            loss_train_list.append(epoch_loss/len(train_dataloader))
            accuracy_train_list.append(correct_sentiments/num_samples * 100)
            loss_validation_list.append(loss_validatoion)
            accuracy_validation_list.append(accuracy_validation * 100)
            loss_info = f"epoch : {epoch + 1}, training loss : {epoch_loss/len(train_dataloader):.4f}, training accuracy : {correct_sentiments/num_samples * 100:.1f}"
            print(loss_info)
            self.training_info.append(loss_info)
            accuracy_info = f"epoch : {epoch + 1}, validation loss : {loss_validatoion:.4f}, validation accuracy : {accuracy_validation * 100:.1f}"
            print(accuracy_info)
            self.training_info.append(accuracy_info)
            print()
            self.training_info.append('\n')
            
        # Plotting the training and validation accuracy and loss during the model training
        # plt.figure()
        # plt.plot(range(1, epochs + 1), loss_train_list, label = "training loss")
        # plt.plot(range(1, epochs + 1), loss_validation_list, label = "validation loss")
        # plt.legend()
        # plt.xlabel("epochs")
        # plt.ylabel("loss")
        # plt.title("Train and Validation Loss per Epoch")
        
        # plt.figure()
        # plt.plot(range(1, epochs + 1), accuracy_train_list, label = "training accuracy")
        # plt.plot(range(1, epochs + 1), accuracy_validation_list, label = "validatoin accuracy")
        # plt.legend()
        # plt.xlabel("epochs")
        # plt.ylabel("accuracy")
        # plt.title("Train and Validation Accuracy per Epoch")
        
    def evaluate(self, test_dataset, threshold, device, batch_size = 128):
        """
        Evaluates the classification of the model based on the test set. 

        Parameters
        ----------
        test_dataset : obj
            PyTorch test Dataset.

        Returns
        -------
        report : dict
            Classification evaluation report.

        """
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True) 
        with torch.no_grad():
            correct_sentiments = 0
            run_loss = 0
            test_sentiment_real_list = torch.tensor([]).to(device)
            predictions_arg_list = torch.tensor([]).to(device)
            for test_review, test_attention_mask, test_sentiment_real in test_loader:

                test_sentiment_real = test_sentiment_real.view(-1, 1)
                test_sentiment_real = test_sentiment_real.to(torch.float)
                test_sentiment_real = test_sentiment_real.to(device)
                
                test_review = test_review.to(device)
                test_attention_mask = test_attention_mask.to(device)
                
                predictions = self.forward(test_review, test_attention_mask)
                predictions_arg = torch.where(predictions >= threshold, torch.tensor(1).to(device), torch.tensor(0).to(device))
                correct_sentiments += torch.sum(predictions_arg == test_sentiment_real).item()
                run_loss +=  self.loss_func(predictions, test_sentiment_real).item()
                
                test_sentiment_real_list = torch.cat((test_sentiment_real_list, test_sentiment_real), dim = 0)
                predictions_arg_list = torch.cat((predictions_arg_list, predictions_arg), dim = 0)
                
            accuracy = correct_sentiments / len(test_dataset) * 100
            loss = run_loss / len(test_loader)
            report = classification_report(test_sentiment_real_list.cpu(), predictions_arg_list.cpu(), output_dict=True, zero_division=0)

            test_info = f"test set loss : {loss:.4}, test set accuracy : {accuracy:.1f}"
            print(test_info)
            self.training_info.append(test_info)
            return report 


def set_to_gpu():
    """
    Sets the device to GPU if available otherwise sets it to CPU. Uses MPS if on mac and CUDA 
    otherwise.

    Returns
    -------
    device : obj.
        PyTorch device object for running the mode on GPU or CPU.
    """
    # Setting the constant seed for repeatability of results.
    seed = 10
    torch.manual_seed(seed)
    
    # Setting the device to CUDA if available
    if torch.cuda.is_available():
       device = torch.device("cuda")
       torch.cuda.manual_seed_all(seed)
       torch.cuda.empty_cache()
    # Setting the device to MPS if available   
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.use_deterministic_algorithms(True)
        torch.mps.manual_seed(seed)
        torch.mps.empty_cache()
    # Setting the device to CPU if GPU not available
    else:
        device = torch.device("cpu")
    # Setting deterministicc behaviour for repatability of results. 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return device


def train_model(dropout = 0.0, learning_rate = 0.001, epochs = 100, threshold = 0.5, batch_size = 128):
    """
    Trains an LSTM model using the input parameters. Saves the model. Also evaluates the 
    classification accuracy on the test set and return the classification report
    besides the model object.

    Parameters
    ----------
    embed_dim : int, optional
        Size of the embedding vector for each token. The default is 20.
    lstm_size : int, optional
        Size of the lstm output. The default is 20.
    bidirectional : boolean, optional
        Whether to run a bidirectional LSTM. The default is True.
    num_layers : int, optional
        Number of LSTM layers. The default is 1.
    dropout : float, optional
        LSTM dropout. The default is 0.
    learning_rate : float, optional
        Learning rate for trianing the model. The default is 0.001.
    epochs : int, optional
        Number of epochs to run. The default is 100.

    Returns
    -------
    report : dict
        A dictionary contatining the classification report based on the test datset.
    model : oobj
        PyTorch model.

    """
    # Starting the timer to measure how long model training takes
    start_time = time.time()
    # Setting the device to GPU if available
    device = set_to_gpu()
    
    # Reading the train, test and validation datasets
    training_dataset = torch.load("./../data/training/training_dataset.pth")
    validation_dataset = torch.load("./../data/validation/validation_dataset.pth")
    test_dataset = torch.load("./../data/test/test_dataset.pth")
    
    # Creating a dataloader from he training dtaset for the model training loop
    train_loader = DataLoader(training_dataset, batch_size = batch_size, shuffle = True)    
    
    

    albert = AlbertModel.from_pretrained('albert-base-v1')
    # Instansiating the model
    model = sentiment_analyzer(albert, dropout).to(device)
    
    # Instansiating the optimizer and loss function
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    # Training the model
    model.train_(train_loader, validation_dataset, epochs, loss_func, optimizer, device, threshold, batch_size)
    
    # Measuring the elapsed time and reporting it
    elapsed_time = time.time() - start_time
    print(f"\nTime it took to train the model: {elapsed_time:.1f}s\n")
    
    #Evaluating the model using the test set and saving the classification report
    # model.to("cpu")
    # report = {}
    model.eval()
    # device = "cpu"
    model.to(device)
    report = model.evaluate(test_dataset, threshold, device, batch_size)
    
    # Saving the model
    now = datetime.now()
        
    if "models" not in os.listdir("./.."):
        os.mkdir("./../models")
        
    model_name = "./../models/Albert model-" + now.strftime("%y_%m_%d-%H_%M_%S")
    
    os.mkdir(model_name)
    
    with open(model_name + "/classification_report.json", "w") as file:
        json.dump(report, file)
    
    with open(model_name + "/training_info.txt", "w") as file:
        file.write("\n".join(model.training_info))
    
    torch.save(model, model_name + '/model.pth')
    
    return report, model
    


#------------------------------------------------------------------------------
# Running the script directly
if __name__ == "__main__":
    
    dropout = 0.3
    # Learning rate for trianing the model
    learning_rate = 0.00001
    # Number of epochs to run
    epochs = 30
    # Setting the threshold for positive and negative labels
    threshold = 0.5
    #
    batch_size = 16
    # Training the model
    report, model = train_model(dropout, learning_rate, epochs, threshold, batch_size)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    