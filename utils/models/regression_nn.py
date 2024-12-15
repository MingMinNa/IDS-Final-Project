import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class RegressionNN(nn.Module):

    def __init__(self, input_size, output_size, hidden_size = [64, 128, 256, 1024, 1024, 512, 128, 128, 64]):
        super(RegressionNN, self).__init__()

        self.hidden_layer_cnt = len(hidden_size)
        self.fc = nn.ModuleList()
        self.bn = nn.ModuleList()

        self.fc.append(nn.Linear(input_size, hidden_size[0]))
        self.bn.append(nn.BatchNorm1d(hidden_size[0]))
        
        for i in range(1, len(hidden_size)):
            self.fc.append(nn.Linear(hidden_size[i - 1], hidden_size[i]))  
            self.bn.append(nn.BatchNorm1d(hidden_size[i]))                 
        

        self.fc.append(nn.Linear(hidden_size[-1], output_size))

        self.relu = nn.ReLU()                            # ReLU function
        self.dropout = nn.Dropout(0.3)                   # Dropout to avoid overfitting
        
    def forward(self, x):

        for i in range(self.hidden_layer_cnt):
            x = self.fc[i](x)   
            x = self.bn[i](x)                                  
            x = self.relu(x)                                 
            x = self.dropout(x)                              
        x = self.fc[-1](x)

        return x

def build_model(X, y, save_path, random_seed = 42, epochs = 300, batch_size = 32):  
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = 0.8, shuffle = True, random_state = random_seed)

    torch.manual_seed(random_seed)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(DEVICE)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(DEVICE)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1).to(DEVICE)
    
    input_size = X_train.shape[1]
    output_size = 1
    regression_model = RegressionNN(input_size = input_size, output_size = output_size).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(regression_model.parameters(), lr = 0.001)
    scheduler = StepLR(optimizer = optimizer, step_size = 10, gamma = 0.7)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf') # min_val_loss
    best_weight = None

    for epoch in tqdm(range(epochs)):
        regression_model.train()

        train_loss = 0
        for i in range(0, len(X_train_tensor), batch_size):
            X_batch = X_train_tensor[i:i+batch_size]
            y_batch = y_train_tensor[i:i+batch_size]

            optimizer.zero_grad()
            outputs = regression_model(X_batch)
            loss = criterion(outputs, y_batch)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        regression_model.eval()  
        with torch.inference_mode():  
            y_pred = regression_model(X_val_tensor)
            val_loss = mean_squared_error(y_val_tensor.cpu().numpy(), y_pred.cpu().numpy())

        scheduler.step()
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if best_val_loss > val_loss:
            best_weight = regression_model.state_dict()
            best_val_loss = val_loss

        if (epoch + 1) % 20 == 0:
            print(f"Epoch[{epoch}/{epochs}] -- train_loss: {train_loss}, val_loss: {val_loss}")

    
    print(f"best val loss: {best_val_loss}")
    torch.save(best_weight, save_path)
    return regression_model, train_losses, val_losses

def predict(regression_model, input_values):

    input_values_tensor = torch.tensor(input_values.values, dtype=torch.float32).to(DEVICE)
    regression_model.to(DEVICE)

    regression_model.eval()
    with torch.inference_mode():
        y_predicts = regression_model(input_values_tensor)
    
    return y_predicts.cpu()

def load_model(input_size, output_size, model_path, weight_only = True):

    if weight_only:
        regression_model = RegressionNN(input_size, output_size)
        weight = torch.load(model_path, weights_only = True)
        regression_model.load_state_dict(weight)
    else:
        regression_model = torch.load(model_path, weights_only = False)
    
    return regression_model
 

if __name__ == '__main__':
    pass