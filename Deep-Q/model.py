import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        # create linear layers (input -> hidden -> output)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
    
    def save(self, file_name='model.pth'):
        
        model_folder_path = os.path.join('.','model')
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr # learning rate
        self.gamma = gamma # discount rate
        
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) # TODO: try different optimizers (e.g. SGD, RMSprop, etc.)
        self.criterion = nn.MSELoss() # loss function TODO: try different loss functions (e.g. Huber, L1, etc.)
        
    def train_step(self, state, action, reward, next_state, done):
        # convert numpy arrays to torch tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # in form of (n, x) where n is the number of samples and x is the number of features
        
        if len(state.shape) == 1: # if there is only one sample (n = 1) then add batch dimension
            # unsqueeze adds a dimension of 1 at the specified position
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, ) # make done a tuple
            
        # 1. predicted Q values with current state
        pred = self.model(state)
        
        # 2. Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # Q_new = reward + gama * max(next_predicted Q value)
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx])) 
            target[idx][torch.argmax(action[idx]).item()] = Q_new
            
            
        # 3. loss = (Q_new - Q_old)^2
        self.optimizer.zero_grad() # reset gradients to 0
        loss = self.criterion(target, pred) # calculate loss
        loss.backward() # backpropagation
        self.optimizer.step() # update weights
            
        
        

        
        