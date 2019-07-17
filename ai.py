# AI for Self Driving Car

# Importing the libraries
import numpy as np
import random #for taking random samples for the Experience Replay
import os #useful to load the model
import torch
import torch.nn as nn #nn module contains all the frameworks for neural networks
import torch.nn.functional as F #contains loss function to be used later on
import torch.optim as optim #optimizer to perform stochastic gradient descent
import torch.autograd as autograd
from torch.autograd import Variable 
##In the forward phase, the autograd tape will remember all the operations 
#it executed, and in the backward phase, it will replay the operations.
##In autograd, if any input Tensor of an operation has requires_grad=True, 
#the computation will be tracked. 
#After computing the backward pass, a gradient w.r.t. this tensor is accumulated into .grad attribute.


#Creating the architecture of the Neural Network
class Network(nn.Module):#inheritance of "Module" parent class properties and methods
    def __init__ (self, input_size, nb_action):
        super(Network, self).__init__() #to inherit from nn.Module
        self.input_size = input_size
        self.nb_action = nb_action 
        self.fc1 = nn.Linear(input_size, 30) #full connection 1 - i/p to hidden
        self.fc2 = nn.Linear(30, nb_action) #full connection 2 - hidden to o/p
        #Hence the above two lines give us the architecture of the Neural Network
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# Implementing Experience Replay

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity #capacity of replay, max 100
        self.memory = [] #list to store the replays(changes in state)
        
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size)) #basically takes transpose of the given list
        return map(lambda x: Variable(torch.cat(x, 0)), samples) #see video to understand

#Implementing Deep Q Learning

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = [] #list to take the mean of the last 100 rewards
        #mean(reward_window) should increase with time
        self.model = Network(input_size, nb_action)#creating/initializing the neural network
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        #creating Tensor for feeding into NN
        #before feeding into NN, fake dimension must be added as the first dimension
        self.last_action = 0 #index number will reference 0, 20, -20 degrees. See map.py for understanding
        self.last_reward = 0        
        
    def select_action(self, state): #as the o/p(action) of the NN depends directly on the state it is present in
        #--argmax just takes the maximum value
        #--hence, softmax is recommended for accounting for "randomness"
        #  and paving the way for exploration
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T = 7
        #--volatile = True
        #  wont be associating the gradients associated with the input states 
        #  to the graph of all the computations of the NN module
        #--7 is the Temperature parameter - allows us to modulate, How sure is the NN of the action to take?
        #  postivie no
        #  closer to 0 --> less sure, higher --> more sure and hence will not move like an insect
        action = probs.multinomial()
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        #We get the maximum of the Q - Values of the next state(rep by index 0) according to all the actions(rep by index 1)
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True) #helps to save memory
        self.optimizer.step() #line of code that updates the weights
        
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)      #first putting number into list and then making it a tensor
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))  #torch.LongTensor--to ensure that self.last_action( 0, 1, 2) is also a tensor
        action = self.select_action(new_state) #play the new action after reaching the new state
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0] #i.e, if there are more than 1000 means of the last 100 rewards
        return action
    
    def score(self):
        return (sum(self.reward_window)/(len(self.reward_window)+1.))
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    },'last-brain.pth')
    
    def load(self):
        if os.path.isfile('last-brain.pth'):
            print(" => Loading Checkpoint. . . ")
            checkpoint = torch.load('last-brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Done!")
        else:
            print("No checkpoint found. . .")
        
        