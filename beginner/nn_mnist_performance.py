import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torch.nn.functional as F 
import torchvision.transforms as transforms

print("hot")

# setting device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# importing the dataset and preparing the dataloader for it
train_dataset=datasets.MNIST(root='dataset/',train=True,transform=transforms.ToTensor(),download=True)
test_dataset=datasets.MNIST(root='dataset/',train=False,transform=transforms.ToTensor(),download=True)

input_size=784
num_classes=10
batch_size=32
learning_rate=0.001

train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

# describing the network architecture and the flow of data from input to output
class NN(nn.Module):
  def __init__(self,input_size,num_classes):
    super(NN,self).__init__()
    self.fc1=nn.Linear(input_size,100)
    self.fc2=nn.Linear(100,50)
    self.fc3=nn.Linear(50,num_classes)
    

  def forward(self,x):
    x=F.relu(self.fc1(x))
    x=F.relu(self.fc2(x))
    x=self.fc3(x)
    return x

# initializing the model and the loss and optimizer
model=NN(input_size,num_classes=num_classes).to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=learning_rate)

def load_checkpoint(checkpoint):
    print("__Loading Checkpoint__")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Loading the saved model 
load_checkpoint(torch.load(f"model/nn_mnist.pth.tar", map_location=device))

# checking the accuracy of the network
def check_accuracy(loader,model):
  if loader.dataset.train:
    print("checking accuracy on training data")
  else:
    print("checking accuracy on test data")
    
  num_correct=0
  num_samples=0
    
  model.eval()
  with torch.no_grad():
    for x,y in loader:
      x=x.to(device=device)
      y=y.to(device=device)
      x=x.reshape(x.shape[0],-1)

      scores=model(x)
      _, predictions=scores.max(1)
      
      num_correct+=(predictions==y).sum()
      num_samples+=predictions.size(0)
    print(f'{num_correct}/{num_samples} Correct,with accuracy {float(num_correct)/float(num_samples)*100:.5f}')
  model.train()

# checking overall accuracy of the model
check_accuracy(train_loader,model)
check_accuracy(test_loader,model)

# checking accuracy for each digits in training dataset
total_train=np.zeros([10,])
train_correct=np.zeros([10,])

model.eval()
with torch.no_grad():
    for x,y in train_dataset:
        x=x.to(device=device)
        y=torch.tensor(y)
        y=y.to(device=device)
        x=x.reshape(x.shape[0],-1)

        scores=model(x)
        _, prediction=scores.max(1)

        if prediction==y:
            train_correct[y]+=1
        for i in range(10):
            if int(y)==int(i):
                total_train[i]+=1
model.train()
print("total training labels:",len(train_dataset))
print("total train labels:",total_train)
print("total train correct:",train_correct)
print("test accuracy:",train_correct/total_train)


# checking accuracy for each digits in testing dataset
total_test=np.zeros([10,])
test_correct=np.zeros([10,])

model.eval()
with torch.no_grad():
    for x,y in test_dataset:
        x=x.to(device=device)
        y=torch.tensor(y)
        y=y.to(device=device)
        x=x.reshape(x.shape[0],-1)

        scores=model(x)
        _, prediction=scores.max(1)

        if prediction==y:
            test_correct[y]+=1
        for i in range(10):
            if int(y)==int(i):
                total_test[i]+=1
model.train()
print("total testing labels:",len(test_dataset))
print("total test labels:",total_test)
print("total test correct:",test_correct)
print("test accuracy:",test_correct/total_test)
