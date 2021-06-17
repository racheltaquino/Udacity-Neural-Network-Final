from torchvision import transforms
from torchvision import datasets
from torchvision import models
from torch import nn, optim
import torch
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse



#Define parser args
parser=argparse.ArgumentParser()
parser.add_argument('--gpu', action='store_true', default='cpu', help="Turn on GPU")
parser.add_argument('--data_dir', action='store', type=str, default='flowers', help="Path to image")
parser.add_argument('--arch', action='store', type=str, default='vgg16', help='Choose model architecture')
parser.add_argument('--save_dir', dest='save_dir', type=str, default='.checkpoint.pth', help="Model saved in directory for Checkpoint, default is current")
parser.add_argument('--learning_rate', action='store', type=float, default=0.001, help="Sets the learning rate")
parser.add_argument('--hidden_units', action='store', type=int, nargs=2, default=[512, 256], help="Sets number of hidden units")
parser.add_argument('--output_size', type=int, action='store', default=102, dest='output_size', help="Output size")
parser.add_argument('--epochs', action='store', type=int, default=2, help="Sets number of epochs")

args = parser.parse_args()

#Define directories
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#Define your transforms for the training, validation, and testing sets
data_transforms = {'train':
transforms.Compose([transforms.Resize(224),
transforms.CenterCrop(224),
transforms.RandomVerticalFlip(),
transforms.RandomRotation(90),
transforms.ToTensor(),
transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
'valid':
transforms.Compose([transforms.Resize(224),
transforms.CenterCrop(224),
transforms.RandomVerticalFlip(),
transforms.RandomRotation(90),
transforms.ToTensor(),
transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
'test':
transforms.Compose([transforms.Resize(224),
transforms.CenterCrop(224),
transforms.RandomVerticalFlip(),
transforms.RandomRotation(90),
transforms.ToTensor(),
transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])} 


# Load the datasets with ImageFolder
image_datasets = {'train':
datasets.ImageFolder(train_dir, transform = data_transforms['train']),
'valid':
datasets.ImageFolder(train_dir, transform = data_transforms['valid']),
'test':
datasets.ImageFolder(test_dir, transform = data_transforms['test'])}


# Using the image datasets and the trainforms, define the dataloaders
trainDataLoader = torch.utils.data.DataLoader(image_datasets['train'], batch_size = 32, shuffle = True)
validDataLoader = torch.utils.data.DataLoader(image_datasets['valid'], batch_size = 32, shuffle = True)
testDataLoader = torch.utils.data.DataLoader(image_datasets['test'], batch_size = 32, shuffle = True)

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
#Use GDU if it's available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Build and train your network
model = models.vgg16(pretrained=True)

#freeze the parameters
for param in model.parameters():
    param.requires_grad = False
    
#define the classifier
model.classifier = nn.Sequential(nn.Linear(25088, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(4096, 102),
                                 nn.LogSoftmax(dim=1))

#to display validation loss and accuracy
#train classifier parameters
criterion = nn.NLLLoss()

learning_rate = 0.0001
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
model.to(device)

#define training paramaters
epochs = 2
train_loss = 0
steps = 0
print_every = 40

#train the model, iterate over data
for epoch in range(epochs):
    model.train()
    for inputs, labels in iter(trainDataLoader):
        steps +=1
        
        #mode to default device
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        
        # Get model outputs and calculate loss
        # Backward + optimize
        logps = model.forward(inputs)
        loss = criterion(logps, labels)

        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        #validate and display accuracy and loss, drop out regularly to test accuracy
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validDataLoader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {train_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(validDataLoader):.3f}.. "
                  f"Test accuracy: {accuracy/len(validDataLoader):.3f}")
            train_loss = 0
            model.train()
            
#Do validation on the test set
test_loss = 0
accuracy = 0
model.eval()
with torch.no_grad():
    for inputs, labels in testDataLoader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)

        test_loss += batch_loss.item()

        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

print(f"Test loss: {test_loss/len(testDataLoader):.3f}.. "
      f"Test accuracy: {accuracy/len(testDataLoader):.3f}")

#Save the checkpoint 

checkpoint = {'classifier': model.classifier,
              'learning_rate': learning_rate,
              'state_dict': model.state_dict(),
              'class_to_idx': image_datasets['train'].class_to_idx,
              'optimizer_dict': optimizer.state_dict()}
torch.save(checkpoint, 'checkpoint.pth')



