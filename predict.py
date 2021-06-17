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
import json

#Define parser args
def get_args():
    parser = argparse.ArgumentParser(description = 'flower classification')
    parser.add_argument('--gpu', action='store_true', default='cpu', help="Turn on GPU")
    parser.add_argument('--data_dir', action='store', type=str, default='/home/workspace/ImageClassifier/flowers/test/1/image_06743.jpg', help="Path to image")
    parser.add_argument('--arch', action='store', type=str, default='vgg16', help='Choose model architecture')
    parser.add_argument('--save_dir', dest='save_dir', type=str, default='/home/workspace/ImageClassifier/checkpoint.pth', help="Model saved in directory for Checkpoint, default is current")
    parser.add_argument('--cat_to_name', type = str, default='cat_to_name.json', help=' flower names from the flower file')
    parser.add_argument('--learning_rate', action='store', type=float, default=0.001, help="Sets the learning rate")
    parser.add_argument('--hidden_units', action='store', type=int, nargs=2, default=[512, 256], help="Sets number of hidden units")
    parser.add_argument('--output_size', type=int, action='store', default=102, dest='output_size', help="Output size")
    parser.add_argument('--epochs', action='store', type=int, default=2, help="Sets number of epochs")
    parser.add_argument('--topk', type = int, default = 5, help = 'top 5 classes')
    args = parser.parse_args()
    return args



def label_mapping(cat_to_name):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    #print(cat_to_name)
    return cat_to_name

# Write a function that loads a checkpoint and rebuilds the model

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    learning_rate = checkpoint['learning_rate']
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)    
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
    return optimizer, model
optimizer, model = load_checkpoint('checkpoint.pth')

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    pil_image = PIL.Image.open(image)
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                       [0.229, 0.224, 0.225])])
    pil_image = transform(pil_image)
    np_image = np.array(pil_image)
    #image_array = np.transpose(np_image, (2,0,1))
    #return torch.FloatTensor([image_array])
    return np_image


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    #Implement the code to predict the class from an image file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = process_image(image_path)
    #image = torch.from_numpy(image).type(torch.FloatTensor)
    image = torch.from_numpy(image).unsqueeze(0)
    #image = torch(image).type(torch.FloatTensor)
    #image = torch(image).unsqueeze(0)
    model.to("cuda")
    model.eval()
    model.type(torch.FloatTensor)
    image.to(device)

    
    load_checkpoint('checkpoint.pth')
    
    model.idx_to_class = dict(map(reversed, model.class_to_idx.items()))
       
    with torch.no_grad():
        outputs = model.forward(image)
        ps = torch.exp(outputs)
        probs, indices = ps.topk(topk)
        probs = probs.squeeze()
        classes = [model.idx_to_class[idx] for idx in indices[0].tolist()]
    
    return probs, classes

def main():
    in_args = get_args()
    cat_to_name = in_args.cat_to_name
    save_dir = in_args.save_dir
    image = in_args.data_dir
    lr = in_args.learning_rate
    arch = in_args.arch
    hidden_units = in_args.hidden_units
    output_size = in_args.output_size
    epochs = in_args.epochs
    topk = in_args.topk
    
    optimizer, model_check = load_checkpoint(save_dir)
    cat_to_name = label_mapping(cat_to_name)
    process_image(image)
    accuracy, names = predict(image,model_check, topk)
    
    for x in names:
        #cat.append(cat_to_name[x])
        print(cat_to_name[x])
    #print(names)
    print (accuracy)

if __name__ == "__main__":
    main()   




