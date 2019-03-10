import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='image classifier parameter/hyperparameter')
parser.add_argument('data_directory', action='store', default='flowers', \
                    help='dir where data are stored')

parser.add_argument('input_image', action='store', \
                    help='dir of test image')

parser.add_argument('checkpoint', action='store', \
                    help='dir where checkpoint is stored')

parser.add_argument('--category_names', action='store', default='cat_to_name.json', \
                    required=False, help='a mapping of classes/categories to names')

parser.add_argument('--top_k', action='store', default=5, type=int, \
                    required=False, help='top k probabilities')

parser.add_argument('--gpu', default=False, action='store_true', \
                    required=False, help='whether gpu is used in training')

args = parser.parse_args()
print(args)

# data loading
data_dir = args.data_directory
test_dir = data_dir + '/test'

# Define transforms for the training, validation, and testing sets
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# TUsing the image datasets and the trainforms, define the dataloaders
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

# a function that loads a checkpoint and rebuilds the model
def load_model(checkpoint_path):
    chpt = torch.load(checkpoint_path)#, map_location=lambda storage, loc: storage)
    # Try to load your state_dict or model with these arguments to force all tensors to be on CPU:
    
    if chpt['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    elif chpt['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    elif chpt['arch'] == 'vgg19':
        model = models.vgg19(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    else:
        print("Sorry base architecture note recognized")
    
    model.class_to_idx = chpt['class_to_idx']
    
    # Create the classifier
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, chpt['hidden_units'])),
                          ('relu1', nn.ReLU()),
                          ('dropout', nn.Dropout(0.1)),
                          ('fc2', nn.Linear(chpt['hidden_units'], 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    # Put the classifier on the pretrained network
    model.classifier = classifier
    
    model.load_state_dict(chpt['state_dict'])
    
    return model

model = load_model(args.checkpoint)

# double-check if model is loaded with test_accuracy WITH CUDA
device = torch.device('cuda')
test_accuracy = 0
model.eval()
model.to(device)
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
print(f"Test accuracy: {test_accuracy/len(testloader):.3f}")

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    
    image_target = Image.open(image)
    
    if image_target.size[0] > image_target.size[1]:
        image_target.thumbnail((1000000, 256))
    else:
        image_target.thumbnail((256, 1000000))
    image_target = image_target.crop((image_target.width/2-112, 
                                      image_target.height/2-112, 
                                      image_target.width/2+112, 
                                      image_target.height/2+112))
    image_target = np.array(image_target) / 255
    nor_mean = np.array([0.485, 0.456, 0.406])
    nor_std = np.array([0.229, 0.224, 0.225])
    image_target = (image_target - nor_mean) / nor_std
    image_target = image_target.transpose((2, 0, 1))
    return image_target


def predict(image_path, model, top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Implement the code to predict the class from an image file
    if args.gpu == True:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    image_numpy = process_image(image_path)
    
    image_tensor = torch.from_numpy(image_numpy).float().to(device)
    print(image_tensor.is_cuda)

    model_input = image_tensor.unsqueeze(0)
    model.eval()
    model.to(device)
    with torch.no_grad():
        ps = torch.exp(model.forward(model_input))
        
        top_ps, top_indices = ps.topk(top_k, dim=1)
        # Change from tensor to numpy array
        # Return the array as a (possibly nested) list.
        # Return a copy of the array data as a (nested) Python list. 
        # Data items are converted to the nearest compatible Python type.
        top_ps = top_ps.detach().cpu().numpy().tolist()[0] 
        top_indices = top_indices.detach().cpu().numpy().tolist()[0]

        idx_to_class = {}
        # use a dict mapping idx to class
        for k, v in model.class_to_idx.items():
            idx_to_class[v] = k
        top_classes = [idx_to_class[idx] for idx in top_indices]
        top_flowers = [cat_to_name[clas] for clas in top_classes]
        return top_ps, top_classes, top_flowers

image_path = args.input_image
print(predict(image_path, model, args.top_k))