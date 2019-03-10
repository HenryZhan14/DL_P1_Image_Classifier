import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from workspace_utils import active_session
import json
from collections import OrderedDict

# argument parsing
parser = argparse.ArgumentParser(description='image classifier parameter/hyperparameter')
parser.add_argument('data_directory', action='store', default='flowers', \
                    help='dir where data are stored')
parser.add_argument('--save_dir', action='store', \
                    required=False, help='dir where checkpoint is stored')
parser.add_argument('--arch', action='store', default='vgg19', required=False, \
                    choices=['vgg13', 'vgg16', 'vgg19'], help='model architecture')
parser.add_argument('--learning_rate', action='store', default=0.01, type=float, \
                    required=False, help='learning rate')
parser.add_argument('--hidden_units', action='store', default=4096, type=int, \
                    required=False, help='# of hidden units')
parser.add_argument('--epochs', action='store', default=15, type=int, \
                    required=False, help='learning epochs')
parser.add_argument('--gpu', default=False, action='store_true', \
                    required=False, help='whether gpu is used in training')

args = parser.parse_args()
print(args)
data_dir = args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

# a mapping of classes/categories to names
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# model selection
if args.arch == 'vgg13':
    model = models.vgg13(pretrained=True)
elif args.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
elif args.arch == 'vgg19':
    model = models.vgg19(pretrained=True)
else:
    print('Please use one of these three models: vgg13, vgg16, and vgg19')

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

# Define a new, untrained feed-forward network as a classifier, 
# using ReLU activations and dropout
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, args.hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dropout', nn.Dropout(0.1)),
                          ('fc2', nn.Linear(args.hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

model.classifier = classifier

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

# Use GPU if it's available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.gpu == True:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
model.to(device)
# Train the classifier layers using backpropagation using the pre-trained 
# network to get the features
# Track the loss and accuracy on the validation set to determine the best hyperparameters
# model training
epochs = args.epochs
steps = 0
running_loss = 0
print_every = 5
with active_session():
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                # train loss = average loss of five mini-batches
                running_loss = 0
                model.train()

# Do validation on the test set (check performance of the model)
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

# save model on cpu
model.class_to_idx = train_data.class_to_idx
model.cpu()

checkpoint = {'arch': args.arch,
              'state_dict': model.state_dict(), 
              'class_to_idx': model.class_to_idx,
              'hidden_units': args.hidden_units}

torch.save(checkpoint, 'checkpoint.pth')