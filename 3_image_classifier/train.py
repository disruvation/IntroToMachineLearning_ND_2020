# Import packages
import argparse
import numpy as np
import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms, models
import time
import copy
from collections import OrderedDict

def arg_parser():
    parser = argparse.ArgumentParser(description='Training Skript for a Deep Neural Network for Image Classification')
    
    parser.add_argument('data_dir', type=str, help="data directory (required)")
	
    parser.add_argument('--save_dir', default='', type=str, help='directory to save the model checkpoint')
    parser.add_argument('--arch', choices=['vgg16', 'densenet121'], default='vgg16', type=str, help='model architecture, options: vgg16, densenet121')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--hidden_units', default=4096, type=int, help='number of hidden layers')
    parser.add_argument('--epochs', default=5, type=int, help='number of epochs for training')
    parser.add_argument('--gpu', default=False, action='store_true', help='activate GPU support for training')

    args = parser.parse_args()
    return args


# load data
def load_training_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    # TODO: Define your transforms for the training, validation, and testing sets
    # dictionaryy forms based on: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#load-data
    data_transforms = {
     'train': transforms.Compose([
         transforms.RandomResizedCrop(224),
         transforms.RandomRotation(30),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
        'validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # creating a dictionary containing all data folder
    dir_dict = {'train':train_dir,
                'validation': valid_dir}

    # TODO: Load the datasets with ImageFolder
    image_datasets = {x: datasets.ImageFolder(dir_dict[x], data_transforms[x])
                      for x in ['train', 'validation']}
    
    class_to_idx = image_datasets['train'].class_to_idx

    print('image dataset loaded.')
    return image_datasets, class_to_idx


# build model
def build_model(arch, nof_hidden_layer, learning_rate):
    # select pretrained model
    if(arch=='vgg16'):
        model = models.vgg16(pretrained=True) 
        nof_input_layer = 25088
    elif(arch=='densenet121'):
        model = models.densenet121(pretrained=True) 
        nof_input_layer = 1024
        if(nof_hidden_layer>1000):
            print('max. 1000 hidden layer allowed for densenet121. number of hidden layer reduced to: 512')
            nof_hidden_layer=512
    else:
        print('no valid model arch selected... will use vgg16.')
        model = models.vgg16(pretrained=True) 
        nof_input_layer = 25088
        
    # Freeze feature parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Build classifier
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(nof_input_layer, nof_hidden_layer)),
                              ('relu1', nn.ReLU()),
                              ('drop1', nn.Dropout(0.4)),
                              ('fc2', nn.Linear(nof_hidden_layer, 102)),
                              ('out', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier

    # select criterion
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, the feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)    
    
    print('model {} built with {} hidden layer and a learning rate of {}.'.format(arch,nof_hidden_layer,learning_rate))
    return model, criterion, optimizer

    
# define function for training the model
# based on https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#load-data
def train_model(model, image_datasets, criterion, optimizer, device, nof_epochs):
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True) 
                   for x in ['train', 'validation']}
    
    # get dataset size
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
    # save start time
    time_epoch = []
    time_epoch.append(time.time())
    
    # move model parameter to "device" (GPU if available)
    model.to(device);
    
    # initialize variables for the best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print('Start training of the model...')
    print()
    for epoch in range(nof_epochs):
        print('Epoch {}/{}'.format(epoch + 1, nof_epochs))
        print('-' * 10)

        # each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                # set model mode to training (dropouts activated)
                model.train()
            else:
                # change model mode to evaluation (no dropouts)
                model.eval()

            # initialize and reset variables
            running_loss = 0.0
            running_corrects = 0

            # iterate over data from dataloader
            for inputs, labels in dataloaders[phase]:
                # move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # in training mode: perform backward + optimize
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # perform some statistics (calculate loss + correct)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # calculate epoch loss
            epoch_loss = running_loss / dataset_sizes[phase]
            # calculate epoch accuracy
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # print losses and accuracy per phase and epoch
            print('{} loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    
        # print timing results for epoch
        time_epoch.append(time.time() - time_epoch[epoch])
        print('epoch duration: {:.0f}m {:.0f}s'.format(time_epoch[epoch+1] // 60, time_epoch[epoch+1] % 60))
        print()

    # print timing results
    time_elapsed = time.time() - time_epoch[0]
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print best accuracy result
    print('Best valid Acc: {:.4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# define function to save checkpoint
def save_checkpoint(model, arch, filepath=''):
    # check model architecture
    if(arch=='vgg16'):
        nof_input_layer = 25088
    elif(arch=='densenet121'):
        nof_input_layer = 1024
    else:
        nof_input_layer = 25088
    
    # create full filname
    filename = filepath + arch + '_checkpoint.pth'
        
    # parameters
    checkpoint_state = {'model_name':arch,
                        'input_size': nof_input_layer,
                        'output_size': 102,
                        'model_state_dict': model.state_dict(),
                        'class_to_idx' : model.class_to_idx,
                        'classifier': model.classifier}
    # save
    torch.save(checkpoint_state, filename)
    
    print('model checkpoint saved to file: {}'.format(filename))

def main():
    # get input args
    args = arg_parser()
    # Check if GPU is available
    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print('calculation will be performe on: {}'.format(device))
        else:
            device = torch.device("cpu")
            print('no GPU available. calculation will be performe on: {}'.format(device))
    else:
        device = torch.device("cpu")
        print('calculation will be performe on: {}'.format(device))
    
    # load image files
    image_datasets, class_to_idx = load_training_data(args.data_dir)
    # build model
    model, criterion, optimizer = build_model(args.arch, args.hidden_units, args.learning_rate)
    model.class_to_idx = class_to_idx
    # train the model
    model = train_model(model, image_datasets, criterion, optimizer, device, args.epochs)
    # save model checkpoint
    save_checkpoint(model, args.arch, args.save_dir)

if __name__ == "__main__":
    main()