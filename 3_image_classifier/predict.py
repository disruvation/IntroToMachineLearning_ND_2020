# Import packages
import argparse
import numpy as np
import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms, models
from PIL import Image
import json
from collections import OrderedDict


def arg_parser():
    parser = argparse.ArgumentParser(description='Skript predicting the flower name from an image using a trained model')
    
    parser.add_argument('filename_image', type=str, help="filename of image to be predicted (required)")
    parser.add_argument('filename_checkpoint', type=str, help="filename of model checkpoint (required)")

    parser.add_argument('--top_k', default='1', type=int, help='return top K most likely classes')
    parser.add_argument('--category_names', default='', type=str, help='filename containing a mapping of categories to real names')
    parser.add_argument('--gpu', default=False, action='store_true', help='activate GPU support for training')

    args = parser.parse_args()
    return args


# define function to load checkpoint
def load_checkpoint(filepath, device):
    # load checkpoint data
    checkpoint = torch.load(filepath)
    
    # load pretrained model 
    if(checkpoint['model_name'] =='vgg16'):
        model = models.vgg16(pretrained=True)
    elif(checkpoint['model_name'] =='densenet121'):
        models.densenet121(pretrained=True) 
    else:
        print('error loading model from checkpoint file')

    # move model parameter to "device" (GPU if available)
    model.to(device);
    
    # Freeze feature parameters
    for param in model.parameters():
        param.requires_grad = False

    # reload model classifier
    model.classifier = checkpoint['classifier']

    # load class to idx
    model.class_to_idx = checkpoint['class_to_idx']
    
    # reload model state dict of classifier
    model.load_state_dict(checkpoint['model_state_dict'])


    print('model {} successfully loaded'.format(checkpoint['model_name']))

    return model


# Process a PIL image for use in a PyTorch model
def process_image(image):
    img = Image.open(image)
    
    # use on top defined definition for transformation
    transformations = transforms.Compose([
                      transforms.Resize(256),
                      transforms.CenterCrop(224),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], 
                                           [0.229, 0.224, 0.225])])
    # perform transformation
    tensor_img = transformations(img)
    
    return tensor_img


def predict(filename_image, filename_category, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file

    # move model parameter to "device" (GPU if available)
    model.to(device)
    # used previously defined process_image function
    tensor_img = process_image(filename_image)
    torch_image = torch.from_numpy(np.expand_dims(tensor_img, axis=0)).type(torch.FloatTensor).to(device)
    
    # forward
    with torch.no_grad():
        output = model(torch_image)
        
        # calculate probability
        prob = torch.exp(output)

        # get top k results
        p_top_k, idx_labels_top_k = prob.topk(topk)

        inv_class_to_idx = { model.class_to_idx[k]: k for k in model.class_to_idx }

        p_top_k = p_top_k.cpu().numpy().flatten()
        idx_labels_top_k = idx_labels_top_k.cpu().numpy().flatten()
        # use model class to idx
        idx_model_labels_top_k = [inv_class_to_idx[ii] for ii in idx_labels_top_k]
        
    # get labels
    labels_top_k = []
    # need to convert indices to str for json
    idx_labels_top_k_str = [str(tmp_label) for tmp_label in idx_model_labels_top_k]
    if(len(filename_category)>0):
        with open(filename_category, 'r') as f:
            cat_to_name = json.load(f)
            print("Category names loaded")
        # save top_k labels in variable labels_top_k
        labels_top_k = [cat_to_name[ii] for ii in idx_labels_top_k_str]
    else:
        # if there is no cat_to_name file, use indices
        labels_top_k = ['index ' + ii for ii in idx_labels_top_k_str]

    return p_top_k, labels_top_k


def print_predictions(probs, classes, top_k):
    for ii in range(top_k):
        print('Selected image shows with a probability of {:02.3f}: {}'.format(probs[ii],classes[ii]))


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
    
    # load model
    model = load_checkpoint(args.filename_checkpoint, device)

    # select criterion
    criterion = nn.NLLLoss()

    # predicts the flower name
    probs, classes = predict(args.filename_image, args.category_names, model, args.top_k, device)

    print_predictions(probs, classes, args.top_k)



if __name__ == "__main__":
    main()