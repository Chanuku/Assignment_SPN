import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from model import createDeepLabv3
from trainer import train_model
import datahandler
import argparse
import os
import torch

# Command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_directory", 
    default='../Downloads/coloring_test/segmentation/train_data/train_6/train', 
    #default='../Downloads/DeepLabv3-TS/train_data/screenshot_burn/train',
    help='Specify the dataset directory path')
parser.add_argument(
    "--exp_directory", 
    default='../Downloads/coloring_test/segmentation/exp_out', 
    help='Specify the experiment directory where metrics and model weights shall be stored.')
parser.add_argument("--epochs", default=21, type=int)
parser.add_argument("--batchsize", default=2, type=int)

args = parser.parse_args()


bpath = args.exp_directory
data_dir = args.data_directory
val_dir = args.data_directory[:-5]+'val'
epochs = args.epochs
batchsize = args.batchsize
print (args)

# Create the deeplabv3 resnet101 model which is pretrained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
model = createDeepLabv3(outputchannels=5, backboneFreez=False) # give nclasses
model.train()

# make a larger image, copy the image and fill the other parts by black color.
if batchsize == 1: 
    def set_bn_to_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()
            print (classname)

    model.apply (set_bn_to_eval)
#

# Create the experiment directraintory if not present
if not os.path.isdir(bpath):
    os.mkdir(bpath)


# Specify the loss function
criterion = torch.nn.CrossEntropyLoss()
# Specify the optimizer with a lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2)
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-4)
#scheduler = None
# Specify the evalutation metrics
metrics = {'acc': accuracy_score}


# Create the dataloader
dataloaders = datahandler.get_dataloader_single_folder(data_dir, 
                                                        batch_size=batchsize,
                                                        imageFolder='image',
                                                        maskFolder='mask')

testloaders = datahandler.get_dataloader_single_folder(val_dir, 
                                                        batch_size=batchsize,
                                                        imageFolder='image',
                                                        maskFolder='mask')

trained_model = train_model(model, criterion, dataloaders,testloaders,
                            optimizer,
                            bpath=bpath, metrics=metrics, num_epochs=epochs
                            )
'''
trained_model = train_model(model, criterion, dataloaders,
                            optimizer,
                            bpath=bpath, metrics=metrics, num_epochs=epochs
                            )
'''

# Save the trained model
weight_file = os.path.join(bpath,'kasumi_line_ep20000.pth.tar')
torch.save({'model_state_dict':trained_model.state_dict()}, weight_file)
print ('@ weight saved to ', weight_file)


# model = createDeepLabv3(outputchannels=4)
#checkpoint = torch.load(os.path.join(bpath,'trained_weights_1122.pth.tar'))
#model.load_state_dict(checkpoint['model_state_dict'])
#torch.save({'model_state_dict':trained_model.state_dict()}, weight_file)
