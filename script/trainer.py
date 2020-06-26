import csv
import copy
import time
from tqdm import tqdm
import torch
import numpy as np
import os, sys
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


def train_model(model, criterion, dataloaders,testloaders, optimizer, metrics, bpath, num_epochs=3, scheduler=None):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    best_acc =  0
    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print ('device = ', device)
    model.to(device)
    # Initialize the log file for training and testing loss and metrics
    
    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Test_{m}' for m in metrics.keys()]
    #fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + [f'Train_{m}' for m in metrics.keys()] + [f'Test_{m}' for m in metrics.keys()]

    with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames}

        niters = len (dataloaders['Train'])

        model.train()  # Set model to training mode

        # Iterate over data.
        i = 0
        phase = 'Train'
        for sample in tqdm(iter(dataloaders['Train'])):
            inputs = sample['image'].to(device)
            masks = sample['mask'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # track history if only in train
            with torch.set_grad_enabled(phase == 'Train'):
                outputs = model(inputs)
                #
                #print ('Train: ', inputs.shape, outputs['out'].shape, masks.shape)
                #sys.exit()
                #
                loss = criterion(outputs['out'], masks)
                #
                y_pred = outputs['out'].data.argmax(axis=1).cpu().numpy().ravel() # (batch, rows, cols)
                y_true = masks.data.cpu().numpy().ravel()
                for name, metric in metrics.items():
                    if name == 'acc':
                        batchsummary[f'{phase}_{name}'].append(metric(y_true, y_pred))
                    elif name == 'f1_score':
                        pass #batchsummary[f'{phase}_{name}'].append(metric(y_true, y_pred, average='micro'))
                    else:
                        pass #batchsummary[f'{phase}_{name}'].append(metric(y_true, y_pred, average='micro'))
                #
                # backward + optimize only if in training phase
                if phase == 'Train':
                    loss.backward()
                    optimizer.step()
                    
                    if scheduler:
                        i += 1
                        scheduler.step (epoch + i / niters)

        # for sample ...; 1-epoch finished

        model.eval()
        phase = 'Test'
        for sample in tqdm(iter(testloaders['Test'])):
            inputs = sample['image'].to(device)
            masks = sample['mask'].to(device)

            loss = criterion(outputs['out'], masks)
            #
            y_pred = outputs['out'].data.argmax(axis=1).cpu().numpy().ravel() # (batch, rows, cols)
            y_true = masks.data.cpu().numpy().ravel()
            for name, metric in metrics.items():
                if name == 'acc':
                    batchsummary[f'{phase}_{name}'].append(metric(y_true, y_pred))
                elif name == 'f1_score':
                    pass #batchsummary[f'{phase}_{name}'].append(metric(y_true, y_pred, average='micro'))
                else:
                    pass #batchsummary[f'{phase}_{name}'].append(metric(y_true, y_pred, average='micro'))
        
        batchsummary['epoch'] = epoch
        epoch_loss = loss
        batchsummary[f'{phase}_loss'] = epoch_loss.item()
        print('{} Loss: {:.4f}'.format(phase, loss))

        # for phase in ['Train', 'Test']
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        print(batchsummary)

        # deep copy the model
        if loss < best_loss:
            pbest = best_loss
            best_loss = loss
            best_model_wts = copy.deepcopy(model.state_dict())
            print ('@@ best Test loss decreased: ', int(100*(pbest - best_loss)/pbest), ' %p')
            if bpath:
                wfile = os.path.join(bpath, f'trained_weights.bestloss.pth.tar')
                torch.save({'model_state_dict':model.state_dict(), 'loss': best_loss}, wfile)
                print (f'@@ best Loss {best_loss:.3f} file: ', wfile)
                
        if batchsummary['Test_acc'] > best_acc:
            best_acc = batchsummary['Test_acc']
            if bpath:
                wfile =  os.path.join(bpath, 'trained_weights.best_acc.pth.tar')
                torch.save({'model_state_dict':model.state_dict(), 
                            'accuracy': best_acc}, 
                            wfile)
                print (f'@@ best ACC {best_acc:.3f} weight file: ', wfile)
        
        if epoch%2==0:
            weight_file = os.path.join(bpath,'kasumi_line_ep'+str(epoch)+'.pth.tar')
            torch.save({'model_state_dict':model.state_dict()}, weight_file)
            print ('@ weight saved to ', weight_file)
        #

        with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
    # end for epoch
    #
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
