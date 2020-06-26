import torch
import torchvision
import cv2
import numpy as np
import os


exp_dir = '../Documents/Term_project/exp_out/'
dir =  '../Documents/Term_project/test/'
out_dir =  '../Documents/Term_project/result/'
filist = os.listdir(dir)

#######################################################
def decode_segmap(image, nc=6):
  
  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (0, 0, 255), (0, 255, 0), (255, 0, 0), (128, 0, 128), (255, 255, 0),
               # 6=bus, 7=car
               (0, 128, 128)])

  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)

  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]

    
  rgbp = np.stack([r, g, b], axis=2)
  return rgbp
################################################
# Load the trained model 
import model as myModel

model = myModel.createDeepLabv3(outputchannels=5) # give nclasses

checkpoint = torch.load(exp_dir + 'kasumi_line_ep8000.pth.tar')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
device = torch.device('cpu')


preprocess = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
####################################################
for file in filist:
    imfile = dir +file
    print(imfile)
    im = cv2.imread(imfile,cv2.IMREAD_COLOR)
    print(im.shape)
    height,width,channels = im.shape
    scale = 400/float(width)
    im = cv2.resize(im,(400,int(height*scale)))

    intensor = preprocess (im)
    batch = intensor.reshape(1,*intensor.shape)

    print (batch.shape, batch.dtype)
    
    with torch.no_grad():
        a = model(batch.to(device))
        re = a['out'].argmax(dim=1).data.cpu().numpy()[0]
        re=decode_segmap(re)
        im = cv2.resize(re, dsize=(width, height), interpolation=cv2.INTER_AREA)

        cv2.imwrite(out_dir + file, im)

        #plt.imshow(re)
        #plt.imshow(im, alpha=0.2)
        #plt.savefig ('test-'+file)
