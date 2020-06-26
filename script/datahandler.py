from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
import cv2
import torch
import codecs
from torchvision import transforms, utils


class TSDataset(Dataset):
    """ Transportation Sign Plate """

    def __init__(self, root_dir, imageFolder, maskFolder, transform=None, seed=None, fraction=None, subset=None, 
                 isize=400):
        """
        Args:
            root_dir (string): Directory with all the images and should have the following structure.
            root
            --Images
            -----Img 1
            -----Img N
            --Mask
            -----Mask 1
            -----Mask N
            imageFolder (string) = 'Images' : Name of the folder which contains the Images.
            maskFolder (string)  = 'Masks : Name of the folder which contains the Masks.

        """
        self.show = False # for debug
        self.cnt = 0
        self.root_dir = root_dir
        self.transform = transform
        self.isize = isize
        self.imfiles = self._choose_imgs (self.root_dir)
        return

    def __len__(self):
        return len(self.imfiles)

    def __getitem__(self, idx):
        """ """
        aug_times = 0 #added
        img_name, msk_name = self.imfiles[idx]
        image = cv2.imread (img_name)[:,:,::-1] # BGR -> RGB
        mask =  cv2.imread(msk_name)[:,:,::-1] # BGR -> RGB
        if self.show:
            print ('--- ', str(img_name))
            print ('--- ', str(msk_name))

        # resize to prototype: self.isize x self.isize
        
        rows, cols, channels = image.shape
        if rows != self.isize or cols != self.isize:
            if cols > rows:
                scale = self.isize / float(image.shape[1])
                image = cv2.resize(image, (self.isize, int(rows*scale)))
                mask  = cv2.resize(mask,  (self.isize, int(rows*scale)))
            else:
                scale = self.isize / float(image.shape[0])
                image = cv2.resize(image, (self.isize, int(cols*scale)))
                mask  = cv2.resize(mask,  (self.isize, int(cols*scale)))
                #pass
                #print ('Not implemented.')
                #import sys
                #sys.exit()
                
        #

        # ----- data augmentation -----
        do_gbflip = np.random.random() > 0.8
        if do_gbflip: # RGB -> RBG, swap G and B, image only
            r, g, b = cv2.split(image)
            image = cv2.merge([r, b, g])
            aug_times += 1
        do_flip = np.random.random() > 0.8
        if do_flip: # do y-axis flip
            image = cv2.flip(image, 1) # image flip
            mask = cv2.flip(mask, 1) # mask flip
            r,g,b = cv2.split(mask) # in mask, green on the left!
            mask = cv2.merge( [r, b, g])
            aug_times += 1
        #
        rcenter = (image.shape[1]//2, image.shape[0]//2)
        do_scale = np.random.random() > 0.8
        scale = 1.
        if do_scale:
            scale = np.random.uniform (0.9, 1.1)
            tm = cv2.getRotationMatrix2D(rcenter, 0., scale=scale)
            image = cv2.warpAffine(image, tm, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
            mask  = cv2.warpAffine(mask,  tm, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)
            aug_times += 1
        #
#        tx = np.random.randint(0,10, size=(2))

        do_rot = np.random.random() > 0.5
        if do_rot:
            angle = np.random.uniform(-10, 10)
            rmat = cv2.getRotationMatrix2D( rcenter, angle, 1. )
            image = cv2.warpAffine(image, rmat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
            mask  = cv2.warpAffine(mask,  rmat, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)
            aug_times += 1

        if self.show:
            cv2.imwrite(f'{self.cnt}-image_tx.png', image[:,:,::-1])
            cv2.imwrite(f'{self.cnt}-mask_tx.png',  mask[:,:,::-1])
            self.cnt += 1
        # -----

        mask = self._convert2labelImage(mask)

        # copy to standard shape
        bimg = np.ascontiguousarray(np.zeros((self.isize, self.isize, 3), dtype=np.uint8))
        #bimg = np.ascontiguousarray(np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8))


        bmsk = np.ascontiguousarray(np.zeros((self.isize, self.isize), dtype=np.int))
        #print(image.shape)
        bimg[0:image.shape[0], 0:image.shape[1]] = image
        bmsk[0:mask.shape[0], 0:mask.shape[1]] = mask

        sample = { 'image': self.transform(bimg), 'mask': torch.LongTensor(bmsk) }
        #print("augtimes is " + str(aug_times))
 #       print ('--------', idx, sample['image'].shape, sample['mask'].shape)
        return sample

    def _convert2labelImage(self, mask):
        
        r = (mask[:,:,0] >= 200) & (mask[:,:,1] < 128) & (mask[:,:,2] < 128)
        #print (r.shape)
        g = ~r & (mask[:,:,0] < 128) & (mask[:,:,1] >= 200) & (mask[:,:,2] < 128)
        b = (~(r | g)) & (mask[:,:,0] < 128) & (mask[:,:,1] < 128) & (mask[:,:,2] >= 200)
        p = (~(r | g | b)) & (mask[:,:,0] >= 125) & (mask[:,:,1] < 128) & (mask[:,:,2] >= 125)
        #y = (~(r | g | b | p)) & (mask[:,:,0] >= 128) & (mask[:,:,1] >= 128) & (mask[:,:,2] < 128)

        label_img = 1*r + 2*g + 3*b + 4*p #+ 5*y
        
        return label_img.astype(np.int)
        
        '''
        y = (mask[:,:,0] >= 128) & (mask[:,:,1] >= 128) & (mask[:,:,2] < 128)
        p = ~y & (mask[:,:,0] >= 128) & (mask[:,:,1] < 128) & (mask[:,:,2] >= 128)
        s = (~(y | p)) & (mask[:,:,0] < 128) & (mask[:,:,1] >= 128) & (mask[:,:,2] >= 128)
        r = (~(y | p | s)) & (mask[:,:,0] >= 128) & (mask[:,:,1] < 128) & (mask[:,:,2] < 128)
        g = (~(y | p | s | r)) & (mask[:,:,0] < 128) & (mask[:,:,1] >= 128) & (mask[:,:,2] < 128)
        b = (~(y | p | s | r | g)) &(mask[:,:,0] < 128) & (mask[:,:,1] < 128) & (mask[:,:,2] >= 128)
        '''

        #label_img = 1*r + 2*g + 3*b + 4*y + 5*p + 6*s
        #label_img = 1*r + 2*g + 3*b + 4*p


        #f = codecs.open("log.txt","w")
        #print(label_img.shape)
        #f.write(str(label_img.tolist()))
        #return label_img.astype(np.int)
        

        

    def _choose_imgs(self, dir):
        jpgs = []
        fdir = dir + '/image/'
        mdir = dir + '/mask/'
        filelist = os.listdir (fdir)
        masklist = os.listdir (mdir)
        #print (masklist)
        for f in filelist:
            name, ext = os.path.splitext (f)
            #print ('split: ', name, ext)
            if ext == '.png': # image file
                mf = name + '_tf.png'
                #print (f, mf)
                if mf in masklist:
                    jpgs.append((fdir + f, mdir + mf))
        #
        return jpgs
# TSDataSet




def get_dataloader_single_folder(data_dir, imageFolder='image', maskFolder='mask', fraction=0.2, batch_size=4):
    """
        Create training and testing dataloaders from a single folder.
    """
    preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
    data_transforms = {
        'Train': preprocess,
        'Test':  preprocess
    }

    image_datasets = {x: TSDataset(data_dir, 
                                    imageFolder=imageFolder, 
                                    maskFolder=maskFolder, 
                                    seed=100, fraction=fraction, subset=x, 
                                    transform=data_transforms[x]
                                    )
                         for x in ['Train', 'Test']
                    }
    dataloaders = {x: DataLoader(image_datasets[x], 
                                    batch_size=batch_size, 
                                    shuffle=True,
                                    drop_last=True)
                      for x in ['Train', 'Test']
                   }

    return dataloaders
