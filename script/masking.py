from PIL import Image 
import os 
import cv2 
import numpy as np
in_dir = '../Downloads/coloring_test/segmentation/test/final_test/'
#out_dir = '../Downloads/coloring_test/segmentation/train_d
out_dir = '../Downloads/coloring_test/segmentation/test/final_test_thick/'


imgs = list(sorted(os.listdir(os.path.join(in_dir))))
for filename in imgs:
    img = Image.open(in_dir  + filename).convert("RGB") 
    name, ext = os.path.splitext (filename)
    #mask_name = name + '_tf.png'
    print(filename)

    #output = Image.open(out_dir+ mask_name).convert("RGB")
    output = Image.open(in_dir  + filename).convert("RGB") 

    masking = output.load()

    (width,height) = img.size
    black = (0,0,0)
    yellow = (255,255,0)
    #output = img.load()
    for y in range(0,height-1):
        for x in range(0,width-1):
            r, g, b = img.getpixel((x,y))
            #print(str(x)+','+str(y))

            if r<5 and g < 5 and b< 5:
                masking[x,y] = black

                masking[x,y+1] = black
                masking[x+1,y] = black

                masking[x-1,y] = black
                masking[x,y-1] = black
    '''
    for x in range(0,width):
        masking[x,0] = (0,0,0)
        masking[x,1] = (0,0,0)
        masking[x,2] = (0,0,0)
        masking[x,3] = (0,0,0)
        masking[x,716] = (0,0,0)
        masking[x,719] = (0,0,0)
        masking[x,718] = (0,0,0)
        masking[x,717] = (0,0,0)
    '''     



    output.save(out_dir + filename)