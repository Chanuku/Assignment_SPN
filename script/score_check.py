import cv2
import numpy as np
import os

dir = '../Downloads/coloring_test/segmentation/present/Trap_ball/output/'
gt_dir = '../Downloads/coloring_test/segmentation/present/gt/'

filename = os.listdir(dir)
total_score=0
for i in filename:
    im = cv2.imread(dir+i)
    gt = cv2.imread(gt_dir+i)
    (width,height,channel) = gt.shape
    point = 0
    divide_point =0
    color_index =np.array([(0,255,0),(255,0,0),(128,0,128),(0,0,0),(0,0,255)])
    print(i)
    for i in range (width):
        for j in range (height):
            if (gt[i][j] != color_index[3]).any():
                divide_point += 1
                if (gt[i][j] == im[i][j]).all():
                    point += 1
    
    total_score += point/(divide_point)
    #print(total_score)

total_score = total_score/len(filename)
print("total score is "+str(total_score))