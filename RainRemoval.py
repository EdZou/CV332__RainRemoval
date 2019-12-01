from matplotlib import pyplot as plt
import matplotlib.colors
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import math
import cv2


'''
Face Detector
developed by Cong Zou, 11/10/2019
'''

class RRfunc(object):
    def __init__(self, kernel, u, epsilon):
        super(RRfunc, self).__init__()
        #all pixels in image should be normalized into [0,1], [B,G,R]
        self.kernel = kernel
        self.u = u
        self.epsilon = epsilon

    def Rain_detect(self, img):
        origin = self.__Gpadding(np.copy(img), self.kernel)
        b_map = np.ones([len(img), len(img[0])])
        total = self.kernel[0]*self.kernel[1]*5
        count = 0
        pbar = tqdm(total = len(b_map)*len(b_map[0]), desc = 'Detecting Rain...')
        for i in range(len(b_map)):
            for j in range(len(b_map[0])):
                tempb = np.sum(origin[i-self.kernel[0]:i+self.kernel[0], j-self.kernel[1]:j+self.kernel[1],0]) + np.sum(origin[i-int(self.kernel[0]/2):i+int(self.kernel[0]/2)+1, j-int(self.kernel[1]/2):j+int(self.kernel[1]/2)+1,0])
                tempg = np.sum(origin[i-self.kernel[0]:i+self.kernel[0], j-self.kernel[1]:j+self.kernel[1],1]) + np.sum(origin[i-int(self.kernel[0]/2):i+int(self.kernel[0]/2)+1, j-int(self.kernel[1]/2):j+int(self.kernel[1]/2)+1,1])
                tempr = np.sum(origin[i-self.kernel[0]:i+self.kernel[0], j-self.kernel[1]:j+self.kernel[1],2]) + np.sum(origin[i-int(self.kernel[0]/2):i+int(self.kernel[0]/2)+1, j-int(self.kernel[1]/2):j+int(self.kernel[1]/2)+1,2])
                if origin[i+self.kernel[0],j+self.kernel[1],0] > tempb/total + self.u and \
                   origin[i+self.kernel[0],j+self.kernel[1],1] > tempg/total + self.u and \
                   origin[i+self.kernel[0],j+self.kernel[1],2] > tempr/total + self.u:
                    b_map[i][j] = 0
                    count += 1
                pbar.update()
        pbar.close()
        print(count)
        '''
        plt.imshow(b_map, cmap = 'gray')
        plt.title('rough')
        plt.show()
        '''
        b_map = self.__3dto2d(b_map, img)
        return b_map


    def __3dto2d(self, b_map, img):
        count = 0
        for i in range(len(b_map)):
            for j in range(len(b_map[0])):
                if not b_map[i][j]:
                    #3D to 2D
                    tempsum = np.sum(img[i][j]) / 3
                    u = (2*tempsum - img[i][j][0]-img[i][j][1])/tempsum
                    v = max((tempsum-img[i][j][1])/tempsum, (tempsum-img[i][j][0])/tempsum)
                    if (u**2 + v**2)**0.5 > self.epsilon:
                        b_map[i][j] = 1
                        count += 1
        print(count)
        return b_map
                


    def __Gpadding(self, img = [], k_size = [7, 7]):
        img_h = len(img)
        img_w = len(img[0])
        pad_h = int(k_size[0])
        pad_w = int(k_size[1])

        #padding [x,y,z] to [x+2*pad_h,y,z]
        temp = img[1: pad_h+1,...]
        img = np.append(temp[::-1,...], img, axis = 0)
        temp = img[-pad_h-1:-1,...]
        img = np.append(img,temp[::-1,...], axis = 0)

        #padding [x+2*pad_h,y,z] to [x+2*pad_h,y+2*pad_w,z]
        temp = img[:,1:pad_w+1,:]
        img = np.append(temp[:,::-1,:], img, axis = 1)
        temp = img[:,-pad_w-1:-1,:]
        img = np.append(img, temp[:,::-1,:], axis = 1)

        return img 


    def Forward(self, img):
        b_map = self.Rain_detect(img)
        return b_map

        
        


        
'''
    def __call__(self):
        return forward()
'''




        










        





            
            
