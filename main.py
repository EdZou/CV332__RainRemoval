import cv2
import numpy as np
from matplotlib import pyplot as plt
from RainRemoval import RRfunc
import argparse
import os
from Dataloader import ImageDataset
import pickle
from PIL import Image
from images2gif import writeGif
import imageio
from tqdm import tqdm

'''
main function of Face Detector Function
developed by Cong Zou, 11/10/2019

To use it, open the terminal in Linux or cmd in windows
enter the directory of main.py, Dataloader and FaceDetector.py
CAUTION: These two .py files should be in the same file folder
imread the img by changing the root
For instance, enter this order in terminal/cmd
python main.py --img.dir D:/files/image/moon.bmp
to find the image.

'''


parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type = str, default = 'images\\rain\\im_real.jpg')
parser.add_argument('--gt_dir', type = str, default = 'images\\gt\\im_0001.png')
parser.add_argument('--res_dir', type = str, default = None)
parser.add_argument('--u_val', type = float, default = 0.01)
parser.add_argument('--epsilon_val',type = float, default = 0.08)
parser.add_argument('--sigma_val',type = float, default = 9)
parser.add_argument('--n_val',type = float, default = 13)
parser.add_argument('--m_val',type = float, default = 85)
parser.add_argument('--lambda_val',type = float, default = 0.0001)
parser.add_argument('--kernel_x',type = int, default = 7)
parser.add_argument('--kernel_y',type = int, default = 7)
parser.add_argument('--alpha',type = float, default = 0.95)
parser.add_argument('--beta',type = float, default = 0.14)
parser.add_argument('--train_mode',action = 'store_true', default = False)


def main(args):
    if not args.kernel_x%2 or not args.kernel_y%2:
        print("Warning: kernel size should be odd number")
    train_dir = os.path.expanduser(args.train_dir)
    
    img = cv2.imread(train_dir).astype('float') / 255
    plt.imshow(img[...,[2,1,0]])
    plt.title('original')
    plt.show()
    kernel = [args.kernel_x, args.kernel_y]
    #dl = ImageDataset(args.train_dir)
    
    rr = RRfunc(kernel, args.u_val, args.epsilon_val)
    b_map = rr.Forward(img)
    
    plt.imshow(b_map, cmap = 'gray')
    plt.title('Binary Map')
    plt.show()
    if args.train_mode:
        #你在这里引用你的类，b_map就是找到的雨点，找到的标记为0
        #python main.py --train_mode进入
        #在这里得到一个alpha和beta
        pass
    else:
        alpha = args.alpha
        beta = args.beta

    #revise image here
    for i in range(len(img)):
        for j in range(len(img[0])):
            if not b_map[i][j]:
                img[i][j][0] += (alpha - 1)*img[i][j][0] + beta
                img[i][j][1] += (alpha - 1)*img[i][j][1] + beta
                img[i][j][2] += (alpha - 1)*img[i][j][2] + beta
    plt.imshow(img[...,[2,1,0]])
    plt.title('Final')
    plt.show()
    '''
    plt.imshow(roi, cmap = 'gray')
    plt.title('Face Part')
    plt.show()
    '''
    
    
    #=====================save images results=============================
    if args.res_dir != None:
        args.res_dir = os.path.expanduser(args.res_dir)
        rpath = ['\\ssd.gif', '\\cc.gif','ncc.gif']
            
        if os.path.exists(args.res_dir) == False:
            os.makedirs(args.res_dir)
        '''
        frames = []

        for img_name in dl.datapaths:
            frames.append(imageio.imread(img_name))

        gifpath = args.res_dir + '\\girl.gif'
        imageio.mimsave(gifpath, frames, 'GIF', duration = 0.1)
        '''
        #save gif and images
        frames = []
        dirs = ['SSD', 'CC', 'NCC','cvpr98','ncvpr98']
        pbar = tqdm(total = len(dl.datapaths), desc = 'Detecting Face...')
        if os.path.exists(args.res_dir + '\\'+ dirs[args.comp_mode-1]) == False:
            os.makedirs(args.res_dir + '\\'+ dirs[args.comp_mode-1])
        for img_name in dl.datapaths:
            fimg = cv2.imread(img_name)
            resimg = fd.Forward(fimg)
            imgpath = args.res_dir + '\\'+ dirs[args.comp_mode-1] + '\\' + img_name[-8:]
            cv2.imwrite(imgpath, resimg)
            frames.append(resimg[...,[2,1,0]])
            pbar.update()
        pbar.close()

        gifpath = args.res_dir + '\\' + dirs[args.comp_mode-1] + '.gif'
        imageio.mimsave(gifpath, frames, 'GIF', duration = 0.1)
        
            


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

    


