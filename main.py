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
parser.add_argument('--train_dir', type = str, default = 'images\\gif\\rain\\frame14.jpg')
parser.add_argument('--gif_dir', type = str, default = 'images\\gif\\rain')
parser.add_argument('--res_dir', type = str, default = None)
parser.add_argument('--u_val', type = float, default = 0.01)
parser.add_argument('--epsilon_val',type = float, default = 0.2)
parser.add_argument('--sigma_val',type = float, default = 9)
parser.add_argument('--n_val',type = float, default = 13)
parser.add_argument('--m_val',type = float, default = 85)
parser.add_argument('--lambda_val',type = float, default = 0.0001)
parser.add_argument('--kernel_x',type = int, default = 7)
parser.add_argument('--kernel_y',type = int, default = 7)
parser.add_argument('--alpha',type = float, default = 0.92)
parser.add_argument('--beta',type = float, default = 0.1)
parser.add_argument('--train_mode',action = 'store_true', default = False)
parser.add_argument('--gif_mode', action = 'store_true', default = False)
parser.add_argument('--his_bmap', type = int, default = 1)


def main(args):
    if not args.kernel_x%2 or not args.kernel_y%2:
        print("Warning: kernel size should be odd number")
    train_dir = os.path.expanduser(args.train_dir)
    
    #img = cv2.imread(train_dir).astype('float') / 255
    '''
    plt.imshow(img[...,[2,1,0]])
    plt.title('original')
    plt.show()
    '''
    kernel = [args.kernel_x, args.kernel_y]
    #dl = ImageDataset(args.train_dir)
    
    rr = RRfunc(kernel, args.u_val, args.epsilon_val)
    #b_map = rr.Forward(img)

    '''
    plt.imshow(b_map, cmap = 'gray')
    plt.title('Binary Map')
    plt.show()
    '''
    if args.train_mode:
        #你在这里引用你的类，b_map就是找到的雨点，找到的标记为0
        #python main.py --train_mode进入
        #在这里得到一个alpha和beta
        pass
    else:
        alpha = args.alpha
        beta = args.beta

    #revise image here
    def img_revise(img, b_map):
        res = np.copy(img)
        for i in range(len(img)):
            for j in range(len(img[0])):
                if not b_map[i][j]:
                    res[i][j][0] = (img[i][j][0] - beta)/alpha if (img[i][j][0] - beta)/alpha <=1 else 1.0
                    res[i][j][0] = res[i][j][0] if img[i][j][0] >= 0 else 0
                    res[i][j][1] = (img[i][j][1] - beta)/alpha if (img[i][j][1] - beta)/alpha <=1 else 1.0
                    res[i][j][1] = res[i][j][1] if img[i][j][1] >= 0 else 0
                    res[i][j][2] = (img[i][j][2] - beta)/alpha if (img[i][j][2] - beta)/alpha <=1 else 1.0
                    res[i][j][2] = res[i][j][2] if img[i][j][2] >= 0 else 0
        return res

    #concat images here
    def img_concat(img, b_map, res):
        b_map = np.reshape(b_map, [len(b_map), len(b_map[0]), 1])
        temp = np.append(b_map, b_map, axis = 2)
        b_map = np.append(temp, b_map, axis = 2)
        conc = np.concatenate((img, b_map), axis = 1)
        conc = np.concatenate((conc, res), axis = 1)
        return conc
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
        if args.gif_mode:
            args.gif_dir = os.path.expanduser(args.gif_dir)
            dl = ImageDataset(args.gif_dir)
            if args.his_bmap:
                his_len = 10 if len(dl.datapaths) > 10 else len(dl.datapaths)
                pbar = tqdm(total = his_len, desc = 'Preparing Historical B_MAP...')
                his_bmap = np.array([])
                for path in dl.datapaths[:his_len]:
                    temp = cv2.imread(path).astype('float') / 255
                    if not len(his_bmap):
                        his_bmap = rr.Forward(np.copy(temp))
                    else:
                        his_bmap += rr.Forward(np.copy(temp))
                    pbar.update()
                pbar.close()
                his_bmap -= his_len
                his_bmap = np.abs(his_bmap) - (his_len-1)
                his_bmap = np.maximum(his_bmap, 0)
                '''
                plt.imshow(his_bmap, cmap = 'gray')
                plt.title('Historical BMAP')
                plt.show()
                '''
                
            frames = []
            dirs = 'rain5'
            pbar = tqdm(total = len(dl.datapaths), desc = 'Making Gif...')
            dirpath = args.res_dir + '\\' + 'frames'+ '_' + dirs
            if os.path.exists(dirpath) == False:
                os.makedirs(dirpath)
            for img_name in dl.datapaths:
                fimg = cv2.imread(img_name).astype('float') / 255
                imgpath = dirpath + '\\' + img_name[-11:]
                b_map = rr.Forward(np.copy(fimg))
                if args.his_bmap:
                    b_map += his_bmap
                    b_map = np.clip(b_map, 0, 1)
                resimg = img_revise(fimg, b_map)
                resimg = cv2.GaussianBlur(resimg, (3,3), 0)
                conc = img_concat(fimg, b_map, resimg)
                plt.imshow(conc[...,[2,1,0]])
                plt.savefig(imgpath)
                frames.append(conc[...,[2,1,0]])
                pbar.update()
            pbar.close()

            gifpath = args.res_dir + '\\' + dirs + '.gif'
            imageio.mimsave(gifpath, frames, 'GIF', duration = 0.1)
        
            


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

    


