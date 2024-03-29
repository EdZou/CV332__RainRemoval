import math
import cv2
import copy
from tqdm import tqdm
import numpy as np
from RainRemoval import RRfunc
from Trainer import Trainer
from matplotlib import pyplot as plt

def genNames(img_name, u_val, epsilon_val):
    full_name = '{0}-{1}-{2}'.format(img_name, u_val, epsilon_val)
    img_path = 'C:/Users/Donnie/Desktop/NU/EE332/FP/LM/data/img/{0}.jpg'.format(img_name)
    b_path = 'C:/Users/Donnie/Desktop/NU/EE332/FP/LM/data/array/{0}.b'.format(full_name)
    q_path = 'C:/Users/Donnie/Desktop/NU/EE332/FP/LM/data/array/{0}.q'.format(full_name)
    return img_path, b_path, q_path

def detectRain(img, b_path):
    rr = RRfunc(img, kernel, 0.01, 0.10)
    b_map = rr.Forward()
    np.savetxt(b_path, b_map)

def main():

    img_path, b_path, q_path = genNames('frame0', 0.01, 0.1)

    tr = Trainer()
    img = cv2.imread(img_path)
    # img = img[0:150,0:150,:]
    rows, cols, _ = img.shape
    kernel = [7, 7]

    # ### Detect rain 
    b_map = RRfunc(kernel, 0.01, 0.1).Forward(img)
    np.savetxt(b_path, b_map)

    # ### Do approximation
    b_map = np.loadtxt(b_path)

    plt.imshow(b_map)
    plt.show()

    qs_3D, qs_2D = tr.Aprox(img, b_map)
    np.savetxt(q_path, qs_2D)

    qs = np.loadtxt(q_path)
    qs = np.reshape(qs, (rows, cols, 3))
    # cv2.imshow('', qs)
    # cv2.waitKey(0)
    # plt.imshow(qs)
    # plt.title('Ref')
    # plt.show()
    # ###Deraining
    derain = copy.deepcopy(img)
    for i in range(rows):
        for j in range(cols):
            if b_map[i, j] == 1.0:
                continue
            derain[i,j] = qs[i, j]
            # derain[i,j] = (0,0,0)
    # print(derain)
    # derain = (derain*255/derain.max()).astype(np.uint8)
    derain.astype(np.uint8)
    cv2.imshow('', derain)
    cv2.waitKey(0)  

    img = img/255
    qs = qs/qs.max()

    alpha = [0,0,0]
    beta = [0,0,0]
    for color in range(3):
        imgChannel = img[:,:,color]
        qChannel = qs[:,:,color]
        approximationValues = []
        intensityValues = []

        # pbar = tqdm(total = rows*cols, desc = 'Collecting...')

        for i in range(rows):
            for j in range(cols):
                if b_map[i, j] == 1.0 or (i*j)%1!=0:
                    continue
                intensityValues.append(imgChannel[i, j])
                approximationValues.append(qChannel[i, j])
        dk = np.array(intensityValues)
        qk = np.array(approximationValues)
        k = len(intensityValues)
        lamda = 0
        nu = (1/k)*np.sum(dk*qk)-np.average(dk)*np.average(qk)
        de = np.var(qk)+lamda
        print(nu, de)
        alpha[color] = nu/de
        beta[color] = np.average(dk) - alpha[color]*np.average(qk)

    print(alpha, beta)
    plt.scatter(approximationValues, intensityValues, alpha=0.6)
    plt.show()  


    derain = copy.deepcopy(img)
    for color in range(3):
        for i in range(rows):
            for j in range(cols):
                if b_map[i, j] == 1.0:
                    continue
                derain[i,j,color] = (img[i,j,color]-beta[color])/alpha[color]
    # print(derain)
    cv2.imshow('', derain)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()