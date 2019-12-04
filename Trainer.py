import numpy as np
import cv2
import math
import copy
from tqdm import tqdm

N = 3
Theta = 2*N+1
class Trainer():
    def Aprox(self, img, bmap):
        rows, cols = bmap.shape
        qs = np.zeros((rows, cols, 3))

        pbar = tqdm(total = rows*cols, desc = 'Doing Approximation...')

        for i in range(rows):
            for j in range(cols):
                pbar.update()
                if bmap[i, j] == 1.0:
                    continue
                p = img[i, j]
                cand = img[i-N:i+N+1,j-N:j+N+1,:]
                mask = bmap[i-N:i+N+1,j-N:j+N+1]
                rrow, ccol = mask.shape
                cand = np.reshape(cand, (rrow*ccol,3))
                mask = np.reshape(mask, rrow*ccol)

                non_rain_index = np.where(mask == 1.0)
                hks = []
                wk2s = []
                for index in non_rain_index[0]:
                    # calculate wk
                    hk = cand[index]
                    try:
                        nu = np.linalg.norm(hk-p, ord=2)
                    except:
                        nu = 0
                    wk = math.exp(-nu/(Theta**2))
                    hks.append(hk)
                    wk2s.append(wk**2)
                wk2s = np.reshape(np.array(wk2s), (len(wk2s),1))
                q = np.sum(wk2s*hks, axis=0)/np.sum(wk2s)
                try:
                    qs[i, j] = q
                except:
                    qs[i, j] = img[i, j]
        pbar.close()
        qs_2D = np.reshape(qs, (rows*cols,3))
        return qs, qs_2D