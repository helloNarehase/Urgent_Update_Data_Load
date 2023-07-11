import cv2
import numpy as np
import pandas as pd

class seedPathArr:
    def __init__(self,csv_file, seed = 0.4) -> None:
        self.seed = seed
        self.pathList = pd.read_csv(csv_file)
        self.spl = int(len(self.pathList)*0.4)
        print(f"all Num : {self.spl}")

    def getDataset(self,num= 0, cropImageCount = 3):
        ass = self.pathList[self.spl*(num):self.spl*(num+1)]
        mxs = len(ass)*cropImageCount
        a = []
        b = []
        for idx in range(len(ass)):
            img_path = ass.iloc[idx, 1]
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask_rle = ass.iloc[idx, 2]
            mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))
            # a.append(cv2.resize(np.array(image), (400,400)))
            for i in range(cropImageCount):
                im, mxk = RandomCrop(image, mask)
                a.append(np.array(im))
                b.append(np.array(mxk, dtype=np.float16))

            print(f"Load {len(a)}/{mxs} : {(len(a)/mxs)*100:3.2f}%", end="\r" if mxs != len(a) else "\n")
        print("finish")
        return np.array(a), np.array(b)

datatset = seedPathArr("train.csv").getDataset(num=1, cropImageCount=4) 
"""
Ver2 
"""
