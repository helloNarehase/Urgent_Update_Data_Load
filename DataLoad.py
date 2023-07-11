import cv2
import numpy as np
import pandas as pd
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def RandomCrop(image, mask, crop_Volum = (224,224)):
    maxy, maxx, c = image.shape
    start_y = np.random.randint(0, maxy-crop_Volum[0])
    start_x = np.random.randint(0, maxx-crop_Volum[1])
    img = image[start_x:start_x+crop_Volum[1], start_y:start_y+crop_Volum[0]]
    rmask = mask[start_x:start_x+crop_Volum[1], start_y:start_y+crop_Volum[0]]
    return img, rmask

class loadDataset:
    def __init__(self, csv_file) -> None:
        self.data = pd.read_csv(csv_file)
        print(f"len : {len(self.data)}")
    def getDataset(self, k = 1.0, cropImageCount = 3,f = True):
        a = []
        b = []
        mx = len(self.data)
        mxs = int(mx*k)*cropImageCount
        for idx in range(int(mx*k)):
            print(f"Load {(idx+1)*cropImageCount}/{mxs} : {((idx+1)/mxs)*100:3.2f}%", end="\r" if mxs-1 != idx else "\n")
            img_path = self.data.iloc[idx+(0 if f else int(mx*k)), 1]
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask_rle = self.data.iloc[idx+(0 if f else int(mx*k)), 2]
            mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))
            # a.append(cv2.resize(np.array(image), (400,400)))
            for i in range(cropImageCount):
                im, mxk = RandomCrop(image, mask)
                a.append(np.array(im))
                b.append(np.array(mxk, dtype=np.int8))
        print("finish")
        return np.array(a), np.array(b)

datatset = loadDataset('train.csv').getDataset(k = 0.4, cropImageCount=5)

Xtran, Xtest = datatset[0][:3100], datatset[0][3100:]
# datatset = onehotc(datatset[1])
datatset = datatset[1]
datatset = np.expand_dims(datatset,axis=-1)
Ytran, Ytest = datatset[:3100], datatset[3100:]
print(len(Ytran))
datatset = None
