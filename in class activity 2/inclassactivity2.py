import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd

class PreProcess_Data:
    def visualization_images(self, dir_path, nimages):
        fig, axs = plt.subplots(4, 4, figsize=(12, 10))
        dpath = dir_path
        count = 0
        for i in os.listdir(dpath):
            train_class = os.listdir(os.path.join(dpath, i))
            for j in range(nimages):
                img_name = train_class[j]
                img_path = os.path.join(dpath, i, img_name)
                img = cv2.imread(img_path)
                axs[count][j].set_title(i)
                axs[count][j].imshow(img)
            count += 1
        fig.tight_layout()
        plt.show(block=True)

    def preprocess(self, dir_path):
        dpath = dir_path
        train = []
        label = []
        for i in os.listdir(dpath):
            train_class = os.listdir(os.path.join(dpath, i))
            for j in train_class:
                img = os.path.join(dpath, i, j)
                train.append(img)
                label.append(i)
        print('Number of train images: {}\n'.format(len(train)))
        print('Number of train image labels: {}\n'.format(len(label)))

        ret_df=pd.DataFrame({'Image':train,'Labesl':label})
        return train, label ,ret_df
