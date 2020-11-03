from PIL import Image
import os, sys
# from scipy.misc import imsave
import time
import numpy as np

'''
Converts all images in a directory to '.npy' format.
Use np.save and np.load to save and load the images.
'''

new_width  = 256
new_height = 256

def load_dataset(path):
    dirs = os.listdir(path)
    # Append images to a list
    x_train=[]

    for item in dirs:
        if item.split(".")[-1] == 'jpg':
            im = Image.open("\\".join([path,item])).convert("RGB")
            im = im.resize((new_width, new_height), Image.ANTIALIAS)
            im = np.array(im)
            x_train.append(im)
            image_arr = np.array(x_train)
    return image_arr.astype(np.float32)


if __name__ == "__main__":

    imgsetA = load_dataset("C:\\Users\\User\\Desktop\\School\\Intmanlab\\cyclegan\\all_images\\samples\\camA")
    imgsetB = load_dataset("C:\\Users\\User\\Desktop\\School\\Intmanlab\\cyclegan\\all_images\\samples\\camB")
    imgsetC = load_dataset("C:\\Users\\User\\Desktop\\School\\Intmanlab\\cyclegan\\all_images\\samples\\camC")
    imgsetD = load_dataset("C:\\Users\\User\\Desktop\\School\\Intmanlab\\cyclegan\\all_images\\samples\\camD")
    imgsetE = load_dataset("C:\\Users\\User\\Desktop\\School\\Intmanlab\\cyclegan\\all_images\\samples\\camE")

    train_arr = np.array((imgsetA))
    val_arr = np.array((imgsetB)).astype(np.float32)
    test_arr = np.array((imgsetD, imgsetE)).astype(np.float32)
    # ty = np.stack((imgsetA,imgsetB))
    # import ipdb; ipdb.set_trace()
    np.save("C:\\Users\\User\\Desktop\\School\\Intmanlab\\DAGAN\\datasets\\train_arr.npy",train_arr)
    np.save("C:\\Users\\User\\Desktop\\School\\Intmanlab\\DAGAN\\datasets\\val_arr.npy",val_arr)
    np.save("C:\\Users\\User\\Desktop\\School\\Intmanlab\\DAGAN\\datasets\\test_arr.npy",test_arr)
