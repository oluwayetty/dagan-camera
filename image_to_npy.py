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

def load_dataset(path, mode):
    dirs = os.listdir(path)
    # Append images to a list
    data =[]

    for item in dirs:
        
        if item.split(".")[-1] == 'jpg':
            im = Image.open("\\".join([path,item])).convert("RGB")
            im = im.resize((new_width, new_height), Image.ANTIALIAS)
            im = np.array(im)
            data.append(im)
            image_arr = np.array(data)
    batch1, batch2, batch3 = np.array_split(image_arr, 3)

    np.save("C:\\Users\\User\\Desktop\\School\\Intmanlab\\DAGAN\\datasets\\{}_1.npy".format(mode),np.array((batch1)))
    np.save("C:\\Users\\User\\Desktop\\School\\Intmanlab\\DAGAN\\datasets\\{}_2.npy".format(mode),np.array((batch2)))
    np.save("C:\\Users\\User\\Desktop\\School\\Intmanlab\\DAGAN\\datasets\\{}_3.npy".format(mode),np.array((batch3)))
    
    return batch1.astype(np.float32), batch2.astype(np.float32), batch3.astype(np.float32)

if __name__ == "__main__":

    load_dataset("C:\\Users\\User\\Desktop\\School\\Intmanlab\\cyclegan\\all_images\\samples\\camA", mode= "train_cam1")
    load_dataset("C:\\Users\\User\\Desktop\\School\\Intmanlab\\cyclegan\\all_images\\samples\\camB", mode= "val_cam2")
    load_dataset("C:\\Users\\User\\Desktop\\School\\Intmanlab\\cyclegan\\all_images\\samples\\camC", mode= "test_cam3")
    # imgsetD = load_dataset("C:\\Users\\User\\Desktop\\School\\Intmanlab\\cyclegan\\all_images\\samples\\camD")
    # imgsetE = load_dataset("C:\\Users\\User\\Desktop\\School\\Intmanlab\\cyclegan\\all_images\\samples\\camE")

    # train_arr = np.array((imgsetA))
    # val_arr = np.array((imgsetB))
    # test_arr = np.array((imgsetC))
    # ty = np.stack((imgsetA,imgsetB))
    # 
    # np.save("C:\\Users\\User\\Desktop\\School\\Intmanlab\\DAGAN\\datasets\\train_arr.npy",train_arr)
    # np.save("C:\\Users\\User\\Desktop\\School\\Intmanlab\\DAGAN\\datasets\\val_arr.npy",val_arr)
    # np.save("C:\\Users\\User\\Desktop\\School\\Intmanlab\\DAGAN\\datasets\\test_arr.npy",test_arr)
