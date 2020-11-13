import numpy as np

array_reloaded = np.load('C:\\Users\\User\\Desktop\\School\\Intmanlab\\DAGAN\\datasets\\imgds.npy')

print(array_reloaded)
print('\nShape: ',array_reloaded.shape)

from matplotlib import pyplot as plt

plt.imshow(img_array, cmap='gray')
plt.show()
