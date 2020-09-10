import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
import PIL


def image_to_tensor(image):
    return tf.keras.preprocessing.image.img_to_array(image, data_format=None, dtype=None)

def de_noise(path,noise=1,plot=True,n_bins=1e2):
    image = PIL.Image.open(path).convert("L")
    tensor = image_to_tensor(image)
    data = tensor/np.max(tensor)
    exp = np.exp(-data)
    norm = exp/np.max(exp)
    n, bins = np.histogram(norm, bins=int(n_bins))
    mids = 0.5*(bins[1:] + bins[:-1])
    mean = np.average(norm)
    var = np.average((mids-mean)**2,weights=n)
    vmax = (mean - var) * noise
    print(vmax)
    if plot == True:
        fig,axes = plt.subplots(2,2,figsize=(20,10))
        axes[0][0].imshow(data, cmap='gray', vmin=0., vmax=1, label='original')
        axes[0][0].set_title('original')
        axes[0][1].hist(norm.flatten(),bins=100,log=False, label='hist')   
        axes[0][1].set_title('hist')   
        axes[1][1].imshow(norm, cmap='gray', vmin=0, vmax=vmax, label='1/denoised')
        axes[1][1].set_title('1/denoised')
        axes[1][0].imshow(-np.log(norm), cmap='gray', vmin=-np.log(vmax), vmax=1, label='denoised')
        axes[1][0].set_title('denoised')
        plt.show()

    norm[np.where(norm > vmax)]=1.
    return norm
