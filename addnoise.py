from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img 
import random
import numpy as np

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
        
def add_noise(img):
    '''Add random noise to an image'''
 
    noised_image = img + 3.5 * img.std() * np.random.random(img.shape)

  
    return noised_image
# Initialising the ImageDataGenerator class. 
# We will pass in the augmentation parameters in the constructor. 
datagen = ImageDataGenerator( 
 
        preprocessing_function=add_noise,
        ) 
    
# Loading a sample image  
img = load_img('D:/PYTHON/螺桿單元noise/sR3-50-33/1.PNG')  
# Converting the input sample image to an array 
x = img_to_array(img) 
# Reshaping the input image 
x = x.reshape((1, ) + x.shape)  
   
# Generating and saving 5 augmented samples  
# using the above defined parameters.  
i = 0
for batch in datagen.flow(x, batch_size = 1, 
                          save_to_dir ='D:/PYTHON/螺桿單元noise/sR3-50-33',  
                          save_prefix ='image', save_format ='jpeg'): 
    i += 1
    if i > 4: 
        break
