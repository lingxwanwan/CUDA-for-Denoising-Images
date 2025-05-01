
import os
import cv2
import numpy as np
from skimage import io
from skimage import img_as_ubyte
from matplotlib import pyplot as plt

# Directory containing the TIFF images
data_dir = 'data'
preprocessed_dir = os.path.join(data_dir, 'preprocessed')
viewer_dir = os.path.join(data_dir,'viewer')


# Function to preprocess images and save them
def preprocess_images(data_dir, preprocessed_dir):
    # Create the directory for preprocessed images if it doesn't exist
    os.makedirs(preprocessed_dir, exist_ok=True)
    #os.makedirs(viewer_dir,exist_ok=True)
    
    images = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.tiff'):
            # Load image
            image_path = os.path.join(data_dir, filename)
            image = io.imread(image_path)
            
            # Convert to grayscale if the image is in color
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Normalize image
            normalized_image = image / 255.0
            
            # Append to the list of preprocessed images
            images.append(normalized_image)
            
            # Save the preprocessed image
            
            #processed_image_path = os.path.join(viewer_dir,os.path.splitext(filename)[0] + '.png')
            preprocessed_image_path2 = os.path.join(preprocessed_dir,filename)
            #io.imsave(processed_image_path, img_as_ubyte(normalized_image))
            io.imsave(preprocessed_image_path2,img_as_ubyte(normalized_image))
            
            # Display the image
            plt.imshow(normalized_image, cmap='gray')
            plt.title(f'Normalized Image - {filename}')
            #plt.show()
    
    return images

preprocess_images(data_dir, preprocessed_dir)
