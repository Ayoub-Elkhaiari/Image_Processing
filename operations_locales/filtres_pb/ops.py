import numpy as np
import cv2
import matplotlib.pylab as plt


def moyenneur(image_path, kernel_size):
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    
    
    kernel_5 = np.array([
                        [1/25, 1/25, 1/25, 1/25, 1/25],
                        [1/25, 1/25, 1/25, 1/25, 1/25],
                        [1/25, 1/25, 1/25, 1/25, 1/25],
                        [1/25, 1/25, 1/25, 1/25, 1/25],
                        [1/25, 1/25, 1/25, 1/25, 1/25]
                                        ])
    kernel_3 = np.array([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
    ])
   

    if kernel_size == 3:
        img_filtered = cv2.filter2D(img, -1, kernel_3)
    elif kernel_size == 5:
        img_filtered = cv2.filter2D(img, -1, kernel_5)
    else :
        print("not available !!!")

    # Display the original and filtered images
    # cv2.imshow('Image originale', img)
    # cv2.imshow('Image resultat', img_filtered)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img, img_filtered
    
    
def gaussian(image_path, kernel_size):
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    
    
    kernel_5 = np.array([
                        [1/256, 4/256, 6/256, 4/256, 1/256],
                        [4/256, 16/256, 24/256, 16/256, 4/256],
                        [6/256, 24/256, 36/256, 24/256, 6/256],
                        [4/256, 16/256, 24/256, 16/256, 4/256],
                        [1/256, 4/256, 6/256, 4/256, 1/256]
                                        ])
    kernel_3 = np.array([
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16]
    ])
   

    if kernel_size == 3:
        img_filtered = cv2.filter2D(img, -1, kernel_3)
    elif kernel_size == 5:
        img_filtered = cv2.filter2D(img, -1, kernel_5)
    else :
        print("not available !!!")

    # Display the original and filtered images
    # cv2.imshow('Image originale', img)
    # cv2.imshow('Image resultat', img_filtered)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return img, img_filtered
    
    
    
def pyramidal(image_path):
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    
    kernel = np.array([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
    ])
    
    kernel_pyramidal = cv2.filter2D(kernel, -1, kernel)
    
    img_filtered = cv2.filter2D(img, -1, kernel_pyramidal)
    
    # cv2.imshow("Image originale", img)
    # cv2.imshow("Image resultat", img_filtered)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return img, img_filtered
    
    
def conique(image_path):
    
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    kernel = np.array([
        [0, 0, 1/25, 0, 0],
        [0, 2/25, 2/25, 2/25, 0],
        [1/25, 2/25, 5/25, 2/25, 1/25],
        [0, 2/25, 2/25, 2/25, 0],
        [0, 0, 1/25, 0, 0]
    ])
    
    img_filtered = cv2.filter2D(img, -1, kernel)
    
    # cv2.imshow("Image originale", img)
    # cv2.imshow("Image resultat", img_filtered)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    
    return img, img_filtered



def median(image_path, kernel_size):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    height, width = image.shape
    output = np.zeros((height, width), dtype=np.uint8)

    padded_image = np.pad(image, ((kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2)), mode='constant')

    for i in range(height):
        for j in range(width):
            
            neighborhood = padded_image[i:i+kernel_size, j:j+kernel_size]

            
            flattened_neighborhood = neighborhood.flatten()

            
            sorted_neighborhood = np.sort(flattened_neighborhood)

            
            median_index = len(sorted_neighborhood) // 2

            
            median_value = sorted_neighborhood[median_index]

            
            output[i, j] = median_value

    
    # cv2.imshow('Image originale', image)
    # cv2.imshow('Image resultat', output)

    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image, output
