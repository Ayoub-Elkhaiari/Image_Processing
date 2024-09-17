import matplotlib.pyplot as plt
import cv2
import numpy as np



def erosion_binary(image_path):
    # Read the image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Threshold the image to convert it to binary
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Define the structuring element (kernel) for erosion
    kernel = np.ones((5,5), np.uint8)
    
    # Apply erosion using OpenCV's morphologyEx function
    eroded_img = cv2.morphologyEx(binary_img, cv2.MORPH_ERODE, kernel)
    
    # Display the original, binary, and eroded images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray'), plt.title('Original Image')
    plt.subplot(1, 3, 2), plt.imshow(binary_img, cmap='gray'), plt.title('Binary Image')
    plt.subplot(1, 3, 3), plt.imshow(eroded_img, cmap='gray'), plt.title('Eroded Image')
    plt.show()
    
    
    
def erosion_grayscale(image_path):
    # Read the image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Define the structuring element (kernel) for erosion
    kernel = np.ones((5,5), np.uint8)
    
    # Apply erosion using OpenCV's morphologyEx function
    eroded_img = cv2.erode(img, kernel, iterations=1)
    
    # Display the original and eroded images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Original Image')
    plt.subplot(1, 2, 2), plt.imshow(eroded_img, cmap='gray'), plt.title('Eroded Image')
    plt.show()
    
    
    
def dilation_binary(image_path):
    # Read the image as binary (0 for black, 255 for white)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ret, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Define the structuring element (kernel) for dilation
    kernel = np.ones((5,5), np.uint8)
    
    # Apply dilation using OpenCV's morphologyEx function
    dilated_img = cv2.dilate(binary_img, kernel, iterations=1)
    
    # Display the original and dilated binary images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.imshow(binary_img, cmap='gray'), plt.title('Original Binary Image')
    plt.subplot(1, 2, 2), plt.imshow(dilated_img, cmap='gray'), plt.title('Dilated Binary Image')
    plt.show()
    
    
    
    
def dilation_grayscale(image_path):
    # Read the image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Define the structuring element (kernel) for dilation
    kernel = np.ones((5,5), np.uint8)
    
    # Apply dilation using OpenCV's morphologyEx function
    dilated_img = cv2.dilate(img, kernel, iterations=1)
    
    # Display the original and dilated grayscale images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Original Grayscale Image')
    plt.subplot(1, 2, 2), plt.imshow(dilated_img, cmap='gray'), plt.title('Dilated Grayscale Image')
    plt.show()