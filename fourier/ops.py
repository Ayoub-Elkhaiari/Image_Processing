import numpy as np 
import matplotlib.pylab as plt
import cv2


def fourier_transform(image_path):
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

   
    f_transform = np.fft.fft2(img)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20*np.log(np.abs(f_transform_shifted))

    
    # plt.subplot(121), plt.imshow(img, cmap='gray')
    # plt.title('Image originale'), plt.xticks([]), plt.yticks([])

    
    # plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    # plt.title('spectre'), plt.xticks([]), plt.yticks([])

    # plt.show()
    
    return img, magnitude_spectrum


def ideal_low_pass_filter_fourier(image_path, cutoff_frequency):
    # Read the image as grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Compute the Fourier transform of the image
    f_transform = np.fft.fft2(image)
    
    # Shift the zero frequency component to the center
    f_transform_shifted = np.fft.fftshift(f_transform)
    
    # Get the dimensions of the image
    rows, cols = image.shape
    
    # Create a grid of frequencies
    crow, ccol = rows // 2, cols // 2
    x = np.arange(-crow, crow)
    y = np.arange(-ccol, ccol)
    x, y = np.meshgrid(x, y)
    distance = np.sqrt(x**2 + y**2)
    
    # Create the ideal low-pass filter mask
    mask = np.zeros((rows, cols), np.uint8)
    mask[distance <= cutoff_frequency] = 1
    
    # Apply the filter to the Fourier transformed image
    filtered_f_transform = f_transform_shifted * mask
    
    # Shift the spectrum back
    filtered_f_transform_shifted = np.fft.ifftshift(filtered_f_transform)
    
    # Apply the inverse Fourier transform
    filtered_image = np.fft.ifft2(filtered_f_transform_shifted)
    filtered_image = np.abs(filtered_image)  # Take the magnitude
    
    # Convert the filtered image to uint8 for display
    filtered_image = np.uint8(filtered_image)
    
    # Display the original and filtered images
    cv2.imshow('Image originale', image)
    cv2.imshow('Image resultat', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def butterworth(image_path, cutoff_frequency, order):
    # Read the image as grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Compute the Fourier transform of the image
    f_transform = np.fft.fft2(image)
    
    # Shift the zero frequency component to the center
    f_transform_shifted = np.fft.fftshift(f_transform)
    
    # Get the dimensions of the image
    rows, cols = image.shape
    
    # Create a grid of frequencies
    crow, ccol = rows // 2, cols // 2
    x = np.arange(-crow, crow)
    y = np.arange(-ccol, ccol)
    x, y = np.meshgrid(x, y)
    distance = np.sqrt(x**2 + y**2)
    
    # Create the Butterworth low-pass filter mask
    mask = 1 / (1 + (distance / cutoff_frequency) ** (2 * order))
    
    # Apply the filter to the Fourier transformed image
    filtered_f_transform = f_transform_shifted * mask
    
    # Shift the spectrum back
    filtered_f_transform_shifted = np.fft.ifftshift(filtered_f_transform)
    
    # Apply the inverse Fourier transform
    filtered_image = np.fft.ifft2(filtered_f_transform_shifted)
    filtered_image = np.abs(filtered_image)  # Take the magnitude
    
    # Convert the filtered image to uint8 for display
    filtered_image = np.uint8(filtered_image)
    
    # Display the original and filtered images
    cv2.imshow('Original Image', image)
    cv2.imshow('Filtered Image', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

