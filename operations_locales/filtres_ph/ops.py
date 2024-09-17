import numpy as np
import matplotlib.pylab as plt
import cv2 



def ph_par_diff(image_path, kernel_size):
    
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
        to_substract = cv2.filter2D(img, -1, kernel_3)
        final = img.astype(np.float32) - to_substract.astype(np.float32)
        final = np.abs(final).astype(np.uint8)
    elif kernel_size == 5:
        to_substract = cv2.filter2D(img, -1, kernel_5)
        final = img.astype(np.float32) - to_substract.astype(np.float32)
        final = np.abs(final).astype(np.uint8)
    else:
        print("Not Available!!!")
        
    
    cv2.imshow("Image originale", img)
    cv2.imshow("Image resultat", final)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # return img, final 
    
    
def moyenne_ph(image_path, kernel_size):
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    kernel_5 = np.array([
                        [-1/25, -1/25, -1/25, -1/25, -1/25],
                        [-1/25, -1/25, -1/25, -1/25, -1/25],
                        [-1/25, -1/25, 24/25, -1/25, -1/25],
                        [-1/25, -1/25, -1/25, -1/25, -1/25],
                        [-1/25, -1/25, -1/25, -1/25, -1/25]
                                        ])
    kernel_3 = np.array([
        [-1/9, -1/9, -1/9],
        [-1/9, 8/9, -1/9],
        [-1/9, -1/9, -1/9]
    ])
    
    if kernel_size == 3:
        final = cv2.filter2D(img, -1, kernel_3)
        
    elif kernel_size == 5:
        final = cv2.filter2D(img, -1, kernel_5)
        
    else:
        print("Not Available!!!")
        
        
    cv2.imshow("Image originale", img)
    cv2.imshow("Image resultat", final)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # return img, final 
       
def gradient_sobel(image_path):
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    kernel_X = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    kernel_Y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    img_x = cv2.filter2D(img, -1, kernel_X)
    final = cv2.filter2D(img_x, -1, kernel_Y)
    
    cv2.imshow("Image originale", img)
    cv2.imshow("Image resultat", final)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # return img, final 


def gradient_prewitt(image_path):
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    kernel_X = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])
    kernel_Y = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ])
    img_x = cv2.filter2D(img, -1, kernel_X)
    final = cv2.filter2D(img_x, -1, kernel_Y)
    
    cv2.imshow("Image originale", img)
    cv2.imshow("Image resultat", final)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # return img, final 


def robert(image_path):
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    kernel_X = np.array([
        [ 0, -1],
        [1, 0]
        
    ])
    kernel_Y = np.array([
        [-1, 0],
        [0, 1]
        
    ])
    img_x = cv2.filter2D(img, -1, kernel_X)
    final = cv2.filter2D(img_x, -1, kernel_Y)
    
    cv2.imshow("Image originale", img)
    cv2.imshow("Image resultat", final)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # return img, final 


def laplacian(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    kernel = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])
    
    final = cv2.filter2D(img, -1, kernel)
    
    cv2.imshow("Image originale", img)
    cv2.imshow("Image resultat", final)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # return img, final


def kirsch(image_path):
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    kernel = np.array([[ -3,  -3, -3],
                        [ 5,  0, -3],
                        [ 5, 5, -3]])
    
    final = cv2.filter2D(img, -1, kernel)
    
    cv2.imshow("Image originale", img)
    cv2.imshow("Image resultat", final)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # return img, final

def kirsch_v2(image_path):
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    kernel_north = np.array([[-3, -3, -3],
                         [ 0,  0,  0],
                         [ 3,  3,  3]])

    kernel_northeast = np.array([[-3, -3,  0],
                             [-3,  0,  3],
                             [ 0,  3,  3]])

    kernel_east = np.array([[-3,  0,  3],
                        [-3,  0,  3],
                        [-3,  0,  3]])

    kernel_southeast = np.array([[ 0,  3,  3],
                             [-3,  0,  3],
                             [-3, -3,  0]])

    kernel_south = np.array([[ 3,  3,  3],
                         [ 0,  0,  0],
                         [-3, -3, -3]])

    kernel_southwest = np.array([[ 0,  3,  3],
                             [ 3,  0, -3],
                             [ 3, -3, -3]])

    kernel_west = np.array([[ 3,  0, -3],
                        [ 3,  0, -3],
                        [ 3,  0, -3]])

    kernel_northwest = np.array([[ 3,  3,  0],
                              [ 3,  0, -3],
                              [ 0, -3, -3]])
    
    final = cv2.filter2D(img, -1, kernel_north)
    final = cv2.filter2D(final, -1, kernel_northeast)
    final = cv2.filter2D(final, -1, kernel_east)
    final = cv2.filter2D(final, -1, kernel_southeast)
    final = cv2.filter2D(final, -1, kernel_south)
    final = cv2.filter2D(final, -1, kernel_southwest)
    final = cv2.filter2D(final, -1, kernel_west)
    final = cv2.filter2D(final, -1, kernel_northwest)
    cv2.imshow("Image originale", img)
    cv2.imshow("Image resultat", final)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # return img, final
    
    
    
    
def marr_hildreth(image_path):
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    kernel = np.array([[-1, -3, -4, -3, -1],
                        [-3, 0, 6, 0, -3],
                        [-4, 6, 20, 6, -4],
                        [-3, 0, 6, 0, -3],
                        [-1, -3, -4, -3, -1]])
    
    final = cv2.filter2D(img, -1, kernel)
    
    cv2.imshow("Image originale", img)
    cv2.imshow("Image resultat", final)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # return img, final



# def canny(image_path, sigma=1.4, low_threshold=20, high_threshold=50):
#     # Read the image in grayscale
#     image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
#     # Step 1: Apply Gaussian blur
#     gaussian_kernel = cv2.getGaussianKernel(5, sigma)
#     image_smoothed = cv2.filter2D(image_gray, -1, gaussian_kernel)
    
#     # Step 2: Calculate gradients (Sobel operator)
#     sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#     sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
#     gradient_x = cv2.filter2D(image_smoothed, -1, sobel_x)
#     gradient_y = cv2.filter2D(image_smoothed, -1, sobel_y)
    
#     gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
#     gradient_direction = np.arctan2(gradient_y, gradient_x)
    
#     # Step 3: Non-maximum suppression
#     gradient_direction = np.rad2deg(gradient_direction)
#     gradient_direction[gradient_direction < 0] += 180
    
#     suppressed_gradient = np.zeros_like(gradient_magnitude)
#     for i in range(1, gradient_magnitude.shape[0] - 1):
#         for j in range(1, gradient_magnitude.shape[1] - 1):
#             orientation = gradient_direction[i, j]
#             mag = gradient_magnitude[i, j]
            
#             if (0 <= orientation < 22.5) or (157.5 <= orientation <= 180) or (22.5 <= orientation < 67.5):
#                 if (mag >= gradient_magnitude[i, j-1]) and (mag >= gradient_magnitude[i, j+1]):
#                     suppressed_gradient[i, j] = mag
#             elif (67.5 <= orientation < 112.5) or (112.5 <= orientation < 157.5):
#                 if (mag >= gradient_magnitude[i-1, j]) and (mag >= gradient_magnitude[i+1, j]):
#                     suppressed_gradient[i, j] = mag
#             elif (112.5 <= orientation < 157.5):
#                 if (mag >= gradient_magnitude[i-1, j-1]) and (mag >= gradient_magnitude[i+1, j+1]):
#                     suppressed_gradient[i, j] = mag
#             elif (22.5 <= orientation < 67.5):
#                 if (mag >= gradient_magnitude[i-1, j+1]) and (mag >= gradient_magnitude[i+1, j-1]):
#                     suppressed_gradient[i, j] = mag
    
#     # Step 4: Apply hysteresis thresholding
#     low_threshold = low_threshold
#     high_threshold = high_threshold
    
#     strong_edges = (suppressed_gradient >= high_threshold)
#     weak_edges = (suppressed_gradient >= low_threshold) & (suppressed_gradient < high_threshold)
    
#     # Step 5: Edge tracking by hysteresis
#     for i in range(1, suppressed_gradient.shape[0] - 1):
#         for j in range(1, suppressed_gradient.shape[1] - 1):
#             if weak_edges[i, j]:
#                 if strong_edges[i-1:i+2, j-1:j+2].any():
#                     suppressed_gradient[i, j] = high_threshold
#                 else:
#                     suppressed_gradient[i, j] = 0
    
#     # Display the original and resulting images
#     original_image = cv2.imread(image_path)
#     cv2.imshow('Original Image', original_image)
#     cv2.imshow('Canny Edge Detection', suppressed_gradient.astype(np.uint8))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()