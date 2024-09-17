import numpy as np
import cv2

def harris_detector_from_scratch(image_path, k=0.04, window_size=3, threshold=100000):
    # Read the image as grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Compute gradients using Sobel operator
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute elements of the structure matrix M
    Ixx = Ix ** 2
    Ixy = Ix * Iy
    Iyy = Iy ** 2
    
    # Compute sums of the structure matrix elements over a local region
    Sxx = cv2.GaussianBlur(Ixx, (window_size, window_size), 0)
    Sxy = cv2.GaussianBlur(Ixy, (window_size, window_size), 0)
    Syy = cv2.GaussianBlur(Iyy, (window_size, window_size), 0)
    
    # Compute the determinant and trace of M for each pixel
    det_M = (Sxx * Syy) - (Sxy ** 2)
    trace_M = Sxx + Syy
    
    # Compute the corner response function R
    R = det_M - k * (trace_M ** 2)
    
    # Threshold R to identify corner points
    corner_mask = R > threshold
    
    # Get coordinates of corner points
    corner_points = np.argwhere(corner_mask)
    
    # Draw circles around corner points
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for point in corner_points:
        x, y = point[::-1]
        cv2.line(result_image, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), 1)
        cv2.line(result_image, (x - 5, y + 5), (x + 5, y - 5), (0, 0, 255), 1)
    
    # Display the result
    cv2.imshow('Image originale', image)
    cv2.imshow('Image resultat', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def harris_detector(image_path, threshold=0.01):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Detect corners using Harris corner detection
    dst = cv2.cornerHarris(img, blockSize=2, ksize=3, k=0.04)
    
    # Thresholding
    dst_thresh = np.zeros_like(dst)
    dst_thresh[dst > threshold * dst.max()] = 255
    
    # Convert thresholded image to uint8 for visualization
    dst_thresh = np.uint8(dst_thresh)
    
    # Mark detected corners on the original image
    img_with_corners = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_with_corners[dst_thresh > 0] = [0, 0, 255]  # Red color for corners
    
    # Display the original image with detected corners
    cv2.imshow('Image originale', img)
    cv2.imshow('Image resultat', img_with_corners)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
def susan_detector(image_path, threshold=27, distance=3):
    # Read the image as grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Define SUSAN mask
    mask = np.array([[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 0, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]], dtype=np.uint8)
    
    # Compute number of similar pixels for each pixel using mask
    similar_pixels = cv2.filter2D(image, -1, mask, borderType=cv2.BORDER_CONSTANT)
    
    # Compute dissimilarity measure for each pixel
    dissimilarity = mask.size - similar_pixels
    
    # Threshold dissimilarity to identify corner points
    corner_mask = dissimilarity >= threshold
    
    # Get coordinates of corner points
    corner_points = np.argwhere(corner_mask)
    
    # Draw circles around corner points
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for point in corner_points:
        x, y = point[::-1]
        cv2.line(result_image, (x - distance, y), (x + distance, y), (0, 0, 255), 1)
        cv2.line(result_image, (x, y - distance), (x, y + distance), (0, 0, 255), 1)
    
    # Display the result
    cv2.imshow('SUSAN Corner Detection Result', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()