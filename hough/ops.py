import cv2
import numpy as np


    
    
# def hough_lines_detector_v2(image_path, rho_resolution, theta_resolution, threshold,
#                          min_line_length=50, max_line_gap=10):
#     # Read the image as grayscale
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
#     # Apply Canny edge detection
#     edges = cv2.Canny(image, 50, 150, apertureSize=3)
    
#     # Initialize accumulator to store the votes for each line parameter combination
#     height, width = edges.shape
#     diagonal_length = np.sqrt(height ** 2 + width ** 2)
#     rho_max = int(diagonal_length)  # Maximum possible rho value
#     accumulator = np.zeros((2 * rho_max, int(np.pi / theta_resolution)), dtype=np.uint64)
    
#     # Compute the sine and cosine of theta values
#     theta_values = np.arange(0, np.pi, theta_resolution)
#     cos_theta = np.cos(theta_values)
#     sin_theta = np.sin(theta_values)
    
#     # Find edge pixels and their coordinates
#     edge_points = np.column_stack(np.nonzero(edges))
    
#     # Vote in the accumulator for each edge point
#     for rho_index in range(accumulator.shape[0]):
#         rho = rho_index - rho_max
#         for theta_index, (cos_val, sin_val) in enumerate(zip(cos_theta, sin_theta)):
#             # Compute rho value for the given (rho, theta) pair
#             rho_val = int(round(rho * cos_val + rho * sin_val))
            
#             # Vote only if the rho value is positive
#             if rho_val >= 0:
#                 accumulator[rho_val, theta_index] += 1
    
#     # Find lines based on the accumulator
#     lines = []
#     for rho_index in range(accumulator.shape[0]):
#         for theta_index in range(accumulator.shape[1]):
#             if accumulator[rho_index, theta_index] > threshold:
#                 rho = rho_index - rho_max
#                 theta = theta_index * theta_resolution
#                 lines.append((rho, theta))
    
#     # Filter lines based on minimum line length and maximum line gap
#     filtered_lines = cv2.HoughLinesP(edges, rho_resolution, theta_resolution, threshold,
#                                      minLineLength=min_line_length, maxLineGap=max_line_gap)
    
#     # Draw detected lines on the original image
#     result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#     if filtered_lines is not None:
#         for line in filtered_lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(result_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    
#     # Display the original image and the result
#     cv2.imshow('Original Image', image)
#     cv2.imshow('Hough Lines Detection Result', result_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
    


def hough(image, rho_resolution, theta_resolution, threshold):
    # Apply Canny edge detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    
    # Initialize accumulator to store the votes for each line parameter combination
    height, width = edges.shape
    diagonal_length = np.sqrt(height ** 2 + width ** 2)
    rho_max = int(diagonal_length)  # Maximum possible rho value
    accumulator = np.zeros((2 * rho_max, int(np.pi / theta_resolution)), dtype=np.uint64)
    
    # Compute the sine and cosine of theta values
    theta_values = np.arange(0, np.pi, theta_resolution)
    cos_theta = np.cos(theta_values)
    sin_theta = np.sin(theta_values)
    
    # Find edge pixels and their coordinates
    edge_points = np.column_stack(np.nonzero(edges))
    
    # Vote in the accumulator for each edge point
    for rho_index in range(accumulator.shape[0]):
        rho = rho_index - rho_max
        for theta_index, (cos_val, sin_val) in enumerate(zip(cos_theta, sin_theta)):
            # Compute rho value for the given (rho, theta) pair
            rho_val = int(round(rho * cos_val + rho * sin_val))
            
            # Vote only if the rho value is positive
            if rho_val >= 0:
                accumulator[rho_val, theta_index] += 1
    
    # Find lines based on the accumulator
    lines = []
    for rho_index in range(accumulator.shape[0]):
        for theta_index in range(accumulator.shape[1]):
            if accumulator[rho_index, theta_index] > threshold:
                rho = rho_index - rho_max
                theta = theta_index * theta_resolution
                lines.append((rho, theta))
    
    return lines

def hough_lines_detector(image_path, rho_resolution, theta_resolution, threshold,
                         min_line_length=50, max_line_gap=10):
    # Read the image as grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Detect lines using Hough transform
    lines = hough(image, rho_resolution, theta_resolution, threshold)
    
    # Filter lines based on minimum line length and maximum line gap
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    filtered_lines = cv2.HoughLinesP(edges, rho_resolution, theta_resolution, threshold,
                                     minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    # Draw detected lines on the original image
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if filtered_lines is not None:
        for line in filtered_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    
    # Display the original image and the result
    cv2.imshow('Image originale', image)
    cv2.imshow('Image Resultat', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# def hough(image, rho_resolution, theta_resolution, threshold):
#     # Apply Canny edge detection
#     edges = cv2.Canny(image, 50, 150, apertureSize=3)
    
#     # Initialize accumulator to store the votes for each line parameter combination
#     height, width = edges.shape
#     diagonal_length = np.sqrt(height ** 2 + width ** 2)
#     rho_max = int(diagonal_length)  # Maximum possible rho value
#     accumulator = np.zeros((2 * rho_max, int(np.pi / theta_resolution)), dtype=np.uint64)
    
#     # Compute the sine and cosine of theta values
#     theta_values = np.arange(0, np.pi, theta_resolution)
#     cos_theta = np.cos(theta_values)
#     sin_theta = np.sin(theta_values)
    
#     # Find edge pixels and their coordinates
#     edge_points = np.column_stack(np.nonzero(edges))
    
#     # Vote in the accumulator for each edge point
#     for rho_index in range(accumulator.shape[0]):
#         rho = rho_index - rho_max
#         for theta_index, (cos_val, sin_val) in enumerate(zip(cos_theta, sin_theta)):
#             # Compute rho value for the given (rho, theta) pair
#             rho_val = int(round(rho * cos_val + rho * sin_val))
            
#             # Vote only if the rho value is positive
#             if rho_val >= 0:
#                 accumulator[rho_val, theta_index] += 1
    
#     # Find lines based on the accumulator
#     lines = []
#     for rho_index in range(accumulator.shape[0]):
#         for theta_index in range(accumulator.shape[1]):
#             if accumulator[rho_index, theta_index] > threshold:
#                 rho = rho_index - rho_max
#                 theta = theta_index * theta_resolution
#                 lines.append((rho, theta))
    
#     return lines

# def draw_lines(image, lines):
#     result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#     height, width = image.shape
#     for rho, theta in lines:
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         x1 = int(x0 + 1000 * (-b))
#         y1 = int(y0 + 1000 * (a))
#         x2 = int(x0 - 1000 * (-b))
#         y2 = int(y0 - 1000 * (a))
#         print(x1, y1, x2, y2)
#         if 0 <= x1 < width and 0 <= x2 < width and 0 <= y1 < height and 0 <= y2 < height:
#             cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     return result_image

# def hough_lines_detector(image_path, rho_resolution, theta_resolution, threshold,
#                          min_line_length=50, max_line_gap=10):
#     # Read the image as grayscale
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
#     # Detect lines using Hough transform
#     lines = hough(image, rho_resolution, theta_resolution, threshold)
#     print("Detected Lines:", lines)
    
#     # Draw detected lines on the original image
#     result_image = draw_lines(image, lines)
    
#     # Display the original image and the result
#     cv2.imshow('Original Image', image)
#     cv2.imshow('Hough Lines Detection Result', result_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()