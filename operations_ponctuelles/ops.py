import cv2
import numpy as np
import matplotlib.pylab as plt



def ajuster_luminosite(image_path, brightness_factor):
    
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    
    if gray_image is None:
        print("Error: Unable to load image")
        return
    
    cv2.imshow("Image originale", gray_image)
    bgr_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 2] = cv2.add(hsv_image[:, :, 2], brightness_factor)
    adjusted_bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    cv2.imshow("Image resultat ", adjusted_bgr_image)

    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def moyenne_et_contraste(path_to_image):
    img = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)


    if img is not None:
    
        mean_value = np.mean(img)
        std_dev = np.std(img)
        print(f"Mean: {mean_value}")
        print(f"Standard Deviation: {std_dev}")
        print(img.shape)
        
    else:
        print("Image not loaded. Please check the file path.")
        
        
def afficher_histogramme(image_path):
   
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    
    if gray_image is None:
        print("Error: Unable to load image")
        return

    
    hist = cv2.calcHist([gray_image], [0], None, [256], [0,256])

    # Display the grayscale image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')

    # Display the histogram
    plt.subplot(1, 2, 2)
    plt.plot(hist, color='black')
    plt.title('Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    plt.show()
    


def ameliorer_contraste_linear(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD2_GRAYSCALE)

    
    if image is None:
        print("Error: Unable to load image")
        return

    
    min_val = np.min(image)
    max_val = np.max(image)

    
    enhanced_image = ((image - min_val) / (max_val - min_val)) * 255

    
    enhanced_image = np.uint8(enhanced_image)

    
    cv2.imshow('Image originale', image)
    cv2.imshow('Image amelioree', enhanced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def ameliorer_contraste_avec_saturation(image_path, alpha=1.0, beta=0):
    grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image has been successfully loaded
    if grayscale_image is None:
        print("Error: Unable to load image")
        return

    
    stretched_image = cv2.normalize(grayscale_image, None, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX)

    
    cv2.imshow('Image Originale', grayscale_image)
    cv2.imshow('Image resultat', stretched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def histogram_equalization(image_path):
    
    grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    
    if grayscale_image is None:
        print("Error: Unable to load image")
        return

    
    hist, bins = np.histogram(grayscale_image.flatten(), 256, [0, 256])

    
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    
    equalized_image = cdf[grayscale_image]

    
    cv2.imshow("Image originale", grayscale_image)
    cv2.imshow("Image resultat", equalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def decalage_additif(image_path, L):
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    img_float = img.astype(np.float32)

    img_adjusted = np.uint8(img_float + L)

    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Image originale')
    plt.subplot(2, 2, 2), plt.hist(img.ravel(), 256, [0, 256]), plt.title('Histogramme originale')

    plt.subplot(2, 2, 3), plt.imshow(img_adjusted, cmap='gray'), plt.title('Image resultat')
    plt.subplot(2, 2, 4), plt.hist(img_adjusted.ravel(), 256, [0, 256]), plt.title('Histogramme resultat')

    plt.show()

def decalage_multiplicatif(image_path, L):
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    img_float = img.astype(np.float32)

    img_adjusted = np.uint8(img_float * L)

    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Image originale ')
    plt.subplot(2, 2, 2), plt.hist(img.ravel(), 256, [0, 256]), plt.title('Histogramme originale')

    plt.subplot(2, 2, 3), plt.imshow(img_adjusted, cmap='gray'), plt.title('Image resultat')
    plt.subplot(2, 2, 4), plt.hist(img_adjusted.ravel(), 256, [0, 256]), plt.title('Histogramme resultat')

    plt.show()

def inversion(image_path):
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    
    img_float = img.astype(np.float32)

    
    img_adjusted = np.uint8(-1 * img_float + 255)

   
    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Image originale')
    plt.subplot(2, 2, 2), plt.hist(img.ravel(), 256, [0, 256]), plt.title('Histogramme originale ')

    plt.subplot(2, 2, 3), plt.imshow(img_adjusted, cmap='gray'), plt.title('Image resultat')
    plt.subplot(2, 2, 4), plt.hist(img_adjusted.ravel(), 256, [0, 256]), plt.title('Histogramme resultat')

    plt.show()
    
def ameliorer_contraste_decalage(image_path):
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    A = float(np.min(img))
    B = float(np.max(img))
    
    P = 255 / (B - A)
    L = -P * A

    
    img_float = img.astype(np.float32)

    
    img_adjusted = np.uint8(P * (img_float) + L)

    
    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Image originale ')
    plt.subplot(2, 2, 2), plt.hist(img.ravel(), 256, [0, 256]), plt.title('Histogramme originale ')

    plt.subplot(2, 2, 3), plt.imshow(img_adjusted, cmap='gray'), plt.title('Image resultat')
    plt.subplot(2, 2, 4), plt.hist(img_adjusted.ravel(), 256, [0, 256]), plt.title('Histogramme resultat')

    plt.show()
    
    
def seuillage(image_path, threshold_value):

    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


    img_binary = img_gray.copy()
    for i in range(img_binary.shape[0]):
        for j in range(img_binary.shape[1]):
            if img_gray[i, j] > threshold_value:
                img_binary[i, j] = 255  # White (binary)
            else:
                img_binary[i, j] = 0    # Black (binary)

    
    cv2.imshow("Image originale", img_gray)
    cv2.imshow("Image resultat", img_binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()