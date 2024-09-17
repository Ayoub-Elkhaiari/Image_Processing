# Custom Image Processing Library

## Overview
This project is a comprehensive image processing library implemented from scratch using OpenCV and NumPy. It provides a wide range of image processing operations, from basic point operations to advanced techniques like Fourier transforms and interest point detection.

## Project Structure
- `main.py`: The main script where users can implement their custom workflows.
- `operations_ponctuelles/`: Module for point operations.
- `operations_locales/`: Module for local operations, including low-pass and high-pass filters.
- `fourier/`: Module for Fourier transform operations.
- `interest_pts/`: Module for interest point detection.
- `hough/`: Module for Hough transform operations.
- `morphology/`: Module for morphological operations.

## Prerequisites
- Python 3.x
- OpenCV
- NumPy

To install the necessary packages, run:
```bash
pip install opencv-python numpy
```

## Available Operations

### Point Operations (`operations_ponctuelles.ops`)
- `ajuster_luminosite`: Adjust image brightness.
- `moyenne_et_contraste`: Calculate mean and contrast.
- `afficher_histogramme`: Display image histogram.
- `ameliorer_contraste_linear`: Linear contrast enhancement.
- `ameliorer_contraste_avec_saturation`: Contrast enhancement with saturation.
- `ameliorer_contraste_decalage`: Contrast enhancement by shifting.
- `histogram_equalization`: Perform histogram equalization.
- `decalage_additif`: Additive shift operation.
- `decalage_multiplicatif`: Multiplicative shift operation.
- `inversion`: Invert image.
- `seuillage`: Thresholding operation.

### Local Operations
#### Low-Pass Filters (`operations_locales.filtres_pb.ops`)
- `moyenneur`: Mean filter.
- `gaussian`: Gaussian filter.
- `pyramidal`: Pyramidal filter.
- `conique`: Conical filter.
- `median`: Median filter.

#### High-Pass Filters (`operations_locales.filtres_ph.ops`)
- `ph_par_diff`: High-pass filter by difference.
- `moyenne_ph`: Average high-pass filter.
- `gradient_sobel`: Sobel gradient operator.
- `gradient_prewitt`: Prewitt gradient operator.
- `robert`: Robert's cross operator.
- `laplacian`: Laplacian operator.
- `kirsch`: Kirsch operator.
- `kirsch_v2`: Alternative Kirsch operator.
- `marr_hildreth`: Marr-Hildreth edge detector.

### Fourier Transform Operations (`fourier.ops`)
- `fourier_transform`: Perform Fourier transform.
- `ideal_low_pass_filter_fourier`: Ideal low-pass filter in Fourier domain.
- `butterworth`: Butterworth filter.

### Interest Point Detection (`interest_pts.ops`)
- `harris_detector_from_scratch`: Custom implementation of Harris corner detector.
- `harris_detector`: Harris corner detector.
- `susan_detector`: SUSAN detector.

### Hough Transform (`hough.ops`)
- `hough_lines_detector`: Hough transform for line detection.

### Morphological Operations (`morphology.ops`)
- `dilation_binary`: Binary dilation.
- `dilation_grayscale`: Grayscale dilation.
- `erosion_binary`: Binary erosion.
- `erosion_grayscale`: Grayscale erosion.

## Usage
1. Import the desired operations from their respective modules.
2. Load your image using OpenCV or provide the path to the image.
3. Apply the operations as needed in your custom `main.py` script.

Example:
```python
import cv2
from operations_ponctuelles.ops import ajuster_luminosite, histogram_equalization
from operations_locales.filtres_pb.ops import gaussian
from interest_pts.ops import harris_detector

# Load an image
image = cv2.imread('path/to/your/image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply operations
ajuster_luminosite(image, 50)
histogram_equalization(image)
gaussian(image, kernel_size=5, sigma=1.0)
harris_detector(image)
```

## Customization
- Modify the existing operations or add new ones by editing the respective module files.
- Create your own custom workflows by combining different operations in `main.py`.
- Implement a user interface to make the library more interactive and user-friendly.

## Notes
- All operations are implemented from scratch, providing a deep understanding of the underlying algorithms.
- The library uses OpenCV for image I/O operations, but the core processing is done using custom implementations.
- Some operations may be computationally intensive for large images. Consider optimizing or using smaller images for testing.

## Future Improvements
- Add more advanced operations such as image segmentation or object detection.
- Implement multithreading for faster processing of large images.
- Create a graphical user interface (GUI) for easier interaction with the library.
- Add support for batch processing of multiple images.

