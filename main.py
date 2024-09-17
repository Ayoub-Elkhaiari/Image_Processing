from operations_ponctuelles.ops import ajuster_luminosite, moyenne_et_contraste, afficher_histogramme, ameliorer_contraste_linear, ameliorer_contraste_avec_saturation, ameliorer_contraste_decalage, histogram_equalization, decalage_additif, decalage_multiplicatif, inversion, seuillage

import cv2
import numpy as np

from operations_locales.filtres_pb.ops import moyenneur, gaussian, pyramidal, conique, median
from operations_locales.filtres_ph.ops import ph_par_diff, moyenne_ph, gradient_sobel, gradient_prewitt, robert, laplacian, kirsch, kirsch_v2, marr_hildreth



from fourier.ops import fourier_transform, ideal_low_pass_filter_fourier, butterworth
from interest_pts.ops import harris_detector_from_scratch, harris_detector, susan_detector
from hough.ops import hough_lines_detector




image_path = r"\images\lena.bmp"
image_path1 = r"\images\lena.bmp"
image_path2 = r"\images\baboon.tiff"
image_path_noise = r"\images\lena_bruitfort.raw"


intrest_detection = r"C:\Users\hp\Downloads\hough.jpeg"

FT_1 = r"C:\Users\hp\Desktop\TP_image_processing\assets\carre_Iss.png"



        
        
'''
    do your customized main function
    or maybe integrate some UI and make all the 
    functions in modules returning the images
    not just the displaying 

'''