import cv2
import numpy as np

def apply_morphology(img, kernel_size=21):
    """
    Performs morphological cleanup to generate a solid anatomical mask.
    
    This function converts the continuous intensity output from the Lorenz 
    transform into a binary mask. It specifically addresses 'pitting' artifacts 
    (small holes inside the body region) by applying a morphological Closing 
    operation, ensuring a contiguous ROI for lung localization.

    Args:
        img (numpy.ndarray): The enhanced image from Global CCS (0-255).
        kernel_size (int): The diameter of the structuring element. Larger 
            values will close larger gaps/holes within the tissue area.

    Returns:
        numpy.ndarray: A binary mask (0 or 255) representing the localized 
            body structure.
    """
    # 1. Define the structuring element (Kernel)
    # A square kernel is used to expand and then shrink the white regions.
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 2. Automated Binarization
    # Otsu's method is used to determine the optimal threshold between 
    # the chaotic 'attractor' state and the background.
    _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Morphological Closing
    # Closing is dilation followed by erosion. It fills small dark holes 
    # (e.g., vascular shadows or noise) while preserving the overall shape.
    return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)