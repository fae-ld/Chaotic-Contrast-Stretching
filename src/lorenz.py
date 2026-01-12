import numpy as np
import cv2

def global_stretching(img, dt=0.1, x=250, y=250, beta=8/3):
    """
    Performs Global Chaotic Contrast Stretching using Lorenz dynamics.
    
    This function treats the entire image as a dynamical system where the 
    coupling parameters (x, y) are set to high values. This creates a 
    non-linear high-gain effect, which is ideal for extracting stable 
    body masks by forcing most anatomical tissues toward saturation.
    
    Args:
        img (numpy.ndarray): Input grayscale image (0-255).
        dt (float): Integration time step. Higher values increase intensity jump.
        x (float): Lorenz coupling parameter X (Gain control).
        y (float): Lorenz coupling parameter Y (Gain control).
        beta (float): Dissipation parameter representing energy loss in the system.
        
    Returns:
        numpy.ndarray: Enhanced 8-bit image optimized for contour extraction.
    """
    # 1. Map pixel intensity to Lorenz Z-space (standard range approx. 0-50)
    z = (img.astype(float) / 255.0) * 50
    
    # 2. Calculate the rate of change based on Lorenz z-component: dz/dt = xy - beta*z
    # High x*y values create a strong 'attractor' towards maximum intensity.
    dz = (x * y - beta * z) * dt
    z_new = z + dz
    
    # 3. Normalize to ensure the output stays within valid 8-bit pixel range
    z_max = np.max(z_new) if np.max(z_new) > 0 else 1
    return np.clip((z_new / z_max) * 255, 0, 255).astype(np.uint8)

def adaptive_stretching(img, window_size=25, dt=0.1, beta=8/3):
    """
    Performs Spatially-Adaptive Chaotic Contrast Stretching.
    
    Unlike the global version, this function derives the coupling parameters 
    (x, y) from local neighborhood statistics. This allows the system to 
    dynamically adjust the stretching force, highlighting internal pulmonary 
    features and vascularity while preserving local contrast.
    
    Args:
        img (numpy.ndarray): Input grayscale image (0-255).
        window_size (int): Size of the local kernel (NxN) to calculate mean drivers.
        dt (float): Integration time step.
        beta (float): Dissipation parameter.
        
    Returns:
        numpy.ndarray: Feature-enhanced 8-bit image with high internal detail.
    """
    # 1. Map to Lorenz Z-space
    z = (img.astype(float) / 255.0) * 50
    
    # 2. Derive adaptive drivers (x, y) from local mean intensity
    # This creates a localized 'force field' for contrast adjustment.
    local_mean = cv2.blur(z, (window_size, window_size))
    x_adaptive = np.sqrt(local_mean * 2)
    y_adaptive = x_adaptive
    
    # 3. Apply dynamical transformation
    dz = (x_adaptive * y_adaptive - beta * z) * dt
    z_new = z + dz
    
    # 4. Normalize and convert back to 8-bit
    z_max = np.max(z_new) if np.max(z_new) > 0 else 1
    return np.clip((z_new / z_max) * 255, 0, 255).astype(np.uint8)