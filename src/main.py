import cv2
import matplotlib.pyplot as plt
from lorenz import global_stretching, adaptive_stretching
from utils import apply_morphology

def run_pipeline(image_path):
    """
    Executes the complete Chaotic Contrast Stretching pipeline.
    
    The process involves:
    1. Global Lorenz transform for anatomical masking.
    2. Morphological refinement to ensure mask solidity.
    3. Adaptive Lorenz transform for pulmonary feature enhancement.
    4. Bitwise fusion to isolate enhanced lungs from the background.
    """
    
    # 1. Image Acquisition and Preprocessing
    # Resizing to 224x224 as standardized for JSRT dataset experiments
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return
    
    img = cv2.resize(img, (224, 224))

    # 2. Anatomical Localization (Global Mode)
    # High-gain parameters (x,y=250) are used to create the body silhouette
    body_mask_raw = global_stretching(img, x=100, y=100)
    body_mask = apply_morphology(body_mask_raw, kernel_size=21)

    # 3. Internal Feature Enhancement (Adaptive Mode)
    # Spatially-varying drivers highlight pulmonary parenchyma
    internal_detail = adaptive_stretching(img, window_size=25)

    # 4. Final Fusion
    # Merging the solid body mask with the adaptive detail for final localization
    final_result = cv2.bitwise_and(internal_detail, internal_detail, mask=body_mask)

    # 5. Visualization Pipeline
    titles = ['Original CXR', 'Body Mask (Global)', 'Detail (Adaptive)', 'Final Localization']
    images = [img, body_mask, internal_detail, final_result]

    plt.figure(figsize=(16, 4))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i], fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Path to your sample image
    SAMPLE_PATH = 'data/example.png'
    run_pipeline(SAMPLE_PATH)