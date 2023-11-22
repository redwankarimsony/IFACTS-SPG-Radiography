import cv2
import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

def linear_contrast_stretching(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

def histogram_equalization(image):
    return cv2.equalizeHist(image)

def clahe_enhancement(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def gamma_correction(image, gamma=1.5):
    return np.array(255 * (image / 255) ** gamma, dtype='uint8')

def laplacian_filter(image):
    return cv2.Laplacian(image, cv2.CV_64F)

def compare_histogram_equalization(path_to_image):
    
    # Load the image in grayscale
    image = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)

    # Perform histogram equalization
    equalized_image = cv2.equalizeHist(image)

    # Calculate histograms
    original_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    equalized_hist = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

    # Plot the histograms and images
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(2, 2, 2)
    plt.imshow(equalized_image, cmap='gray')
    plt.title('Equalized Image')

    plt.subplot(2, 2, 3)
    plt.plot(original_hist)
    plt.title('Original Histogram')

    plt.subplot(2, 2, 4)
    plt.plot(equalized_hist)
    plt.title('Equalized Histogram')

    plt.tight_layout()
    plt.savefig(path_to_image.replace(".png", "_histogram.png"), dpi=300)



if __name__ == "__main__":

    # Load the image in grayscale
    # image = cv2.imread('/localscratch2/sonymd/MSU-SPG-Radiography-Dataset/Photographs/084_AM_Chest_AP2.png', cv2.IMREAD_GRAYSCALE)
    image = cv2.imread('iris.png', cv2.IMREAD_GRAYSCALE)

    # print(image.shape)
    # # Remove noise by gaussian blur
    # image = cv2.GaussianBlur(src=image,
    #                          ksize=(3, 3),
    #                          sigmaX=0)


    # Apply the functions
    img_linear = linear_contrast_stretching(image)
    img_hist_eq = histogram_equalization(image)
    img_clahe = clahe_enhancement(image)
    img_gamma = gamma_correction(image)
    img_laplacian = laplacian_filter(image)

    # Plotting
    plt.figure(figsize=(20, 10), dpi=300)
    images = [image, img_linear, img_hist_eq, img_clahe, img_gamma]#, img_laplacian]
    titles = ['Original', 'Linear Stretching', 'Histogram Eq.', 'CLAHE', 'Gamma Correction']#, 'Laplacian Filter']

    for i in range(len(images)):
        plt.subplot(2, math.ceil(len(images)/2), i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("iris_out.png", dpi=300)
    # plt.show()

    compare_histogram_equalization("iris.png")


    # cv2.imwrite("iris_out.png", image, params=[cv2.IMWRITE_PNG_COMPRESSION, 0])