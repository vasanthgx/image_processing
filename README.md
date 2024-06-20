

![logo](https://github.com/vasanthgx/image_processing/blob/main/images/logo.gif)


# Project Title


**Introduction to Image Processing with OpenCV**
 <img src="https://github.com/Anmol-Baranwal/Cool-GIFs-For-GitHub/assets/74038190/b3fef2db-e671-4610-bb84-1d65533dc5fb" width="300" align='right'>

<br><br>


## Overview

Image processing is a method to perform operations on an image to enhance it or extract useful information. It is a rapidly growing technology with applications across various domains such as medical imaging, autonomous vehicles, robotics, and computer vision. This project aims to introduce the basics of image processing using the Open Source Computer Vision Library (OpenCV), a powerful and widely-used library for image and video processing.



### Objectives

The main objectives of this project are to:

1. Introduce the fundamental concepts of image processing.
2. Demonstrate the use of OpenCV for performing basic image processing tasks.
3. Provide hands-on experience with common image processing techniques.
4. Develop skills to apply image processing in real-world applications.



### Why OpenCV ?

OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library. It has over 2,500 optimized algorithms, which can be used for various applications such as detecting and recognizing faces, identifying objects, classifying human actions in videos, tracking camera movements, extracting 3D models of objects, and much more. The library is written in C++ and has interfaces for Python, Java, and MATLAB/OCTAVE.

Key features of OpenCV include:

Wide Range of Functions: OpenCV offers a comprehensive set of functions for image processing, computer vision, and machine learning.
Ease of Use: With interfaces in multiple programming languages, OpenCV is user-friendly and suitable for both beginners and professionals.
Performance: OpenCV is highly optimized for real-time applications and can leverage hardware acceleration.
Community and Support: As an open-source project, OpenCV has a large community of developers and extensive documentation and tutorials.

![alt text](https://github.com/vasanthgx/image_processing/blob/main/images/opencv.png)

### Loading an Image
```
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('portrait_lady.png', cv.IMREAD_COLOR)
plt.imshow(img)
img.shape

```
![alt text](https://github.com/vasanthgx/image_processing/blob/main/images/pic1.png)



### Changing the colorspace to HSV
![alt text](https://github.com/vasanthgx/image_processing/blob/main/images/pic2.png)

The HSV (Hue, Saturation, Value) color space is a cylindrical representation of colors, designed to be more intuitive for human perception. Hue represents the color type and is measured in degrees from 0 to 360. Saturation indicates the vibrancy of the color, ranging from 0 (gray) to 100% (full color). Value represents the brightness, ranging from 0 (black) to 100% (full brightness). Unlike the RGB color model, which is based on primary colors, HSV separates color information (hue) from intensity (value), making it particularly useful in image processing tasks like color segmentation and object detection.

```
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv2_imshow(hsv)
print(hsv.shape)
```
### Enhancing the contrast of the image through Histogram Equalization
Histogram equalization is a technique in image processing used to enhance the contrast of an image. It works by redistributing the image's pixel intensity values so that they span the entire range of possible values, making the histogram of the output image approximately flat. This process increases the global contrast of images, especially when the usable data of the image is represented by close contrast values.

It involves the following steps :

- **Calculate Histogram**: Determine the frequency of each intensity level in the image.
- **Compute Cumulative Distribution Function (CDF)**: Calculate the cumulative sum of the histogram values, which represents the cumulative distribution of pixel intensities.
- **Normalize the CDF**: Normalize the CDF to ensure the intensity values span the entire range (e.g., 0 to 255 for an 8-bit image).
- **Map Original Intensities**: Use the normalized CDF as a mapping function to transform the original pixel values to the new values.

This results in an image with improved contrast, where details in darker or brighter regions become more visible. Histogram equalization is particularly effective for images with backgrounds and foregrounds that are both bright or both dark, increasing the dynamic range and making the features more distinct.

using the function OpenCV function *equalizeHist( )* we acheive the above.

```
# Equivalise the histogram
new_value = cv.equalizeHist(hsv[..., 2])

# Update the value channel
hsv[:, :, 2] = new_value

# Convert image back to BGR
new_rgb_image = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

# Display
cv2_imshow(new_rgb_image)
```
![alt text](https://github.com/vasanthgx/image_processing/blob/main/images/pic3.png)



### Extracting the mask of the lady from the image through Otsu thresholding

Otsu's thresholding is a global thresholding technique used in image processing to automatically perform clustering-based image thresholding or the reduction of a gray-level image to a binary image. It works by finding the threshold that minimizes the intra-class variance (or equivalently, maximizes the inter-class variance) of the black and white pixels.

It involves the following process

- **Read the Image**: Load the image using OpenCV.
- **Convert to Grayscale**: Convert the image to grayscale, as Otsu's method works on single-channel images.
- **Apply Gaussian Blur**: Optionally apply a Gaussian blur to the image to reduce noise and improve thresholding.
- **Otsu's Thresholding**: Apply Otsu's thresholding to binarize the image.
- **Extract the Mask**: The result will be a binary image where the lady is separated from the background.

```
img = cv.imread('portrait_lady.png', cv.IMREAD_GRAYSCALE)
ret2, th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
cv2_imshow(th2)

```
![alt text](https://github.com/vasanthgx/image_processing/blob/main/images/pic4.png)



### Extracting the edges of just the person using only the morphological operations

Morphological operations in image processing are techniques that process images based on shapes. They apply a structuring element to an input image, producing an output image of the same size. The primary operations are dilation and erosion. Dilation adds pixels to the boundaries of objects, making them larger, while erosion removes pixels from object boundaries, making them smaller. These operations can be combined into more complex operations like opening (erosion followed by dilation) and closing (dilation followed by erosion). Morphological operations are used for tasks such as noise removal, image enhancement, object segmentation, and shape analysis.


![alt text](https://github.com/vasanthgx/image_processing/blob/main/images/pic5.png)



### Extracting the edges using the  Canny Edge Detector

The Canny edge detector is a popular technique in image processing used to detect a wide range of edges in images. It operates by identifying points where the gradient of intensity changes sharply, indicating an edge. The process involves several steps: smoothing the image to reduce noise using a Gaussian filter, computing the gradient magnitude and direction, applying non-maximum suppression to thin the edges, and finally, using hysteresis thresholding to detect and link edges based on high and low thresholds. The Canny detector is widely used in tasks like object detection, image segmentation, and feature extraction due to its robustness and accuracy.

```
img = cv.imread('portrait_lady.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
edges = cv.Canny(img,100,200)
 
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
 
plt.show()

```
![alt text](https://github.com/vasanthgx/image_processing/blob/main/images/pic6.png)

### Using Grabcut method to segment the given image

The GrabCut algorithm is a powerful tool in image processing for foreground extraction and image segmentation. It refines an initial rough segmentation of an image into foreground and background regions. The process starts with the user defining a rectangle around the object of interest. The algorithm then iteratively uses a combination of graph cuts and Gaussian Mixture Models (GMM) to model the foreground and background. By minimizing the energy function that represents the color distributions and smoothness of the edges, GrabCut effectively separates the object from the background. This technique is widely used in applications requiring precise object extraction and image editing.

```
img = cv.imread('portrait_lady.png')

mask = np.zeros(img.shape[:2],np.uint8)
 
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
 
rect = (1,1,288,175)

cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
 
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
 
plt.imshow(img)
# ,plt.colorbar(),plt.show()
```

![alt text](https://github.com/vasanthgx/image_processing/blob/main/images/pic7.png)




   
### Key Observations

Key observations for this project on image processing using OpenCV include:

- **Versatility of OpenCV**: OpenCV provides extensive functionalities for various image processing tasks, making it an essential tool for beginners and professionals.
- **Efficiency**: OpenCV's optimized algorithms enable real-time processing, crucial for applications like video analysis and interactive systems.
- **HSV Color Space**: Utilizing the HSV color space enhances tasks like color segmentation due to its intuitive representation of colors.
- **Histogram Equalization**: This technique significantly improves image contrast, aiding in better feature extraction and visualization.
- **Otsu's Thresholding**: It effectively binarizes images for object extraction without manual threshold selection.
- **Morphological Operations**: These operations are vital for noise removal and shape analysis.
- **Canny Edge Detection**: This method provides accurate edge detection, crucial for object recognition.
- **GrabCut Algorithm**: It excels in precise foreground extraction, important for image editing and segmentation tasks.








## References

1.	[OpenCV documentation](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

2.  [Interactive Foreground Extraction using GrabCut Algorithm](https://docs.opencv.org/4.x/d8/d83/tutorial_py_grabcut.html)







## Contact

If you have any feedback/are interested in collaborating, please reach out to me at vasanth_1627@gmail.com


## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

