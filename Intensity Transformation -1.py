import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Define the breakpoints for the intensity transformation
c = np.array([(50, 100), (150, 255)])

# Create the transformation function for each segment
t1 = np.linspace(0, c[0,0], c[0,0] + 1 - 0).astype('uint8')
t2 = np.linspace(c[0,0] + 1, c[1,1], c[1,0] - c[0,0]).astype('uint8')
t3 = np.linspace(c[1,0] + 1, 255, 255 - c[1,0]).astype('uint8')

# Combine the transformations
transform = np.concatenate((t1, t2), axis=0).astype('uint8')
transform = np.concatenate((transform, t3), axis=0).astype('uint8')

# Print the length of the transformation (it should be 256, one for each intensity level)
print(len(transform))

# Load the grayscale image
img_orig = cv.imread(r'C:\Users\USER\Desktop\Lectures\SEM 5\EN3160 - Image Processing and Machine Vision\Assignment 1\Intensity_Transformation\Img\emma.jpg', cv.IMREAD_GRAYSCALE)

# Apply the intensity transformation using cv.LUT
image_transformed = cv.LUT(img_orig, transform)

# Display the original and transformed image
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(img_orig, cmap='gray')
plt.title('Original Image')

# Transformed image
plt.subplot(1, 2, 2)
plt.imshow(image_transformed, cmap='gray')
plt.title('Transformed Image')

plt.show()

# Save the result
cv.imwrite('image_transformed.jpg', image_transformed)
