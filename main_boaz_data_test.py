import os

import cv2 as cv
from matplotlib import pyplot as plt

from detection_stars import starsDetection, drawRectangles, findMatching, save_stars_to_txt
from functionForMain import process_images_in_folder

"""
Main that output the coordinates of the Imgs test at boaz_data_test folder 

output_folder = "output_boaz_data_test"

"""

# Folder paths
input_folder = "boaz_data_test"
output_folder = "output_boaz_data_test"

# Process all images in the folder and save coordinates
process_images_in_folder(input_folder, output_folder)

folder_path = "boaz_data_test"

# List of image filenames
image_filenames = [
    "fr1.jpg",
    "fr2.jpg",
    "ST_db1.png",
    "ST_db2.png",

]

# Load images dynamically
images = [cv.imread(os.path.join(folder_path, filename)) for filename in image_filenames]

# Process each image
for img in images:
    if img is not None:  # Check if image is loaded successfully
        points = starsDetection(img)
        drawRectangles(img, points, (255, 0, 0))  # Draw rectangles on the image
        plt.imshow(img)  # Convert BGR to RGB for displaying with plt
        plt.show()
    else:
        print("❌ Failed to load one of the images.")



#MY TEST

folder_path = "data_images"

# List of image filenames
image_filenames = [
    "IMG_3052.jpg",
    "IMG_3053.jpg",


]

# Load images dynamically
images = [cv.imread(os.path.join(folder_path, filename)) for filename in image_filenames]

# Process each image
for img in images:
    if img is not None:  # Check if image is loaded successfully
        points = starsDetection(img)
        drawRectangles(img, points, (255, 0, 0))  # Draw rectangles on the image
        plt.imshow(img)  # Convert BGR to RGB for displaying with plt
        plt.show()
    else:
        print("❌ Failed to load one of the images.")


#matching stars

#findMatching('boaz_data_test/fr1.jpg', 'boaz_data_test/fr1copy.jpg')
#findMatching('boaz_data_test/fr1.jpg', 'boaz_data_test/ST_db1.png')
#findMatching('boaz_data_test/fr1.jpg', 'boaz_data_test/ST_db2.png')
#findMatching('boaz_data_test/fr2.jpg', 'boaz_data_test/ST_db2.png')
#findMatching('boaz_data_test/fr1.jpg', 'boaz_data_test/fr2.jpg')
#findMatching('boaz_data_test/ST_db1.png', 'boaz_data_test/ST_db2.png')
#findMatching('data_images/IMG_3052.jpg', 'data_images/IMG_3053.jpg')


