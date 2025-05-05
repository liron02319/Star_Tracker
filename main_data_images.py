

from detection_stars import starsDetection, drawRectangles, findMatching, save_stars_to_txt
from functionForMain import process_images_in_folder

"""
Main that output the coordinates of the Imgs at data_images folder 

output_folder = "output_data_images"

"""

# Folder paths
input_folder = "data_images"
output_folder = "output_data_images"

# Process all images in the folder and save coordinates
process_images_in_folder(input_folder, output_folder)



