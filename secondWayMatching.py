"""
Introduction:
This program aims to match stars between two images by detecting their coordinates and finding optimal pairings using the Hungarian algorithm. The steps involved in this process include:

1. **Star Detection**: The program detects stars in two input images by processing their contours and calculating the center of bounding rectangles around the detected areas.
2. **Matching Stars**: Once the stars are detected, the program calculates a distance matrix between the star coordinates of the two images. Using the Hungarian algorithm, the program identifies the optimal matching of stars between the two images.
3. **Visualization**: After the matching process, the program draws circles around the matching stars and adds labels to indicate their corresponding IDs. The final result is a side-by-side comparison of the two images with the matching stars highlighted.
4. **File Output**: The program has the option to save the matching pairs in a file and the processed images with the marked stars for further analysis.

This approach allows us to efficiently align and compare star patterns between two images, useful in astronomical observations, image registration, and other domains requiring precise matching of points between two visual datasets.
"""

from detection_stars import ransacLineFit, computeTransformation, initialDetection, drawRectangles
from star import Star

import cv2
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random
import math



######### Matching Images  #########


def findMatchingStars(img1_path, img2_path, output_path=None):
        """
          Find matching pairs of stars between two input images.

          Parameters:
              - img1_path (str): path to the first image file
              - img2_path (str): path to the second image file
              - output_path (str, optional): path to output file to write matching pairs (default=None)

          Returns:
              - matching_pairs (list of tuples): list of matching star pairs between the two images,
                  where each pair is a tuple of Star objects representing the corresponding stars in each image
          """
        # Find coordinates of stars in the first image
        stars1 = findCoordinates(img1_path)
        # Draw the first image with the found stars marked
        drawImage(img1_path, stars1)
        # Find coordinates of stars in the second image
        stars2 = findCoordinates(img2_path)
        # Draw the second image with the found stars marked
        drawImage(img2_path, stars2)
        # Compute the distance matrix between the star coordinates in the two images
        distances = distance_matrix(np.array([(s.x, s.y) for s in stars1]), np.array([(s.x, s.y) for s in stars2]))
        # Use the Hungarian algorithm to find the optimal matching between the stars in the two images
        row_ind, col_ind = linear_sum_assignment(distances)
        # Create a list of matching star pairs
        matching_pairs = []
        for r, c in zip(row_ind, col_ind):
            matching_pairs.append((stars1[r], stars2[c]))
        # Write the matching pairs to an output file if an output path is provided
        if output_path is not None:
            with open(output_path, "w") as file:
                for pair in matching_pairs:
                    p = (pair[0].id, pair[1].id)
                    file.write(str(p) + "\n")
        # Return the list of matching star pairs
        return matching_pairs


def drawImage(img_path, coordinates):
    """
     This function draws circles on the input image to mark the detected stars.

     Args:
         img_path (str): Path to the image file where stars are to be marked.
         coordinates (list): A list of Star objects, where each Star object contains information
                             about the star's position (x, y) and radius (r).

     Returns:
         None: The function saves the processed image with marked stars as a new image file.
     """
    # Load image from file
    image = cv2.imread(img_path)

    # Draw circles on the image where there are stars
    for star in coordinates:
        cv2.circle(image, (int(star.x), int(star.y)), int(star.r) + 5, (0, 255, 255), 5)

    # Save the processed image to file
    filename = "%s_processedSecondWay.jpg" % img_path
    cv2.imwrite(filename, image)
    cv2.destroyAllWindows()

def findCoordinates(img_path, output_path=None):
        """
        This function finds the coordinates of stars in an image using image processing techniques.

        Parameters:
        img_path (str): The path of the image file.
        output_path (str): The path of the output file. If None, the function does not save the results.

        Returns:
        coordinates (list): A list of Star objects representing the stars found in the image.
        """
        # Load the image and convert it to grayscale
        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to the image
        _, thresh = cv2.threshold(gray, 180, 220, cv2.THRESH_BINARY)

        # Find the contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize variables
        coordinates = []
        check = []
        count = 0

        # Loop through the contours
        for i, c in enumerate(contours):
            # Get the coordinates and size of the bounding rectangle around the contour
            x, y, w, h = cv2.boundingRect(c)

            # Calculate the radius of the circle that encloses the rectangle
            r = int((w + h) / 4)

            # Calculate the average brightness of the region inside the rectangle
            b = int(gray[y:y + h, x:x + w].mean())

            # Calculate the center coordinates of the rectangle
            x = x + w / 2
            y = y + h / 2

            # Create a Star object with the calculated parameters
            st = Star(x, y, r, b, count, image)

            # Check if the coordinates have already been added
            ans = (x, y)
            if ans in check:
                continue

            # Add the Star object to the list of coordinates
            coordinates.append(st)
            check.append(ans)
            count += 1

        # If an output path is provided, write the coordinates to a file
        if output_path is not None:
            with open(output_path, 'w') as f:
                for result in coordinates:
                    f.write(f"{result.id},{result.x},{result.y},{result.r},{result.b}\n")

        # Return the list of coordinates
        return coordinates



def showStars(filename_1, filename_2, matching, output_file):

    """
    This function visualizes the matching stars between two images by drawing circles and
    labels on the corresponding stars in each image. It then combines the two images side-by-side
    for comparison and saves the result as a new image file.

    Args:
        filename_1 (str): Path to the first image file.
        filename_2 (str): Path to the second image file.
        matching (list of tuples): A list of tuples where each tuple contains two Star objects,
                                   one from each image, representing the matching stars.
        output_file (str): The path where the resulting image with matching stars will be saved.

    Returns:
        None: The function saves the side-by-side image with labeled matching stars to the output path.
    """

    # Load images from file
    image1 = cv2.imread(filename_1)
    image2 = cv2.imread(filename_2)

    # Iterate over matching stars and draw circles and labels on both images
    for match in matching:
        star1 = match[0]
        star2 = match[1]

        # Draw circle and label on first image
        cv2.circle(image1, (int(star1.x), int(star1.y)), int(star1.r) + 5, (0, 255, 255), 5)
        cv2.putText(image1, "ID: " + star1.id, (int(star1.x) + int(star1.r) + 5, int(star1.y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.5,
                    (0, 0, 255), 2)

        # Draw circle and label on second image
        cv2.circle(image2, (int(star2.x), int(star2.y)), int(star2.r) + 5, (0, 255, 255), 5)
        cv2.putText(image2, "ID: " + star2.id, (int(star2.x) + int(star2.r) + 5, int(star2.y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.5,
                    (0, 0, 255), 2)

    # Resize second image to match the size of the first image, and concatenate the images side-by-side
    img2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    result = cv2.hconcat([image1, img2_resized])

    # Save the resulting image to file
    cv2.imwrite(output_file, result)
    cv2.destroyAllWindows()