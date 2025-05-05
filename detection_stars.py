"""
This project implements a computer vision pipeline designed to detect and match star-like features across two astronomical images.
The goal is to identify corresponding constellations or feature groups by detecting bright spots (representing stars), fitting lines using RANSAC, and calculating similarity transformations to align one image with another.

The core components of this system are:

Star Detection: Bright objects are extracted from grayscale images using intensity thresholding and contour filtering to identify their centers and brightness.

RANSAC Line Fitting: Lines are fitted through detected stars to find dominant alignments. The top three best-fitting lines are selected based on the number of inliers.

Shape Similarity Matching: Groups of three points from both images are compared using relative geometric distances to find the best match.

Transformation Calculation: Once matched, the scale, rotation, and translation needed to align the two images are computed using the corresponding triangles.

Image Alignment: Using the computed transformation, the system transforms the detected points to estimate their new positions in the second image.
"""
import os
import re

"""
This program compares two images of stars and tries to find how they match.

Here's what the code does:

1. Finds the stars in each image by looking for bright spots.
2. Looks for lines of stars in each image using a method called RANSAC.
3. Tries to find three stars in image 1 that match three stars in image 2
   by checking if the distances between the stars are similar.
4. If it finds a good match, it calculates how to rotate, move, and scale image 1
   so it lines up with image 2.
5. It also draw rectangles to show the results on the images.

"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random
import math


# Detection
def starsDetection(img):
    """
      Detect star-like bright objects in an image based on contours.

      Args:
          img (np.ndarray): BGR image.

      Returns:
          list: Detected star centers as (x, y, radius, brightness).
      """
    tmp_min = 0
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Apply a threshold to create a binary image
    ret, thresh = cv.threshold(img, 179, 255, cv.THRESH_BINARY)

    # Find contours in the binary image
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Loop through the contours and find the centers of the stars
    centers = []
    eps = 1 / 225000
    for cnt in contours:
        # Calculate the bounding box of the contour
        x, y, w, h = cv.boundingRect(cnt)

        # Check if the bounding box is too close to any of the previous centers
        too_close = False
        for center in centers:
            if abs(center[0] - x) < 50 and abs(center[1] - y) < 50:
                too_close = True
                break
        scale = img.shape[0] * img.shape[1] * eps
        if not too_close and max(w, h) < scale:
            # Calculate the center of the star
            center_x = x + w // 2
            center_y = y + h // 2
            brightness = img[center_y, center_x]
            centers.append((x, y, max(w, h), brightness))

    return centers

def initialDetection(img):
    """
    Perform initial detection of bright objects using thresholding and morphology.

    Args:
        img (np.ndarray): BGR image.

    Returns:
        list: Detected centers as (x, y, radius, brightness).
    """
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply a threshold to create a binary image
    ret, thresh = cv.threshold(img, 160, 255, cv.THRESH_BINARY)

    # Apply morphological operations
    kernel = np.ones((6, 6), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)

    # Find contours
    contours, hierarchy = cv.findContours(opening, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Loop through the contours and find the centers of the stars
    centers = []
    for cnt in contours:
        # Calculate the minimum enclosing circle of the contour
        (center_x, center_y), radius = cv.minEnclosingCircle(cnt)

        # Check if the center is too close to any of the previous centers
        too_close = False
        for center in centers:
            if abs(center[0] - center_x) < 50 and abs(center[1] - center_y) < 50:
                too_close = True
                break
        if not too_close:
            # Get the brightness value of the center pixel
            brightness = img[int(center_y), int(center_x)]

            # Add the center to the list
            centers.append((int(center_x), int(center_y), int(radius), brightness))

    return centers



# RANSAC
def drawLine(image, m, b, color=(255, 255, 255), thickness=2):
    """Draws a line on an image given a slope and y-intercept."""

    """
     Draw a line on an image using the given slope and y-intercept.

     Args:
         image (np.ndarray): The image on which to draw the line.
         m (float): Slope of the line.
         b (float): Y-intercept of the line.
         color (tuple): Color of the line in BGR format. Default is white.
         thickness (int): Thickness of the line. Default is 2.

     Returns:
         np.ndarray: The image with the drawn line.
     """

    h, w = image.shape[:2]

    # Compute the starting and ending points of the line
    x1 = 0
    y1 = int(b)
    x2 = w - 1
    y2 = int(m * x2 + b)

    # Draw the line on the image
    cv.line(image, (x1, y1), (x2, y2), color, thickness)

    return image


def sortTopThree(lst, points_on_line, line, count):
    """
    Maintain the top three lines with the highest inlier count.

    Args:
        lst (list): List of the top three lines so far.
        points_on_line (list): Points close to the current line.
        line (tuple): Current line parameters (slope, intercept).
        count (int): Number of inliers for the current line.

    Returns:
        list: Updated top three list.
    """
    candidate = points_on_line, count, line
    if candidate not in lst:
        if lst[0][1] < count:
            lst[0] = candidate
        elif lst[1][1] < count:
            lst[1] = candidate
        elif lst[2][1] < count:
            lst[2] = candidate
        else:
            return lst
    return lst


def ransacLineFit(points, threshold=20, max_iterations=5000):
    """
       Fit a line to a set of points using the RANSAC algorithm.

       Args:
           points (list): List of (x, y, radius, brightness) points.
           threshold (float): Distance threshold to consider a point an inlier.
           max_iterations (int): Number of iterations to run RANSAC.

       Returns:
           tuple: Best line fit (slope, intercept), inlier points, top three lines.
       """
    best_fit = None
    best_count = 0
    points_on_line = []
    ans = []
    top_three = [[(0, 0), 0], [(0, 0), 0], [(0, 0), 0]]
    for i in range(max_iterations):
        # Randomly select two points from the set
        sample = random.sample(points, 2)
        curr_points = []
        # Fit a line to the selected points
        x1, y1, r, b = sample[0]
        x2, y2, r, b = sample[1]
        if x1 == x2:
            continue  # avoid division by zero
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        # Count the number of inliers
        count = 0
        for x, y, r, c in points:
            if abs(y - (m * x + b)) < threshold:
                count += 1
                curr_points.append([x, y, r, c])
        # Update the best-fit line if we found more inliers than before
        top_three = sortTopThree(top_three, curr_points, (m, b), count)
        if count > best_count and count >= 3:
            best_fit = (m, b)
            best_count = count
            points_on_line = curr_points
            ans.insert(0, best_fit)

    return best_fit, points_on_line, top_three






def drawRectangles(img, rectangles, color):
    """
       Draw rectangles around given points on an image.

       Args:
           img (np.ndarray): The image to draw on.
           rectangles (list): List of rectangles as (x, y, radius, brightness).
           color (tuple): Color for the rectangle.

       Returns:
           None
       """
    for rect in rectangles:
        cv.rectangle(img, (rect[0] - 40, rect[1] - 40), (rect[0] + rect[2] + 40, rect[1] + rect[2] + 40), color, 7)


def drawRectanglesManually(img, rectangles):
    """
    Draw red rectangles around given coordinates manually.

    Args:
        img (np.ndarray): The image to draw on.
        rectangles (list): List of rectangles as (x, y, radius, brightness).

    Returns:
        None
    """
    for rect in rectangles:
          cv.rectangle(img, (rect[0]-40,rect[1]-40), (rect[0]+rect[2]+40,rect[1]+rect[2]+40), (0,0,255), 7)

# Find matching Stars
def similarity(line_points1, line_points2):
    """
        Compare similarity of two sets of 3 points by relative distances.

        Args:
            line_points1 (list): First set of points.
            line_points2 (list): Second set of points.

        Returns:
            tuple: Minimum difference and best-matching 3 points from line_points1.
        """
    AB = distance(line_points2[0], line_points2[1])
    BC = distance(line_points2[1], line_points2[2])
    AC = distance(line_points2[2], line_points2[0])
    min_diff = 50000
    same_points = []
    for i in range(len(line_points1)-2):
      curr_points = []
      DE = distance(line_points1[i], line_points1[i+1])
      EF = distance(line_points1[i+1], line_points1[i+2])
      FD = distance(line_points1[i+2], line_points1[i])
      curr_points.append(line_points1[i])
      curr_points.append(line_points1[i+1])
      curr_points.append(line_points1[i+2])
      curr_diff = abs(abs(AB/DE-BC/EF) + abs(BC/EF-AC/FD) + abs(AB/DE-AC/FD))
      if curr_diff < min_diff:
        min_diff = curr_diff
        same_points = curr_points
    for point in same_points:
      point = tuple(point)

    return min_diff, same_points

def distance(point1, point2):
  """
    Calculate Euclidean distance between two points.

    Args:
        point1 (tuple): First point (x, y, radius, brightness).
        point2 (tuple): Second point (x, y, radius, brightness).

    Returns:
        float: Euclidean distance between the two points.
   """

  x1, y1, r1, b1 = point1
  x2, y2, r2, b2 = point2
  return math.sqrt((x2-x1)**2+(y2-y1)**2)



# Transformation and Translation
def computeTransformation(best_correlation):
  """
    Compute the scale, rotation matrix, and translation vector between two sets of three matching points.

    Args:
        best_correlation (tuple): Two sets of three matching points.

    Returns:
        tuple: Scale factor, rotation matrix (2x2), translation vector (2x1).
 """
  x1_1, y1_1, r1_1, b1_1 = best_correlation[0][0]
  x1_2, y1_2, r1_2, b1_2 = best_correlation[0][1]
  x1_3, y1_3, r1_3, b1_3 = best_correlation[0][2]
  x2_1, y2_1, r2_1, b2_1 = best_correlation[1][0]
  x2_2, y2_2, r2_2, b2_2 = best_correlation[1][1]
  x2_3, y2_3, r2_3, b2_3 = best_correlation[1][2]

  # Define the points in img1 and img2 as numpy arrays
  img1_points = np.array([[x1_1, y1_1], [x1_2, y1_2], [x1_3, y1_3]])
  img2_points = np.array([[x2_1, y2_1], [x2_2, y2_2], [x2_3, y2_3]])

  # Compute the difference vectors
  v1 = img1_points[1] - img1_points[0]
  v2 = img2_points[1] - img2_points[0]

  # Compute the scaling factor
  s = np.linalg.norm(v2) / np.linalg.norm(v1)

  # Compute the rotation matrix
  v1_unit = v1 / np.linalg.norm(v1)
  v2_unit = v2 / np.linalg.norm(v2)
  cos_theta = np.dot(v1_unit, v2_unit)
  sin_theta = np.cross(v1_unit, v2_unit)
  R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

  # Compute the translation vector
  t = img2_points[0] - s * R.dot(img1_points[0])

  return s, R, t

# transformation function
def transform(x1, y1, s, R, t):
    """
      Apply a similarity transformation (scale, rotate, translate) to a point.

      Args:
          x1 (float): X-coordinate of the point.
          y1 (float): Y-coordinate of the point.
          s (float): Scaling factor.
          R (np.ndarray): Rotation matrix.
          t (np.ndarray): Translation vector.

      Returns:
          tuple: Transformed (x, y) coordinates.
      """
    p = np.array([x1, y1])
    p_transformed = s * R.dot(p) + t
    return p_transformed[0], p_transformed[1]






######### Matching Images  #########


def save_stars_to_txt(stars, filename):
    """
    Save the coordinates (x, y), radius (r), and brightness (b) of stars to a text file.

    Args:
        stars (list): List of detected stars as (x, y, radius, brightness).
        filename (str): Name of the file where data should be saved.
    """
    with open(filename, 'w') as file:
        # Write header
        file.write("x, y, radius, brightness\n")

        # Write each star's data
        for star in stars:
            x, y, r, b = star
            file.write(f"{x}, {y}, {r}, {b}\n")



def findMatching(img1_path, img2_path):
  """
       Find matching patterns between two images using RANSAC line fitting and similarity metrics.

       Args:
           img1_path (str): Path to the first image.
           img2_path (str): Path to the second image.

       Returns:
           None
           """

  # Load the image
  img1 = cv.imread(img1_path)
  img2 = cv.imread(img2_path)

  orig_img1 = img1
  orig_img2 = img2

  rect1 = initialDetection(img1)
  rect2 = initialDetection(img2)




  # Find the lines using RANSAC
  lines1, line_points1, top_three1 = ransacLineFit(rect1)
  lines2, line_points2, top_three2 = ransacLineFit(rect2)

  best_correlation = None  # כדי למנוע שגיאה אם לא נכנס ל-if

  best_diff = 2
  for three_points1, c1, (m1, b1) in top_three1:
    for three_points2, c2, (m2, b2) in top_three2:
      curr_diff = similarity(three_points1, three_points2)[0]
      if curr_diff < best_diff:
        best_diff = curr_diff
        best_line = ((m1, b1), (m2, b2))
        best_correlation = (three_points1[:3], three_points2[:3])

  # Default values for s, R, t
  s, R, t = 0, np.eye(2), np.zeros(2)

  if best_correlation is not None:
      s, R, t = computeTransformation(best_correlation)


  transformed_points = []

  # Detection
  img1 = orig_img1
  img2 = orig_img2

  s_points = starsDetection(img1)
  drawRectangles(img1, s_points, (0,0,255))

  # Transform
  for x, y, r, b in s_points:
    newx, newy = transform(x,y, s, R, t)
    transformed_points.append((int(newx), int(newy), r, b))

  drawRectanglesManually(img2, transformed_points)

##

  # Display images side by side
  plt.figure(figsize=(12, 6))  # Adjust figure size as needed

  # Display the first image
  plt.subplot(1, 2, 1)
  plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
  plt.title("Image 1")
  plt.axis('off')  # Hide axes for a cleaner look

  # Display the second image
  plt.subplot(1, 2, 2)
  plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
  plt.title("Image 2")
  plt.axis('off')  # Hide axes for a cleaner look

  # Show the images
  plt.tight_layout()  # Adjust layout for better spacing
  plt.show()

 #SHOW EACH IMAGE ALONE
  #plt.imshow(img1), plt.show()
 # plt.imshow(img2), plt.show()


  # Define the file path
  file_path = os.path.join("outputMatchingDetectionStars", "correlation_output.txt")

  # Determine the next available filename
  existing_files = os.listdir("outputMatchingDetectionStars")
  pattern = re.compile(r'correlation_output(\d+)\.txt')
  numbers = [int(match.group(1)) for fname in existing_files if (match := pattern.match(fname))]
  next_number = max(numbers, default=0) + 1
  file_name = f"correlation_output{next_number}.txt"
  file_path = os.path.join("outputMatchingDetectionStars", file_name)

  print("### Correlation ###")

  with open(file_path, "w") as f:
      f.write("### Correlation ###\n")
      for i in range(len(transformed_points)):
          tmp = transformed_points[i]
          if all(val >= 0 for val in tmp):
              transformed_str = ', '.join(
                  str(int(val)) if isinstance(val, np.uint8) else str(val)
                  for val in transformed_points[i])
              s_points_str = ', '.join(
                  str(int(val)) if isinstance(val, np.uint8) else str(val)
                  for val in s_points[i])
              line = f"{s_points_str} => {transformed_str}"
              print(line)
              f.write(line + "\n")

#########################

