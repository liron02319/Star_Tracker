
<p align="center">
  <img src="https://www.moveshootmove.com/cdn/shop/files/arirex1_0802_4b4aa7c8-46d5-4eea-9c60-339f94a49960_540x.jpg?v=1715072387" alt="Logo" width="450" height="300" align="right">
</p>


#  Star_Tracker

**Welcome to Star Tracker Project !**  
- In This project there a Python implementation of a program that detects stars in two images and matches them based on their coordinates.
- In addition, the results and analysis are presented in a Colab report, which can be found ***here***.


## Team Members
Liron Cohen - 312324247

## Table of Contents
- [Background](#Background)
- [THE TASK](#THE-TASK)
- [MORE INFO](#MORE-INFO)

- [Google Colab notebook](#Google-Colab-notebook,)
- [Detection Example](#Detection-Example)
- [Matching Imgs Example](#Matching-Imgs-Example)



## Background
Star trackers are essential tools used in a variety of fields, particularly in astronomy, aerospace, and satellite navigation. These systems are designed to capture images of the night sky, identify distinct star patterns, and match these patterns to known star catalogs. By comparing the positions of stars in the captured images, star trackers can calculate orientation, attitude, and trajectory, providing critical data for navigation and positioning.

In this project, we implemented a Python-based star tracking system. The program detects stars in two given images and matches them based on their coordinates. The core idea is to extract star-like features from the images, compute their locations, and compare the coordinates between the two images to identify matching stars. 



![selfportrait_startracker_monikadeviat](https://github.com/user-attachments/assets/ab6035f8-fa22-40ba-8367-3668a1557ada)

## THE TASK

## Q1
- Write a simple and effective algorithm to match two pictures - one with hundreds of stars and the other with 10-20 stars. Choose the simplest existing method in this section..

## Q2
- Create a library that takes a star image and converts it to a file of coordinates x, y, r, b where x and y represent the coordinates of each star while r represents the radius and b represents the    brightness.

## Q3
- Create a library that takes two images and calculates the best match between them by generating a list of coordinate pairs that point to the same star in each image.
  The library compare each two images and return a matching in case it detect two similar stars in both files. The results are saved into a result file.

## Q4
- We conducted a large-scale experiment using image datasets to evaluate the performance of our star-matching algorithm.
The results were visualized and analyzed using Python — optionally within Google Colab for convenience and reproducibility.  



## MORE INFO

This project is a Python implementation of a program that detects stars in two images and matches them based on their coordinates.
At this project I used 2 ways for matching -   
1- with RANSAC  
2- Hungarian algorithm.  *** NOTE : after many attempts, it sometimes work.




```text
The core components of this system are:

Star Detection: Bright objects are extracted from grayscale images using intensity thresholding and contour filtering to identify their centers and brightness.

RANSAC Line Fitting: Lines are fitted through detected stars to find dominant alignments. The top three best-fitting lines are selected based on the number of inliers.

Shape Similarity Matching: Groups of three points from both images are compared using relative geometric distances to find the best match.

Transformation Calculation: Once matched, the scale, rotation, and translation needed to align the two images are computed using the corresponding triangles.

Image Alignment: Using the computed transformation, the system transforms the detected points to estimate their new positions in the second image.


////////////////////////////////////////////////////////////////////////////


This program compares two images of stars and tries to find how they match.

Here's what the code does:

1. Finds the stars in each image by looking for bright spots.
2. Looks for lines of stars in each image using a method called RANSAC.
3. Tries to find three stars in image 1 that match three stars in image 2
   by checking if the distances between the stars are similar.
4. If it finds a good match, it calculates how to rotate, move, and scale image 1
   so it lines up with image 2.
5. It also draws rectangles to show the results on the images.

////////////////////////////////////////////////////////////////////////////


Algorithm detect:

Find contours in the image.
Set a list 'Centers' .
Loop through the contours, for each contour CNT:
Calculate the bounding box of CNT (as {x,y,r,b}).
Check if the bounding box of CNT is too close to any of the previous centers, if it too close - break.
If CNT is not to close and the width and height of CNT are in scale, Calculate the center of the star and save it in 'Centers'.
return 'Centers'.

////////////////////////////////////////////////////////////////////////////

1)
Algorithm Matching img: 1 Way

Apply Detection Algorithm on img1 and img2 =result> rect1 and rect2.
Use RANSAC Algorithm to get the two pictures lines and top three points that representing this line.
In each rect, after phase II, there will be three points representing its RANSAC line - then can be interrupted as triangle.
check the similarity of both triangles => which gives you the Ratio.
Compute the transformation between img1 to img2, using the Ratio.
Draw the images after the transformation.


2)
Algorithm Matching img: 2 Way

This program aims to match stars between two images by detecting their coordinates and finding optimal pairings using the Hungarian algorithm. The steps involved in this process include:

1. **Star Detection**: The program detects stars in two input images by processing their contours and calculating the center of bounding rectangles around the detected areas.
2. **Matching Stars**: Once the stars are detected, the program calculates a distance matrix between the star coordinates of the two images. Using the Hungarian algorithm, the program identifies the optimal matching of stars between the two images.
3. **Visualization**: After the matching process, the program draws circles around the matching stars and adds labels to indicate their corresponding IDs. The final result is a side-by-side comparison of the two images with the matching stars highlighted.
4. **File Output**: The program has the option to save the matching pairs in a file and the processed images with the marked stars for further analysis.

This approach allows us to efficiently align and compare star patterns between two images, useful in astronomical observations, image registration, and other domains requiring precise matching of points between two visual datasets.

*** NOTE : after many attempts, it sometimes work.
"""

```


## Google Colab notebook
- In This project there a Python implementation of a program that detects stars in two images and matches them based on their coordinates.
- In addition, the results and analysis are presented in a Colab report, which can be found ***here***.

![COLAB](https://github.com/user-attachments/assets/c55f536c-a63d-4187-ae08-933ae0e8834e)


## Detection Example

***"ST_db1.png"***   
   
![EXAM2](https://github.com/user-attachments/assets/b177f342-923f-450e-afd2-97b16f38b4e0)
   
***"fr1.jpg"***   
   
![EXAM1](https://github.com/user-attachments/assets/1b80fa67-818c-4f03-b080-607bd26dae9c)




## Matching Imgs Example


***findMatching('boaz_data_test/ST_db1.png', 'boaz_data_test/ST_db2.png')***   
   
![img6](https://github.com/user-attachments/assets/a989d684-7a7d-474f-9448-f8bd17e97970)


***findMatching('data_images/IMG_3052.jpg', 'data_images/IMG_3053.jpg')***   
   
![img5](https://github.com/user-attachments/assets/431e7018-8d51-433a-8c77-00765fa5625f)

***for example - the Correlation of***  ***findMatching('data_images/IMG_3052.jpg', 'data_images/IMG_3053.jpg')***   
***the Correlation is*** [here](https://github.com/liron02319/Star_Tracker/blob/master/outputMatchingDetectionStars/correlation_output5.txt)

