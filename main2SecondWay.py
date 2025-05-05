import os
from detection_stars import starsDetection, drawRectangles, findMatching, save_stars_to_txt
from functionForMain import process_images_in_folder
from secondWayMatching import findMatchingStars, showStars


"""
Main of Matching stars - secondWayMatching

Hungarian Algo


  output_dir = "outputMatching"
  
"""


# Loop through the range of images
for i in range(3046, 3063):  # start from 3046 to 3061
    file1 = os.path.join("data_images", f"IMG_{i}.jpg")

    # Check if file1 exists
    if not os.path.exists(file1):
        print(f"Error: File does not exist: {file1}")
        continue  # Skip this image and continue with the next one

    for j in range(i + 1, 3063):  # Compare with subsequent images, i+1 to 3062 (not going beyond 3062)
        file2 = os.path.join("data_images", f"IMG_{j}.jpg")

        # Check if file2 exists
        if not os.path.exists(file2):
            print(f"Error: File does not exist: {file2}")
            continue  # Skip this comparison if the second file doesn't exist

        try:
            print(f"Matching images: {file1} VS {file2}")
            temp_list = findMatchingStars(file1, file2)

            # Define the output directory and create it if it doesn't exist
            output_dir = "outputMatching"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Create the output filename for each pair
            output_img = os.path.join(output_dir, f"output{str(i)}vs{str(j)}.jpg")

            # Call the function to show the stars and save the result
            showStars(file1, file2, temp_list, output_img)

            # Check if temp_list is empty
            if not temp_list:
                print(f"Warning: No matches found between {file1} and {file2}")
                continue  # Skip this pair if no matches found

            # Print the matched stars' IDs
            tmp = [(m[0].id, m[1].id) for m in temp_list]
            print(file1, "vs.", file2, "= ", tmp)
        except Exception as e:
            print(f"Error processing {file1} and {file2}: {e}")