import os
from detection_stars import starsDetection, drawRectangles, findMatching, save_stars_to_txt
from functionForMain import process_images_in_folder
from secondWayMatching import findMatchingStars, showStars

"""
Main of Matching stars - detections_stars

Ransac Algo

"""


#matching stars
findMatching('boaz_data_test/fr1.jpg', 'boaz_data_test/ST_db1.png')
findMatching('boaz_data_test/fr1.jpg', 'boaz_data_test/ST_db2.png')
findMatching('boaz_data_test/fr2.jpg', 'boaz_data_test/ST_db2.png')
findMatching('boaz_data_test/fr1.jpg', 'boaz_data_test/fr2.jpg')
findMatching('boaz_data_test/ST_db1.png', 'boaz_data_test/ST_db2.png')
findMatching('data_images/IMG_3052.jpg', 'data_images/IMG_3053.jpg')






"""
for i in range(3046, 3062):  # stop at 3061, last pair is 3061 and 3062
    file1 = os.path.join("data_images", f"IMG_{i}.jpg")
    file2 = os.path.join("data_images", f"IMG_{i + 1}.jpg")

    # Check if the files exist
    if not os.path.exists(file1) or not os.path.exists(file2):
        print(f"Error: One of the files does not exist: {file1} or {file2}")
        continue  # Skip this pair and continue with the next pair

    try:
        print(f"Matching images: {file1} VS {file2}")
        temp_list = findMatchingStars(file1, file2)


        output_dir = "outputMatching"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_img = os.path.join(output_dir, f"output{str(i)}vs{str(i + 1)}.jpg")
        #output_img = os.path.join(output_dir, f"output{str(i)}.jpg")
        showStars(file1, file2, temp_list, output_img)

        # Check if temp_list is empty
        if not temp_list:
            print(f"Warning: No matches found between {file1} and {file2}")
            continue  # Skip this pair if no matches found

        tmp = [(m[0].id, m[1].id) for m in temp_list]
        print(file1, "vs.", file2, "= ", tmp)
    except Exception as e:
        print(f"Error processing {file1} and {file2}: {e}")

        """
