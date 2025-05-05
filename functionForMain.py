import os
import cv2 as cv
from detection_stars import starsDetection

#HELp FUNCTIONS FOR MAIN


def save_coordinates_to_txt(image_path, coordinates, output_folder):
    """
    Save the coordinates of detected stars to a text file in the specified folder.
    Args:
        image_path (str): Path to the image file.
        coordinates (list): List of detected star centers (x, y, radius, brightness).
        output_folder (str): Path to the folder where the text files will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_file = os.path.join(output_folder, f"{base_name}_coordinates.txt")

    with open(output_file, 'w') as f:
        f.write("x,y,r,b\n")  # Header row
        for coord in coordinates:
            f.write(f"{coord[0]}, {coord[1]}, {coord[2]}, {int(coord[3])}\n")

    print(f"✅ Coordinates saved to {output_file}")

def process_images_in_folder(folder_path, output_folder):
    """
    Process all images in the given folder and save detected star coordinates to text files.
    Args:
        folder_path (str): Path to the folder containing input images.
        output_folder (str): Path to the folder to save output text files.
    """

    for filename in os.listdir(folder_path):
        # Skip non-image files or already-processed images
        if (
                not filename.lower().endswith(('.jpg', '.png', '.jpeg'))
                or '_processedSecondWay' in filename
        ):
            continue

        # Now process the valid image
        image_path = os.path.join(folder_path, filename)
        img = cv.imread(image_path)
        if img is None:
            print(f"❌ Failed to load {filename}")
            continue

        coordinates = starsDetection(img)
        save_coordinates_to_txt(image_path, coordinates, output_folder)