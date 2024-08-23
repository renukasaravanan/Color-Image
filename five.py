import cv2  # type: ignore
import numpy as np  # type: ignore
import pytesseract  # type: ignore
import re
import os

def mask_email_and_url_areas(image, text_data):
    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for i, box in enumerate(text_data.splitlines()):
        if i == 0:  # skip header
            continue
        
        b = box.split()
        text = b[0]
        x1, y1, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        
        # Convert y-coordinates since pytesseract uses a different origin (bottom-left)
        y1, y2 = h - y1, h - y2
        
        # Check if the text is an email or URL
        if re.match(r'[\w\.-]+@[\w\.-]+', text) or re.match(r'(www\.|https?://)\S+', text):
            # Mask this area as white (255)
            cv2.rectangle(mask, (x1, y2), (x2, y1), (255), -1)
    
    # Mask the email/URL areas in the original image
    masked_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    return masked_image, mask

def is_color_image(image, filename):
    # Extract text data with bounding boxes from the image
    text_data = pytesseract.image_to_boxes(image)
    text = pytesseract.image_to_string(image)
    
    # Mask the areas containing emails and URLs
    masked_image, mask = mask_email_and_url_areas(image, text_data)
    
    # Convert the masked image to grayscale
    gray_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    
    # Calculate the difference between the color image and the grayscale image
    diff_image = cv2.absdiff(masked_image, cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR))
    
    # Check if there are any non-zero pixels in the diff image
    has_color = np.count_nonzero(diff_image) > 0
    
    # Analyze masked regions
    if has_color:
        # Convert original image to grayscale for checking
        original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Check for non-black pixels in the grayscale image
        non_black_pixels = np.count_nonzero(original_gray < 200)  # Adjust threshold if needed
        
        if non_black_pixels > 0:
            return True  # The image is considered color if non-black pixels are present
    
    return False  # The image is black and white if no non-black pixels are found

def process_images_in_directory(directory_path):
    results = {}
    
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return results
    
    # Loop through all the files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.PNG')):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)
            
            if image is not None:
                classification = is_color_image(image, filename)
                results[filename] = 'Color' if classification else 'Black and White'
            else:
                results[filename] = 'Error loading image'
    
    return results

# Directory containing the images
directory_path = r'C:\Users\Renuka.DESKTOP-RE39MJS\Desktop\Color Image\images'

# Process the images and get the results
results = process_images_in_directory(directory_path)

# Print the results
for filename, classification in results.items():
    print(f"{filename}: {classification}")
