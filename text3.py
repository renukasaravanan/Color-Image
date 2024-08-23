import pytesseract  # type: ignore
import cv2  # type: ignore
import re
import numpy as np # type: ignore
from PIL import Image  # type: ignore

# Function to detect emails and URLs
def detect_emails_and_urls(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    emails = re.findall(email_pattern, text)
    urls = re.findall(url_pattern, text)

    return emails + urls

# Extract text from image using pytesseract
image_path = r'C:\Users\Renuka.DESKTOP-RE39MJS\Desktop\Color Image\images\black and color.PNG'
extracted_text = pytesseract.image_to_string(Image.open(image_path))
print("Extracted Text:", extracted_text)

# Detect emails and URLs in the text
emails_and_urls = detect_emails_and_urls(extracted_text)
print("Detected Emails and URLs:", emails_and_urls)

# Read the original image
original_image = cv2.imread(image_path)
modified_image = original_image.copy()

# Convert emails and URLs to black in the image
for item in emails_and_urls:
    h, w, _ = modified_image.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    text_boxes = pytesseract.image_to_boxes(Image.open(image_path))  # Get bounding boxes
    
    for box in text_boxes.splitlines():
        box = box.split(' ')
        char, x, y, x2, y2, _ = box[0], int(box[1]), int(box[2]), int(box[3]), int(box[4]), int(box[5])
        if char in item:
            cv2.rectangle(modified_image, (x, h - y), (x2, h - y2), (0, 0, 0), -1)

# Convert the modified image to grayscale
gray_image = cv2.cvtColor(modified_image, cv2.COLOR_BGR2GRAY)

# Convert original image to grayscale for comparison
original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Check if there are still colored regions by comparing the images
diff = cv2.absdiff(original_gray, gray_image)
colored_regions = cv2.countNonZero(diff)

if colored_regions > 0:
    print("The image contains colored parts other than emails/URLs. It is classified as a colored image.")
else:
    print("The image is classified as black and white.")

# Display images
cv2.imshow('Modified Image', modified_image)
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
