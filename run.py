import cv2
import pytesseract
import json
import os
from datetime import datetime
import re  # Import the regex module for validation

# Path to the JSON file for storing entry times
DATA_FILE = 'parking_data.json'

# Load saved data from JSON file
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as file:
            return json.load(file)
    return {}

# Save data to JSON file
def save_data(data):
    with open(DATA_FILE, 'w') as file:
        json.dump(data, file, indent=4)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Load existing data
parking_data = load_data()

# Function to preprocess the image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    return edged

# Function to detect and recognize number plates
def detect_number_plate(image):
    preprocessed = preprocess_image(image)
    contours, _ = cv2.findContours(preprocessed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        
        if 2 < aspect_ratio < 6:  # Filtering for rectangular shapes
            plate_image = image[y:y + h, x:x + w]
            plate_text = pytesseract.image_to_string(plate_image, config='--psm 8').strip()
            
            # Validate plate text: at least 3 alphabets and 4 numbers
            if plate_text and re.match(r'^(?=(.*[A-Za-z]){3,})(?=(.*\d){4,}).*$', plate_text):
                # Check if plate text is already in the parking data
                if plate_text in parking_data:
                    if parking_data[plate_text].get('entry_time'):
                        # Calculate parking duration
                        entry_time = datetime.strptime(parking_data[plate_text]['entry_time'], "%Y-%m-%d %H:%M:%S")
                        exit_time = datetime.now()
                        duration = exit_time - entry_time
                        print(f"Plate '{plate_text}' has been parked for {duration}.")
                        # Optionally calculate fee based on duration
                        # fee = calculate_fee(duration)
                        # print(f"Parking fee: ${fee:.2f}")
                        del parking_data[plate_text]  # Remove entry after exit
                        save_data(parking_data)  # Save updated data
                    else:
                        print(f"Error: No entry time found for plate '{plate_text}'.")
                else:
                    # Record entry time
                    parking_data[plate_text] = {
                        'entry_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    save_data(parking_data)
                    print(f"Plate '{plate_text}' entered at {parking_data[plate_text]['entry_time']}.")
                
                # Annotate the image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                return plate_text
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    plate_text = detect_number_plate(frame)
    
    # Display the annotated image (optional)
    cv2.imshow('Number Plate Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
