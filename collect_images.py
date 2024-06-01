import os
import cv2
import time
import uuid

# Creating a directory to store all the dataset images
DATA_DIR = 'dataset'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of hand signs I want to collect images for
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
          'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
number_of_images = 100

# Function to wait for spacebar press to continue


def wait_for_space():
    print("Press the spacebar to continue to the next label...")
    while True:
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break


# Try different indices to find the correct webcam
webcam_index = 0  # Start with 0, then try 1, 2, etc. if needed

# I will use the OpenCV library to capture images from the webcam
for label in labels:
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {webcam_index}")
        cap.release()
        continue

    print('Collecting images for {}'.format(label))
    time.sleep(2)
    for imgnum in range(number_of_images):
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        imgname = os.path.join(label_dir, label + '.' +
                               '{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imgname, frame)
        # Display how many images have been collected out of the 100 for each label
        cv2.putText(frame, f'Label: {label} | Image: {imgnum+1}/{number_of_images}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        time.sleep(0.5)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Wait for spacebar to continue to the next label
    wait_for_space()
