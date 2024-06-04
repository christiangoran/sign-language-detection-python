# Sign Language Detection Using VGG16

This project aims to detect and classify hand gestures representing the alphabet in American Sign Language (ASL) using a pre-trained VGG16 model. The project involves collecting hand gesture images, preparing a dataset, training the VGG16 model, and running the model to recognize gestures in real-time via a webcam.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training the Model](#training-the-model)
- [Running the Model](#running-the-model)
- [Contributing](#contributing)
- [License](#license)

## Overview

The project consists of three main stages:

1. **Image Collection**: Capture images of hand gestures for different ASL alphabets.
2. **Model Training**: Train a VGG16 model using the collected dataset.
3. **Real-Time Gesture Recognition**: Use the trained model to recognize hand gestures from a webcam feed.

## Installation

Clone the repository and install the required dependencies.

```sh
git clone https://github.com/yourusername/sign-language-detection.git
cd sign-language-detection

# Set up a virtual environment (optional but recommended)
python -m venv myenv
source myenv/bin/activate  # On Windows use `myenv\Scriptsctivate`

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

Collect images for each ASL alphabet and prepare the dataset.

1. **Image Collection**:
   - Run the `collect_images.py` script to capture images of hand gestures for each alphabet.
   - Adjust the `folder` variable to specify which letter you are collecting images for.

```sh
python collect_images.py
```

2. **Create Dataset**:
   - Run the `create_dataset.py` script to process the collected images and save them into a dataset.

```sh
python create_dataset.py
```

## Training the Model

Train the VGG16 model using the prepared dataset.

```sh
python train_vgg16.py
```

## Running the Model

Use the trained model to recognize hand gestures in real-time via a webcam.

```sh
python run_model.py
```

## Contributing

Contributions are welcome! Please create an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

### Description of Each Script

#### `collect_images.py`

This script captures images of hand gestures for different ASL alphabets using a webcam and saves them to the specified directory.

#### `create_dataset.py`

Processes the collected images, extracts hand landmarks, and prepares a dataset for training.

#### `train_vgg16.py`

Trains a VGG16 model on the prepared dataset. The model is then saved for later use in real-time gesture recognition.

#### `run_model.py`

Uses the trained VGG16 model to recognize hand gestures in real-time from a webcam feed and displays the predicted alphabet on the screen.

## Tech Stack

- OpenCV using webcam to capture dataset images
- Mediapipe provides a suite of libraries that can be used. We will use:
  - Landmark for hand landmark detection
    [Mediapipe Hand Landmark Detection website](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
    [Github repository](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md)

## Encountered bugs & problems

- The webcam doesnt seem to be accessed correctly
  Resolution: Entered a webcam index that starts with '0' and goes up until it works
  '

- The model is not working properly. It is identifying the wrong letters.
  - I tried using another pre trained model (Randomforestclassifier) but this didn't make any difference.
  - I re trained the model, this actually makes the model predict different kind of letters when testing.
  - I ended up re-writing my entire image capturing script, capping the size at 224x224 pixels of each image so I could run them through the VGG16 pre trained model.

## Recognition and sources

- Great tutorial by Ivan Goncharov [Youtube Channel](https://www.youtube.com/watch?v=a99p_fAr6e4&list=PL0FM467k5KSyt5o3ro2fyQGt-6zRkHXRv)
