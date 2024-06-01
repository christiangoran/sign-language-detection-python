# Sign Language Detection using Python and Tensorflow

## Tech Stack

- OpenCV using webcam to capture dataset images
- Mediapipe provides a suite of libraries that can be used. We will use:
  - Landmark for hand landmark detection
    [Mediapipe Hand Landmark Detection website](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)

## Encountered bugs & problems

- The webcam doesnt seem to be accessed correctly
  Resolution: Entered a webcam index that starts with '0' and goes up until it works

- A but is causing some sort of effect to throw in party balloons in some of the webcam footage which ruins that perticular letter and its corresponding images. I will have to re-make 'k', 'u', 'v' and 'w'

  - I can't seem to be able to find the source of this problem. The balloons keep reappering on these specific letters everytime I try to collect images using OpenCV and my webcam.
  - I will make a model where these letters are not part of it.

- The letter 'z' has a movement required to identify the letter. Without the movement it is just the letter 'd'

## Recognition and sources

- Great tutorial by Ivan Goncharov [Youtube Channel](https://www.youtube.com/watch?v=a99p_fAr6e4&list=PL0FM467k5KSyt5o3ro2fyQGt-6zRkHXRv)
