#### Quick example

Install mediapipe and opencv on your laptop and run the mediapipe_webcam.py file and watch the magic. Mediapipe's blaze model struggle's with different face angels. Which means it's hard if the user is not actively looking at the screen. 

### Possible fixes

1. Use HopeNet or Deep HopeNet lite.
2. Use combination of YOLO and mediapipe
3. Train a keras model from scratch on 300 L3 dataset or our own annotated dataset adn train our model.

fixes are dependent on the task itself, output type (image or video). How many frames we are talking about per second and many other factors. 

