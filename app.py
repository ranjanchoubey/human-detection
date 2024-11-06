import numpy as np
import cv2
import joblib
from skimage.feature import hog
from skimage import color
from skimage.transform import pyramid_gaussian
from sklearn.svm import LinearSVC
from imutils.object_detection import non_max_suppression
import gradio as gr

# Define the sliding window function
def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[0] - window_size[1], step_size[1]):
        for x in range(0, image.shape[1] - window_size[0], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

# Load the trained SVM model
model_path = 'models/models.dat'  # Update with your model path
model = joblib.load(model_path)

# Define the human detection function
def detect_human(image):
    # Preprocess the image
    image = cv2.resize(image, (400, 256))
    size = (64, 128)
    step_size = (9, 9)
    downscale = 1.25

    # List to store detections
    detections = []

    # Initialize scale
    scale = 0

    # Apply sliding window and pyramid
    for im_scaled in pyramid_gaussian(image, downscale=downscale):
        if im_scaled.shape[0] < size[1] or im_scaled.shape[1] < size[0]:
            break
        for (x, y, window) in sliding_window(im_scaled, size, step_size):
            if window.shape[0] != size[1] or window.shape[1] != size[0]:
                continue
            window = color.rgb2gray(window)
            fd = hog(window, orientations=9, pixels_per_cell=(8, 8), visualize=False, cells_per_block=(3, 3))
            fd = fd.reshape(1, -1)
            pred = model.predict(fd)
            if pred == 1 and model.decision_function(fd) > 0.5:
                detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), model.decision_function(fd), 
                                   int(size[0] * (downscale**scale)), int(size[1] * (downscale**scale))))
        scale += 1

    # Copy the original image
    clone = image.copy()

    # Convert detections to numpy array
    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    sc = [score[0] for (x, y, score, w, h) in detections]
    sc = np.array(sc)

    # Apply non-max suppression
    pick = non_max_suppression(rects, probs=sc, overlapThresh=0.3)

    # Draw rectangles on the image
    for (x1, y1, x2, y2) in pick:
        cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(clone, 'human', (x1-2, y1-2), 1, 0.75, (121, 12, 34), 1)

    # Return the processed image
    return clone

# Create a Gradio interface with specified width and height
iface = gr.Interface(
    fn=detect_human, 
    inputs=gr.Image(type="numpy", image_mode="RGB", label="Input Image", width=400, height=256),  # Set the input image size
    outputs=gr.Image(type="numpy", image_mode="RGB", label="Human Detected Image", width=400, height=256),  # Set the output image size
    title="Human Detection with HOG and SVM",
    description="Upload an image to detect humans using Histogram of Oriented Gradients (HOG) and Support Vector Machine (SVM)."
)

# Launch the Gradio app
iface.launch()