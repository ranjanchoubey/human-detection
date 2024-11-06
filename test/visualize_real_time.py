import numpy as np 
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import imutils
import pafy, joblib, cv2
from skimage import color
import Sliding as sd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from skimage.feature import hog

size = (64,128)
step_size = (10,10)
downscale = 1.25

# Load the model using PyTorch
class HOGModel(nn.Module):
    def __init__(self):
        super(HOGModel, self).__init__()
        self.fc = nn.Linear(3780, 1)  # 3780 is the length of HOG feature vector

    def forward(self, x):
        x = self.fc(x)
        return x

model = HOGModel()
model.load_state_dict(torch.load('models/models.dat'))
model.eval()
model = model.cuda()

# Load the video using OpenCV
video_url = 'vido.mp4'
cap = cv2.VideoCapture(video_url)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    image = cv2.resize(frame, (512, 512))
    detections = []
    scale = 0
    for im_scaled in pyramid_gaussian(image, downscale=downscale):
        if im_scaled.shape[0] < size[1] or im_scaled.shape[1] < size[0]:
            break
        for (x, y, window) in sd.sliding_window(im_scaled, size, step_size):
            if window.shape[0] != size[1] or window.shape[1] != size[0]:
                continue
            window = color.rgb2gray(window)
            fd = hog(window, orientations=9, pixels_per_cell=(8, 8), visualize=False, cells_per_block=(3, 3))
            fd = torch.tensor(fd, dtype=torch.float32).cuda().unsqueeze(0)
            with torch.no_grad():
                pred = model(fd)
                if pred.item() > 0.5:
                    detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), pred.item(), 
                    int(size[0] * (downscale**scale)),
                    int(size[1] * (downscale**scale))))
        scale += 1
    clone = image.copy()
    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    sc = [score for (x, y, score, w, h) in detections]
    sc = np.array(sc)
    pick = non_max_suppression(rects, probs=sc, overlapThresh=0.3)

    for (x1, y1, x2, y2) in pick:
        cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(clone, 'Human : {:.2f}'.format(np.max(sc)), (x1-2, y1-2), 1, 1, (0, 122, 12), 1)
    cv2.imshow('Human Detection', clone)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()