import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt0.xml")


# Read in and simultaneously preprocess video
def read_video(path):
    cap = cv2.VideoCapture(path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_frames = []
    face_rects = ()

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        roi_frame = img

        # Detect face
        if len(video_frames) == 0:
            face_rects = faceCascade.detectMultiScale(gray, 1.3, 5)

        # Select ROI
        if len(face_rects) > 0:
            for (x, y, w, h) in face_rects:
                roi_frame = img[y:y + h, x:x + w]
            if roi_frame.size != img.size:
                roi_frame = cv2.resize(roi_frame, (500, 500))
                frame = np.ndarray(shape=roi_frame.shape, dtype="float")
                frame[:] = roi_frame * (1. / 255)
                video_frames.append(frame)

    frame_ct = len(video_frames)
    cap.release()

    return video_frames, frame_ct, fps
