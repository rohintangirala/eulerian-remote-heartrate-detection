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
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Converts to gray image
        roi_frame = img  # initial frame is stored in roi_frame

        # Detect face
        if len(video_frames) == 0:  # When the first frame is passed
            face_rects = faceCascade.detectMultiScale(gray, 1.3, 5)  # Returns the location rectangle for face detection

        # Select ROI
        if len(face_rects) > 0:  # If len(face_rects) > 0 that represents that face is found
            for (x, y, w, h) in face_rects:  # x, y denotes the top left most point of the image and w, h is the
                # width and height of rectangle
                roi_frame = img[y:y + h, x:x + w]  # region of interest
            if roi_frame.size != img.size:  # region of interest is subpart of whole image
                roi_frame = cv2.resize(roi_frame, (500, 500))

                # print("ROI_Frame: %s",{type(roi_frame), np.shape(roi_frame)});
                # cv2.imshow('frame',roi_frame)
                # k = cv2.waitKey(1000)

                frame = np.ndarray(shape=roi_frame.shape, dtype="float")
                frame[:] = roi_frame * (1. / 255)

                # cv2.imshow('scaled',frame)
                # k = cv2.waitKey(0)
                # if (k == 27):
                #     cv2.destroyAllWindows()

                video_frames.append(frame)
                # break

    frame_ct = len(video_frames)
    cap.release()
    print(f"Video_frames-length: {frame_ct}, fps: {fps}")
    return video_frames, frame_ct, fps
