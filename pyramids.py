import cv2
import numpy as np


# Build Gaussian image pyramid
def build_gaussian_pyramid(img, levels):
    float_img = np.ndarray(shape=img.shape, dtype="float")
    float_img[:] = img
    pyramid = [float_img]

    for i in range(levels-1):
        float_img = cv2.pyrDown(float_img)
        pyramid.append(float_img)

    return pyramid


# Build Laplacian image pyramid from Gaussian pyramid
def build_laplacian_pyramid(img, levels):
    gaussian_pyramid = build_gaussian_pyramid(img, levels)
    laplacian_pyramid = []

    for i in range(levels-1):
        upsampled = cv2.pyrUp(gaussian_pyramid[i+1])
        (height, width, depth) = upsampled.shape
        gaussian_pyramid[i] = cv2.resize(gaussian_pyramid[i], (height, width))
        diff = cv2.subtract(gaussian_pyramid[i],upsampled)
        laplacian_pyramid.append(diff)

    laplacian_pyramid.append(gaussian_pyramid[-1])

    return laplacian_pyramid


# Build video pyramid by building Laplacian pyramid for each frame
def build_video_pyramid(frames):
    lap_video = []

    for i, frame in enumerate(frames):
        pyramid = build_laplacian_pyramid(frame, 3)
        for j in range(3):
            if i == 0:
                lap_video.append(np.zeros((len(frames), pyramid[j].shape[0], pyramid[j].shape[1], 3)))
            lap_video[j][i] = pyramid[j]

    return lap_video


# Collapse video pyramid by collapsing each frame's Laplacian pyramid
def collapse_laplacian_video_pyramid(video, frame_ct):
    collapsed_video = []

    for i in range(frame_ct):
        prev_frame = video[-1][i]

        for level in range(len(video) - 1, 0, -1):
            pyr_up_frame = cv2.pyrUp(prev_frame)
            (height, width, depth) = pyr_up_frame.shape
            prev_level_frame = video[level - 1][i]
            prev_level_frame = cv2.resize(prev_level_frame, (height, width))
            prev_frame = pyr_up_frame + prev_level_frame

        # Normalize pixel values
        min_val = min(0.0, prev_frame.min())
        prev_frame = prev_frame + min_val
        max_val = max(1.0, prev_frame.max())
        prev_frame = prev_frame / max_val
        prev_frame = prev_frame * 255

        prev_frame = cv2.convertScaleAbs(prev_frame)
        collapsed_video.append(prev_frame)

    return collapsed_video
