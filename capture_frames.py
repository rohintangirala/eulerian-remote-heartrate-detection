import cv2
import numpy as np
import torch
from torch import nn
from models import LinkNet34
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image, ImageFilter
import time
import sys


class CaptureFrames():

    def __init__(self, fps=None, video_frames=[], source=None):  # show_mask=False
        self.stop = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = LinkNet34()
        self.model.load_state_dict(torch.load('linknet.pth', map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        # self.show_mask = show_mask
        self.video_frames = video_frames
        self.fps = fps

    def __call__(self, source):
        self.capture_frames(source)

    def capture_frames(self, source):

        img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        count = 0
        camera = cv2.VideoCapture(source)
        fps = int(camera.get(cv2.CAP_PROP_FPS))
        # time.sleep(1)
        self.model.eval()
        (grabbed, frame) = camera.read()

        time_1 = time.time()

        while camera.isOpened():

            (grabbed, orig) = camera.read()
            if not grabbed:
                continue

            shape = orig.shape[0:2]
            frame = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (256, 256), cv2.INTER_LINEAR)

            k = cv2.waitKey(1)
            if k != -1:
                self.terminate(camera)
                break

            a = img_transform(Image.fromarray(frame))
            a = a.unsqueeze(0)
            imgs = Variable(a.to(dtype=torch.float, device=self.device))
            pred = self.model(imgs)

            pred = torch.nn.functional.interpolate(pred, size=[shape[0], shape[1]])
            mask = pred.data.cpu().numpy()
            mask = mask.squeeze()

            mask = mask > 0.8
            orig[mask == 0] = 0
            count+=1
            if(count%10 ==0):
                print(f"Frame Appended: {count}")
            self.video_frames.append(orig)

        self.terminate(camera)
        frame_ct = len(self.video_frames)
        print(f"Video_frames-length: {frame_ct}, fps: {fps}")
        return self.video_frames, frame_ct, fps

    def terminate(self, camera):
        cv2.destroyAllWindows()
        camera.release()



    # return self.video_frames, frame_ct, fps