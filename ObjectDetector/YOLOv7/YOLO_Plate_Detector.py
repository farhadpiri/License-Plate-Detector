import time
import sys
sys.path.insert(0, './ObjectDetector/YOLOv7/yolov7')

import numpy as np
from Config import Addresses
from Config.YOLOConfig import DetectionConfig
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from pathlib import Path

from ObjectDetector.YOLOv7.yolov7.models.experimental import attempt_load
from ObjectDetector.YOLOv7.yolov7.utils.datasets import LoadStreams, LoadImages
from ObjectDetector.YOLOv7.yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from ObjectDetector.YOLOv7.yolov7.utils.plots import plot_one_box
from ObjectDetector.YOLOv7.yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from ObjectDetector.YOLOv7.yolov7.utils.datasets import letterbox


class YOLO_Plate_Detector:
    def __init__(self):
        self.weights = DetectionConfig.weights
        self.view_img = DetectionConfig.view_img
        self.save_txt = DetectionConfig.save_txt
        self.nosave = DetectionConfig.no_save
        self.img_size = DetectionConfig.imgsz
        self.no_trace = DetectionConfig.no_trace
        self.device = ''
        self.project = 'runs/detect'
        self.name = 'exp'
        self.exist_ok = False
        self.augment = False
        self.conf_thres = DetectionConfig.conf_thres
        self.iou_thres = DetectionConfig.iou_thres
        self.classes = [0,1]
        self.agnostic_nms = True
        self.path = Addresses.YOLO_check_point_address
        self.stride = DetectionConfig.stride


    def detect(self,image):

        weights, view_img, save_txt, imgsz, trace = self.weights, self.view_img, self.save_txt, self.img_size, not self.no_trace

        save_img = not self.nosave  # save inference images

        # Directories
        save_dir = Path(increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(self.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        # model = torch.load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if trace:
            model = TracedModel(model, device, self.img_size)

        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        t0 = time.time()
        im0s = image
        # Padded resize
        img = letterbox(image, imgsz, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=self.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=self.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                                    agnostic=self.agnostic_nms)
        t3 = time_synchronized()


        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
                p, s, im0 = self.path , '', im0s

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  + "saved.png" # img.jpg
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(3000)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")

        print(f'Done. ({time.time() - t0:.3f}s)')

        return pred