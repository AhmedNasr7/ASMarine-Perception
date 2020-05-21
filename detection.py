
import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import cv2
import time


class Detector:

    def __init__(self, opt, source_img, cfg, weights, save_img=False):
        #soource_img, weilghts_path, 
        self.source_img = source_img
        self.img_size = 416
        self.cfg = cfg
        self.device = torch_utils.select_device(device='0')
        self.weights = weights
        self.save_img = save_img

        self.model = Darknet(self.cfg, self.img_size)   # Initialize model
        self.opt = opt
        
        # Load weights
        attempt_download(self.weights) 
        
        if self.weights.endswith('.pt'):  # pytorch format
            self.model.load_state_dict(torch.load(self.weights, map_location=self.device)['model']) ## model loading is here
        else:  # darknet format
            load_darknet_weights(self.model, self.weights) ## or here???
        
        self.model.to(self.device).eval() ## evaluation mode

        if save_img:
            self.save_path = '/content/output/' # to the current dir

        self.half = False

        # Half precision 

        self.half = self.half and self.device.type != 'cpu'  # half precision only supported on CUDA
        #if self.half:
        #self.model.half()

        self.agnostic_nms = True

        # Get names and colors
        self.names = '/content/robosub.names'
        self.names = load_classes(self.names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        self.classes = self.opt.classes

        self.half = self.half and self.device.type != 'cpu'  # half precision only supported on CUDA

        self.dataset = LoadImages(self.source_img, img_size=self.img_size, half=self.half)

        self.detections= []
        self.confidence = -1 # initially



    def detect(self, conf_thres = 0.3, iou_thres = 0.5):


        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
  

        #image = cv2.imread(self.source_img) ## to be changed
        t0 = time.time()
        for path, img, im0s, vid_cap in self.dataset:

            t = time.time()

            # Get detections
            img = torch.from_numpy(img).to(self.device)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = self.model(img)[0]

            if self.half:
                pred = pred.float()

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.opt.classes, agnostic=self.agnostic_nms)

          
            # Process detections
            for i, det in enumerate(pred):  # detections per image
               
                p, s, im0 = path, '', im0s

                save_path = str(Path(self.save_path) / Path(p).name)
                s += '%gx%g ' % img.shape[2:]  # print string
                
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in det:

                        detection = [int(cls.item()), xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]
                        self.detections.append(detection)
                        self.confidence = conf.item()

                        if self.save_img:  # Add bbox to image
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)])

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, time.time() - t))

                # Save results (image with detections)
                if self.save_img:
                    if self.dataset.mode == 'images':
                        cv2.imwrite(save_path, im0)
                    

        if self.save_img:
            print('Results saved to %s' % self.save_path)
            print('Done. (%.3fs)' % (time.time() - t0))

        return self.detections, self.confidence




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    source_img = '/content/1.jpg'
    cfg = '/content/yolov3-spp.cfg'
    weights = './best-AUV-gate.weights'
    opt = parser.parse_args()
    detector = Detector(opt, source_img, cfg, weights, save_img=True)
    detections, confidence = detector.detect(0.99, 0.1)
    print(detections)
    print("confidence: ", confidence)