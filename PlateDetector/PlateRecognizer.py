import os
import torchvision.transforms as transforms
import cv2
import torch
from Config import Addresses
from PlateDetector.Models.Deepayan.Model.Models import CRNN
from PIL import Image

from PlateDetector.Models.Deepayan.utils.utils import OCRLabelConverter


class PlateRecognizerArgs:
    def __init__(self):
        self.imgH = 48
        self.nHidden = 256
        self.nChannels = 1
        self.alphabet = """Only thewigsofrcvdampbkuq.$A-210xT5'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%"""
        self.nClasses = len(self.alphabet)



class PlateRecognizer():
    def __init__(self):
        self.prepare_model()

    def prepare_model(self):
        args = PlateRecognizerArgs()
        model = CRNN(args)
        self.converter = OCRLabelConverter(args.alphabet)
        if (torch.cuda.is_available()):
            model = model.cuda()
        check_point_address = Addresses.OCR_check_point_address
        if os.path.isfile(check_point_address):
            print('Loading model %s' % check_point_address)
            checkpoint = torch.load(check_point_address, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])
            self.model = model

    def predict(self, plate_image):
        plate_image_rs = cv2.resize(plate_image, (int(plate_image.shape[0] * (plate_image.shape[1]/48)), 48))
        im_pil = Image.fromarray(plate_image_rs)

        transform_list = [transforms.Grayscale(1),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5,), (0.5,))]

        transform = transforms.Compose(transform_list)

        # Convert the image to PyTorch tensor
        tensor = transform(im_pil)
        tensor = tensor[None, :, :, :]
        logits = self.model(tensor).transpose(1, 0)

        logits = torch.nn.functional.log_softmax(logits, 2)
        logits = logits.contiguous().cpu()
        T, B, H = logits.size()
        pred_sizes = torch.LongTensor([T for i in range(B)])
        probs, pos = logits.max(2)
        pos = pos.transpose(1, 0).contiguous().view(-1)
        sim_preds = self.converter.decode(pos.data, pred_sizes.data, raw=False)
        return sim_preds

    def recognize(self,plates_dets,image):
        text_estimations = []
        for plate in plates_dets:
            for *xyxy, conf, cls in reversed(plate):
                plate_image = image[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]
                text_estimation = self.predict(plate_image)
                text_estimations.append(text_estimation)
        return text_estimations



