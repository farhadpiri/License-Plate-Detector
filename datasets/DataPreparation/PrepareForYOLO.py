import os.path
import random
import shutil

import cv2
import numpy as np
import pandas as pd
import ntpath

class PrepareForYOLO:
    def __init__(self):
        self.root = "/content/drive/MyDrive/projects/LicensePlate/output_folder_2/YoloDataset_2/"
        self.train_folder = os.path.join(os.getcwd(),"YoloDataset/train")
        if(not os.path.isdir(self.train_folder)):
            os.mkdir(self.train_folder)
        self.train_images_and_labels = os.path.join(os.getcwd(),"YoloDataset/train.txt")
        with open(self.train_images_and_labels, "w"):
            pass
        self.val_folder = os.path.join(os.getcwd(),"YoloDataset/val")
        if(not os.path.isdir(self.val_folder)):
            os.mkdir(self.val_folder)
        self.val_images_and_labels = os.path.join(os.getcwd(), "YoloDataset/val.txt")
        with open(self.val_images_and_labels, "w"):
            pass
        self.test_folder = os.path.join(os.getcwd(),"YoloDataset/test")
        if(not os.path.isdir(self.test_folder)):
            os.mkdir(self.test_folder)
        self.test_images_and_labels = os.path.join(os.getcwd(), "YoloDataset/test.txt")
        with open(self.val_images_and_labels, "w"):
            pass

    def copy_image(self,path):
        pass

    def create_txt(self):
        pass

    def create_yaml(self):
        pass

    def create_YOLO_dataset(self):
        df = self.read_dataframe("data.xlsx")

        for index, row in df.iterrows():
            path = row["image_path"]

            head, tail = ntpath.split(path)
            splited_tail = tail.split('.')
            name = splited_tail[0]
            format = splited_tail[-1]
            print(tail)
            src_img_path = os.path.join(os.getcwd(),"image_store",tail)

            r = random.randint(0,10)
            if(r < 8):
                train_image_dir = os.path.join(self.train_folder,"images")
                train_label_dir = os.path.join(self.train_folder,"labels")
                if(not os.path.isdir(train_image_dir)):
                    os.mkdir(train_image_dir)
                if (not os.path.isdir(train_label_dir)):
                    os.mkdir(train_label_dir)
                dst_img_path = os.path.join(train_image_dir,tail)
                dst_txt_path = os.path.join(train_label_dir,f"{name}.txt")
                with open(self.train_images_and_labels, "a+") as f:
                    f.write(self.root+"train/"+"images/"+tail + "\n")
                    f.write(self.root+"train/"+"labels/"+f"{name}.txt"+ "\n")
            elif(r < 9):
                val_image_dir = os.path.join(self.val_folder,"images")
                val_label_dir = os.path.join(self.val_folder,"labels")
                if(not os.path.isdir(val_image_dir)):
                    os.mkdir(val_image_dir)
                if (not os.path.isdir(val_label_dir)):
                    os.mkdir(val_label_dir)
                dst_img_path = os.path.join(val_image_dir,tail)
                dst_txt_path = os.path.join(val_label_dir,f"{name}.txt")
                with open(self.val_images_and_labels, "a+") as f:
                    f.write(self.root+"val/"+"images/"+tail + "\n")
                    f.write(self.root+"val/"+"labels/"+f"{name}.txt"+ "\n")
            else:
                test_image_dir = os.path.join(self.test_folder,"images")
                test_label_dir = os.path.join(self.test_folder,"labels")
                if(not os.path.isdir(test_image_dir)):
                    os.mkdir(test_image_dir)
                if (not os.path.isdir(test_label_dir)):
                    os.mkdir(test_label_dir)
                dst_img_path = os.path.join(test_image_dir,tail)
                dst_txt_path = os.path.join(test_label_dir,f"{name}.txt")
                with open(self.test_images_and_labels, "a+") as f:
                    f.write(self.root+"test/"+"images/"+tail + "\n")
                    f.write(self.root+"test/"+"labels/"+f"{name}.txt"+ "\n")

            [x_c, y_c, w, h] = self.prepare_output(row, src_img_path,dst_img_path,add_label=False)

            with open(dst_txt_path, "a+") as f:
                line = f"0 {x_c} {y_c} {w} {h}\n"
                f.write(line)



    def read_dataframe(self, path):
        return pd.read_excel(path)

    def prepare_output(self, row, src_img_path,dst_img_path,img_size=720,add_label=False):


        img = cv2.imread(src_img_path)

        [H, W, C] = img.shape
        image_scale = max(H/img_size,W/img_size)
        img_resized = cv2.resize(img,(int(W/image_scale),int(H/image_scale)))

        width_buttom = int((img_size - img_resized.shape[0])/2)
        width_up = int((img_size - img_resized.shape[0]+1)/2)
        width_left = int((img_size - img_resized.shape[1])/2)
        width_right = int((img_size - img_resized.shape[1]+1)/2)

        img_resized_pdded = np.pad(img_resized, ((width_buttom, width_up), (width_left, width_right), (0, 0)), 'constant',constant_values=0)

        xmin = int(int(row["xmin"])/image_scale + width_left)
        xmax = int(int(row["xmax"])/image_scale + width_left)
        ymax = int(int(row["ymax"])/image_scale + width_up)
        ymin = int(int(row["ymin"])/image_scale + width_up)


        x_c = int(xmin + (xmax - xmin) / 2) / img_size
        y_c = int(ymin + (ymax - ymin) / 2) / img_size
        w = (xmax - xmin) / img_size
        h = (ymax - ymin) / img_size

        if(add_label):
            img_resized_pdded= cv2.rectangle(img_resized_pdded,(int(x_c-w*img_size/2),int(y_c-h*img_size/2)),(int(x_c+w*img_size/2),int(y_c+h*img_size/2)),(0,0,255),2)
        cv2.imwrite(dst_img_path,img_resized_pdded)


        return [x_c, y_c, w, h]


if(__name__=="__main__"):
    prepare_dataset = PrepareForYOLO()
    prepare_dataset.create_YOLO_dataset()