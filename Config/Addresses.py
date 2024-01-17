import os
root_address = "/mnt/f/edu/LicensePlateDetector/"
datasets_root_address = os.path.join(root_address,"datasets")

intigrated_dataset_address = os.path.join(datasets_root_address,"integrated")
OCR_check_point_address = os.path.join(os.getcwd(),"Assets/OCR.ckpt")
YOLO_check_point_address = os.path.join("best.pt")