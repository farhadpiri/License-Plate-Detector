import os.path

import pandas as pd
import ntpath
import os
import cv2


def from_excel():
    DataFrame = pd.DataFrame(columns = ["plate_path","license_number"])
    data_store = "F:\edu\LicensePlateDetector\datasets\DataPreparation\image_store"
    plate_store = "F:\edu\LicensePlateDetector\datasets\DataPreparation\plate_store"
    df = pd.read_excel("data.xlsx")

    for idx, row in df.iterrows():
        if(idx < 0):
            continue
        path = row["image_path"]
        license_number = row["license_number"]
        try:
            if(license_number != "Unspecified"):
                head, tail = ntpath.split(path)
                new_path = os.path.join(data_store,tail)
                img = cv2.imread(new_path)
                plate_img = img[row["ymin"]:row["ymax"],row["xmin"]:row['xmax'],:]
                reshaped_img = cv2.resize(plate_img,(int(img.shape[1]* (48/img.shape[0])),48))
                new_path = os.path.join(plate_store,tail)
                cv2.imwrite(new_path,reshaped_img)
                new_row = {"plate_path":new_path, "license_number": license_number}
                DataFrame = DataFrame.append(new_row, ignore_index=True)
        except:
            print("could not detect plate")
    DataFrame.to_excel("plate_data.xlsx")

if(__name__=="__main__"):
    from_excel()