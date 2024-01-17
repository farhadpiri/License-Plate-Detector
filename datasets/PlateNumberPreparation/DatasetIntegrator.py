import cv2
import pandas as pd
import ntpath

def move_data(excel_path):
        df_old = pd.read_excel(excel_path)
        df_new = pd.DataFrame(columns=["plate_path", "license_number"])

        for idx,row in df_old.iterrows():
            path = row["plate_path"]
            label = str(row["license_number"])
            head, tail = ntpath.split(path)
            df_new = df_new.append({"plate_path": "all_data/" + tail, "license_number":label}, ignore_index=True)
        df_new.to_excel("all_data.xlsx")

if(__name__=="__main__"):
    move_data("downloaded.xlsx")


