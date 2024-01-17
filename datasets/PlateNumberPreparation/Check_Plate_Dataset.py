import cv2
import pandas as pd


def read_plates():
    df = pd.read_excel("downloaded.xlsx")

    for idx, row in df.iterrows():
        if(idx % 100 == 0):
            path = row["plate_path"]
            img = cv2.imread(path)
            license_number = row['license_number']

            print(license_number)
            cv2.imshow("1",img)
            cv2.moveWindow("1", 20, 20)
            cv2.waitKey()
            cv2.destroyAllWindows()

if(__name__=="__main__"):
    read_plates()

