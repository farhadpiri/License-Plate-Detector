import cv2
import pandas as pd

class CheckDataset:
    def __init__(self):
        self.DataFrame = None

    def read_excel(self,name):
        self.DataFrame =  pd.read_excel(name)

    def read_data(self,step):
        df = self.DataFrame.reset_index()  # make sure indexes pair with number of rows

        for index, row in df.iterrows():
            if(index % step != 0):
                continue
            image_path = row["image_path"]
            xmin = int(row["xmin"])
            xmax = int(row["xmax"])
            ymin = int(row["ymin"])
            ymax = int(row["ymax"])

            print(f"index:{index}, number:{row['license_number']}")
            img = cv2.imread(image_path)
            img = cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,0,0),3)

            cv2.imshow("1",img)
            cv2.moveWindow("1", 40, 30)
            cv2.waitKey()
            cv2.destroyAllWindows()


if(__name__=="__main__"):
    check = CheckDataset()
    check.read_excel("data.xlsx")
    check.read_data(step=100)
