import os
import pandas as pd
import cv2
import glob
import ntpath
import numpy as np

# char_map1 = {"0":, "1":, "2":, "3":, "4":, "5":, "6":, "7":, "8":, "9":,
#             "10":, "11":, "12":, "13":, "14":, "15":, "16":, "17":, "18":, "19":,
#             "20":, "21":, "22":, "23":, "24":, "25":, "26":, "27":, "28":, "29":,
#             "30":, "31":, "32":, "33":, "34":, "35":}
write_path = os.path.join(os.getcwd() , "downloaded")

def Add_downloaded1(name):
    char_map1 = {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "8": "8",
                 "9": "9", "10": "A", "11": "B", "12": "C", "13": "D", "14": "E", "15": "F", "16": "G", "17": "H",
                 "18": "I",
                 "19": "J", "20": "K", "21": "L", "22": "M", "23": "N", "24": "O", "25": "P", "26": "R",
                 "27": "S", "28": "T", "29": "U", "30": "V", "31": "W", "32": "X", "33": "Y", "34": "Z"}

    df = pd.DataFrame(columns=["plate_path", "license_number"])

    address = "F:\edu\LicensePlateDetector\datasets\\08\\01\com\\*.jpg"
    # write_path = os.path.join(os.getcwd() , "01")
    images_path = glob.glob(address)

    for img_path in images_path:
            img = cv2.imread(img_path)
            img = img[100:500,:,:]
            img = cv2.resize(img,(int(img.shape[1] * (48/img.shape[0])),48))
            lable_path = img_path.replace(".jpg",".txt")
            string = ""
            with open(lable_path, "r") as f:
                lines = f.readlines()
                chars = []
                xs = []
                for line in lines:
                    chars.append(char_map1[line.split(' ')[0]])
                    xs.append(line.split(' ')[1])
                xs_arg_sort = np.argsort(np.asarray(xs))
                for i in xs_arg_sort:
                    string +=chars[i]
                a=0
            print(string)
            head, tail = ntpath.split(img_path)
            new_path = os.path.join(write_path,tail)
            df = df.append({"license_number":string, "plate_path":new_path}, ignore_index=True)
            cv2.imwrite(new_path,img)
    df.to_excel(f"{name}.xlsx")
    return df

def Add_downloaded2(name):
    char_map1 = {"1": "0", "2": "1", "3": "2", "4": "3", "5": "4", "6": "5", "7": "6", "8": "7", "9": "8",
                 "10": "9", "11": "A", "12": "B", "13": "C", "14": "D", "15": "E", "17": "F", "18": "G", "19": "H",
                 "20": "I",
                 "21": "J", "22": "K", "23": "L", "24": "M", "25": "N","26":"O", "27": "P", "28": "Q", "29": "R",
                 "30": "S", "31": "T", "32": "U", "33": "V", "34": "W", "35": "X", "36": "Y", "37": "Z"}

    df = pd.DataFrame(columns=["plate_path", "license_number"])

    address = "F:\edu\LicensePlateDetector\datasets\\08\\02\com\\*.jpg"
    # write_path = os.path.join(os.getcwd() , "02")
    images_path = glob.glob(address)

    for img_path in images_path:
        try:
            img = cv2.imread(img_path)
            # img = img[100:500,:,:]
            img = cv2.resize(img,(int(img.shape[1] * (48/img.shape[0])),48))
            lable_path = img_path.replace(".jpg",".txt")
            string = ""
            with open(lable_path, "r") as f:
                lines = f.readlines()
                chars = []
                xs = []
                for line in lines:
                    cls = line.split(' ')[0]
                    if(cls == "16"):
                        continue
                    chars.append(char_map1[cls])
                    xs.append(line.split(' ')[1])
                xs_arg_sort = np.argsort(np.asarray(xs))
                for i in xs_arg_sort:
                    string +=chars[i]
                a=0
            print(string)
            head, tail = ntpath.split(img_path)
            new_path = os.path.join(write_path,tail)
            df = df.append({"license_number":string, "plate_path":new_path}, ignore_index=True)
            cv2.imwrite(new_path,img)
        except:
            print(img_path)
    df.to_excel(f"{name}.xlsx")
    return df

def Add_downloaded3(name):
    char_map1 = {"1": "0", "2": "1", "3": "2", "4": "3", "5": "4", "6": "5", "7": "6", "8": "7", "9": "8",
                 "10": "9", "11": "A", "12": "B", "13": "C", "14": "D", "15": "E", "17": "F", "18": "G", "19": "H",
                 "20": "I", "21": "J", "22": "K", "23": "L", "24": "M", "25": "N", "26":"O", "27": "P", "28": "Q", "29": "R",
                 "30": "S", "31": "T", "32": "U", "33": "V", "34": "W", "35": "X", "36": "Y", "37": "Z"}

    df = pd.DataFrame(columns=["plate_path", "license_number"])

    address = "F:\edu\LicensePlateDetector\datasets\\08\\03\com\\*.jpg"
    # write_path = os.path.join(os.getcwd() , "03")
    images_path = glob.glob(address)

    for img_path in images_path:
        try:
            img = cv2.imread(img_path)
            # img = img[100:500,:,:]
            img = cv2.resize(img,(int(img.shape[1] * (48/img.shape[0])),48))
            lable_path = img_path.replace(".jpg",".txt")
            string = ""
            with open(lable_path, "r") as f:
                lines = f.readlines()
                chars = []
                xs = []
                for line in lines:
                    cls = line.split(' ')[0]
                    if(cls == "16"):
                        continue
                    chars.append(char_map1[cls])
                    xs.append(line.split(' ')[1])
                xs_arg_sort = np.argsort(np.asarray(xs))
                for i in xs_arg_sort:
                    string +=chars[i]
                a=0
            print(string)
            head, tail = ntpath.split(img_path)
            new_path = os.path.join(write_path,tail)
            df = df.append({"license_number":string, "plate_path":new_path}, ignore_index=True)
            cv2.imwrite(new_path,img)
        except:
            print(img_path)
    df.to_excel(f"{name}.xlsx")
    return df

def Add_downloaded4(name):
    char_map1 = {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "8": "8",
                 "9": "9", "10": "A", "11": "B", "12": "C", "13": "D", "14": "E", "15": "F", "16": "G", "17": "H",
                 "18": "I",
                 "19": "J", "20": "K", "21": "L", "22": "M", "23": "N", "24": "O", "25": "P", "26": "Q",
                 "27": "R", "28": "S", "29": "T", "30": "U", "31": "V", "32": "W", "33": "X", "34": "Y", "35":"Z"}

    df = pd.DataFrame(columns=["plate_path", "license_number"])

    address = "F:\edu\LicensePlateDetector\datasets\\08\\04\com\\*.jpg"
    # write_path = os.path.join(os.getcwd() , "04")
    images_path = glob.glob(address)

    for img_path in images_path:
        try:
            img = cv2.imread(img_path)
            # img = img[100:500,:,:]
            img = cv2.resize(img,(int(img.shape[1] * (48/img.shape[0])),48))
            lable_path = img_path.replace(".jpg",".txt")
            string = ""
            with open(lable_path, "r") as f:
                lines = f.readlines()
                chars = []
                xs = []
                for line in lines:
                    cls = line.split(' ')[0]
                    chars.append(char_map1[cls])
                    xs.append(line.split(' ')[1])
                xs_arg_sort = np.argsort(np.asarray(xs))
                for i in xs_arg_sort:
                    string +=chars[i]
                a=0
            print(string)
            head, tail = ntpath.split(img_path)
            new_path = os.path.join(write_path,tail)
            df = df.append({"license_number":string, "plate_path":new_path}, ignore_index=True)
            cv2.imwrite(new_path,img)
        except:
            print(img_path)
    df.to_excel(f"{name}.xlsx")
    return df

def Add_downloaded5(name):
    char_map1 = {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "8": "8",
                 "9": "9", "10": "A", "11": "B", "12": "C", "13": "D", "14": "E", "15": "F", "16": "G", "17": "H",
                 "18": "I",
                 "19": "J", "20": "K", "21": "L", "22": "M", "23": "N", "24": "O", "25": "P", "26": "Q",
                 "27": "R", "28": "S", "29": "T", "30": "U", "31": "V", "32": "W", "33": "X", "34": "Y", "35":"Z"}

    df = pd.DataFrame(columns=["plate_path", "license_number"])

    address = "F:\edu\LicensePlateDetector\datasets\\08\\05\com\\*.jpg"

    images_path = glob.glob(address)

    for img_path in images_path:
        try:
            img = cv2.imread(img_path)
            # img = img[100:500,:,:]
            img = cv2.resize(img,(int(img.shape[1] * (48/img.shape[0])),48))
            lable_path = img_path.replace(".jpg",".txt")
            string = ""
            with open(lable_path, "r") as f:
                lines = f.readlines()
                chars = []
                xs = []
                for line in lines:
                    cls = line.split(' ')[0]
                    chars.append(char_map1[cls])
                    xs.append(line.split(' ')[1])
                xs_arg_sort = np.argsort(np.asarray(xs))
                for i in xs_arg_sort:
                    string +=chars[i]
                a=0
            print(string)
            head, tail = ntpath.split(img_path)
            new_path = os.path.join(write_path,tail)
            df = df.append({"license_number":string, "plate_path":new_path}, ignore_index=True)
            cv2.imwrite(new_path,img)
        except:
            print(img_path)
    df.to_excel(f"{name}.xlsx")
    return df

if(__name__=="__main__"):
    df1 = Add_downloaded1(name="01")
    df2 = Add_downloaded2(name="02")
    df3 = Add_downloaded3(name="03")
    df4 = Add_downloaded4(name="04")
    df5 = Add_downloaded5(name="05")
    df6 = pd.read_excel("plate_data.xlsx")
    data = [df1, df2, df3, df4, df5,df6]
    df = pd.concat(data)
    df.to_excel("downloaded.xlsx")




