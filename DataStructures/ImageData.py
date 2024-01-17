from bs4 import BeautifulSoup

class ImageData:
    def __init__(self):
        self.image = None
        self.bounding_box = None

    def set_image(self,image):
        self.image = image

    def set_bounding_box(self,x_min,y_min,x_max,y_max):
        self.bounding_box = {"tl":(x_min,y_min),"bl":(x_min,y_max),"br":(x_max,y_max),"tr":(x_min,y_min)}


class XML_Data:
    def __init__(self):
        self.x_min = None
        self.y_min = None
        self.x_max = None
        self.y_max = None

    def set_bounds(self,x_min,y_min,x_max,y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def read_XML_data(self,XML_file_address):
        with open(XML_file_address, 'r') as xml_file:
            data = xml_file.read()
        bs_data = BeautifulSoup(data, 'xml')

        a = 0

