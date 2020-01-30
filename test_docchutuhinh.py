import cv2
from pdf2image import convert_from_path
import imutils
import numpy as np
import pytesseract
from pytesseract import Output




def DocChu_TuHinh(img_docChu):
    #img_docChu = cv2.cvtColor(img_docChu, cv2.COLOR_BGR2GRAY)
    #img_docChu = cv2.threshold(src=img_docChu, thresh=0, maxval=256, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    data = pytesseract.image_to_data(img_docChu,
                                     lang="vie", output_type=Output.DICT)
    orderedNames = ['block_num', 'left', 'top', 'width', 'height', 'text']
    data = np.array([data[i] for i in orderedNames], dtype=str)
    data = data.T
    data = data[data[:, 5] != '']
    data = data[data[:, 5] != ' ']
    return data


img_docChu = cv2.imread("temp/page0.jpg")

data = DocChu_TuHinh(img_docChu)
print(data)


