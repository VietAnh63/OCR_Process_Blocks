import re
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from pytesseract import Output
import pyvi.ViUtils as ViUtils


words = {}
with open("dict.txt", encoding="utf8") as f:
    for line in f:
        (key, val) = line.split()
        words[key] = val


def in_same_line(top1, top2):
    if top1 + 37 >= top2 >= top1 - 8:
        return True
    return False

def is_next_word(word1, word2, space_length=10):
    word1_end = int(word1[1]) + int(word1[3])
    word2_start = int(word2[1])
    if word1_end - 5 <= word2_start <= word1_end + 20:
        return True
    word1_end += space_length
    if word1_end - 5 <= int(word2[1]) <= word1_end + 20:
        return True
    return False


def is_key_word(s, keyword):
    return words.get(ViUtils.remove_accents(s).decode("utf8")) == keyword


def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3

def change_brightness(img, alpha, beta):
    img_new = np.asarray(alpha*img + beta, dtype=int)   # cast pixel values to int
    img_new[img_new>255] = 255
    img_new[img_new<0] = 0
    return img_new

def ocr(path_to_file):
    pages = convert_from_path(path_to_file)
    # All starting position is set to -1, along with their corresponding block
    dataNumber = ""
    dataPurpose = ""
    dataFor = ""
    dataDate = ""
    dataReceiver = ""

    # -1 means the keyword wasn't found
    # -2 means the field has already been matched
    check_purpose = check_for = purpose_block = for_block = check_date = \
        check_receive = check_number = number_block = -1
    alpha = 1.25
    beta = 0.1
    r_left = 0

    for j, page in enumerate(pages):

        file_name = "tmp/pdf_img/cv_page" + str(j) + ".jpg"
        page.save(file_name, "JPEG")
        img = cv2.imread(file_name)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        #img = change_brightness(img,alpha,beta)
        height, width = img.shape[0], img.shape[1]

        H = 2560.

        imgScale = H / height
        newX = img.shape[1] * imgScale
        # H is new height

        img = cv2.resize(img, (int(newX), int(H)), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = cv2.medianBlur(img,5)
        kernel = cv2.getStructuringElement( cv2.MORPH_RECT, (1, 1))
        filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
        opening = cv2.morphologyEx( filtered, cv2.MORPH_OPEN, kernel )
        closing = cv2.morphologyEx( opening, cv2.MORPH_CLOSE, kernel )
        img = image_smoothening(img)
        img = cv2.bitwise_or( img, closing )
        img = cv2.filter2D(img, -1, kernel)

        # change image to binary image
        # img = cv2.threshold(src=img, thresh=0, maxval=256, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cv2.imwrite('tmp/processed_img/Result.jpg', img)

        data = pytesseract.image_to_data(cv2.imread("tmp/processed_img/Result.jpg"),
                                         lang="vie", output_type=Output.DICT)
        orderedNames = ['block_num', 'left', 'top', 'width', 'height', 'text']
        data = np.array([data[i] for i in orderedNames], dtype=str)
        data = data.T
        data = data[data[:, 5] != '']
        data = data[data[:, 5] != ' ']

        space_length = 0
        if data.shape[0] != 0:
            space_length = int(data[1, 1]) - int(data[0, 1]) - int(data[0, 3])

        # Find block that included documentary purpose, for, etc
        shape = data.shape[0]
        for i in range(shape):
            # Detect document's number by detecting the document's id pattern, (notice that the leading numbers could be
            # falsely recognized, so we match using the trailing characters)
            if re.match(".*\\d*/\\w+.*$", data[i, 5]) and check_number == -1 \
                    and check_number != -2:
                check_number = i
                number_block = data[i, 0]

            elif is_key_word(data[i, 5], "Kính") and is_key_word(data[i + 1, 5], "gửi:") \
                    and check_for == -1 and check_for != -2:
                check_for = i + 2
                for_block = data[i, 0]

            elif is_key_word(data[i, 5], "V/v") and check_purpose == -1 and check_purpose != -2:
                check_purpose = i + 1
                purpose_block = data[i, 0]
            elif is_key_word(data[i, 5], "về") and is_key_word(data[i + 1, 5], "việc") \
                    and check_purpose == -1 and check_purpose != -2:
                check_purpose = i + 2
                purpose_block = data[i, 0]

            elif re.match("ngày.*", data[i, 5]) and check_date == -1 and check_date != -2:
                check_date = i

            elif is_key_word(data[i, 5], "Nơi") and is_key_word(data[i + 1, 5], "nhận:") \
                    and check_receive == -1 and check_receive != -2:
                check_receive = i
                r_left = int(data[i, 1])
                top = int(data[check_receive, 2])
                while in_same_line(int(data[check_receive, 2]), top):
                    check_receive += 1
                i += 1
        # end finding block

        # Initiation purpose string

        if data.shape[0] != 0:
            if check_number not in [-1, -2]:
                top = int(data[check_number, 2])
                # Move starting position to beginning of the ID field, since our starting point can start from the
                # onwards, stopping if we find any non numeric character
                while in_same_line(int(data[check_number - 1, 2]), top) \
                        and data[check_number - 1, 5].isdigit():
                    # if int(data[check_number, 1]) < newX / 2:
                    check_number -= 1

                while data[check_number, 0] == number_block and in_same_line(int(data[check_number, 2]), top):
                    if int(data[check_number, 1]) < newX / 2:
                        dataNumber += data[check_number, 5]
                    check_number += 1

                t = dataNumber.find(':')
                if t != -1:
                    dataNumber = dataNumber[t + 1::]
                check_number = -2

            if check_purpose not in [-1, -2]:
                while data[check_purpose, 0] == purpose_block:
                    dataPurpose += data[check_purpose, 5] + " "
                    check_purpose += 1
                # Remove redundant spaces
                dataPurpose = dataPurpose.strip()
                check_purpose = -2

            if check_for not in [-1, -2]:
                while data[check_for, 0] == for_block:
                    dataFor += data[check_for, 5] + " "
                    check_for += 1
                dataFor = dataFor.strip()
                check_for = -2

            if check_date not in [-1, -2]:
                top = int(data[check_date, 2])
                while in_same_line(int(data[check_date, 2]), top):
                    dataDate += data[check_date, 5] + " "
                    check_date += 1
                # Remove trailing spaces
                dataDate = dataDate.strip(' ')
                check_date = -2

            if check_receive not in [-1, -2]:
                while check_receive in range(data.shape[0]):
                    top = int(data[check_receive, 2])
                    if r_left - 10 <= int(data[check_receive, 1]) <= r_left + 10 \
                            and '-' in data[check_receive, 5]:
                        while check_receive < data.shape[0] \
                                and in_same_line(int(data[check_receive, 2]), top):
                            dataReceiver += data[check_receive, 5] + " "

                            if check_receive + 1 < data.shape[0] \
                                    and not is_next_word(data[check_receive], data[check_receive + 1], space_length):
                                break
                            check_receive += 1

                    if len(dataReceiver) >= 1 and dataReceiver[len(dataReceiver) - 1] != '\n':
                        dataReceiver += '\n'
                    check_receive += 1
                # Remove redundant spaces
                dataReceiver = dataReceiver.strip()
                check_receive = -2

    return dataNumber, dataFor, dataDate, dataPurpose, dataReceiver


print(ocr('VB mau/263QĐ 2019.PDF'))
