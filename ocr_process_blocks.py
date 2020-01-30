import cv2
from pdf2image import convert_from_path
import imutils
import numpy as np
import pytesseract
from pytesseract import Output


VanBan = 'VB mau/282QĐ 2019.PDF'
# Chuyển pdf sang ảnh
def Chuyen_PDF_img(VanBan):
    pages = None
    try:
        pages = convert_from_path(VanBan)
    except:
        print("khong đúng định dạng")
        return None
    TapHop_img = []
    for j, page in enumerate(pages):
        file_name = "temp/page" + str(j) + ".jpg"
        page.save(file_name, "JPEG")
        img = cv2.imread(file_name)
        TapHop_img.append(img)
    return TapHop_img

# resize ảnh
def resize_ingshow(img):
    img_return = img.copy()
    img_return = imutils.resize(img_return, height=800)
    return img_return

# Làm mịn ảnh
def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


# Lấy dữ liệu dựa trên tọa độ của hình chữ nhật bao quanh và độ dài rộng của hình chữ nhật đó
def DocChu_TuHinh(img_docChu):

    data = pytesseract.image_to_data(img_docChu,
                                     lang="vie", output_type=Output.DICT)
    orderedNames = ['block_num', 'left', 'top', 'width', 'height', 'text']
    data = np.array([data[i] for i in orderedNames], dtype=str)
    data = data.T
    data = data[data[:, 5] != '']
    data = data[data[:, 5] != ' ']
    return data

# Lấy dữ liệu dưới dạng chuỗi
def DocChu_TuHinh_2(img_docChu):
    #img_docChu = cv2.cvtColor(img_docChu, cv2.COLOR_BGR2GRAY)
    #img_docChu = cv2.threshold(src=img_docChu, thresh=0, maxval=256, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(img_docChu, lang="vie")
    return text

def TienXuLy(img):
    alpha = 1.25
    beta = 0.1
    r_left = 0
    # 2560.
    H = 2560.
    height, width = img.shape[0], img.shape[1]
    imgScale = H / height
    newX = img.shape[1] * imgScale
    # H is new height


    img = cv2.resize(img, (int(newX), int(H)), interpolation=cv2.INTER_CUBIC)
    img_goc_sau_resize = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # change image to binary image
    img = cv2.threshold( src=img, thresh=0, maxval=256, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU )[1]

    # Tạo kernel
    kernel = cv2.getStructuringElement( cv2.MORPH_RECT, (1, 1))
    # Khử nhiễu ảnh
    # img = cv2.erode(img, kernel, iterations=1)
    opening = cv2.morphologyEx(img , cv2.MORPH_OPEN, kernel)

    # Không làm mất biến dạng ảnh
    # img = cv2.dilate(img, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # Làm mờ cạnh (không cần thiết)
    img = cv2.medianBlur(closing, 1)

    # Làm mịn ảnh
    img = image_smoothening( img )

    # Ghi ảnh
    cv2.imwrite("tmp/processed_img/Result.jpg", img)

    # Đọc ảnh
    img_docChu = cv2.imread("tmp/processed_img/Result.jpg")

    return (img_docChu, img_goc_sau_resize)


def CapNhat_PhamVi(PhamVi_current, PhamVi_old):
    # phạm vi old: phạm vi của block cần cập nhật
    # phạm vi current: phạm vi của cái chữ mà cần add vào block

    # left, right, bot, top của chữ
    left_current, right_current, top_current, bot_current = PhamVi_current

     #left right, bot top của block cần update
    left_old, right_old, top_old, bot_old = PhamVi_old


    left = left_current
    if left_old < left:
        left = left_old
    right = right_current
    if right_old > right:
        right = right_old
    top = top_current
    if top_old < top:
        top = top_old
    bot = bot_current
    if bot_old > bot:
        bot = bot_old
    PhamVi_new = (left, right, top, bot)
    return PhamVi_new


# Loại bỏ các dữ liệu có kích thước lớn hơn kích thước 1 trang giấy
# True là dữ liệu không lỗi
def KiemTra(PhamVi, img):

    left, right, top, bot = PhamVi
    width = right - left
    height = bot - top

    width_img = img.shape[1]
    height_img = img.shape[0]

    if width > width_img - 5:
        if height > height_img -5 :
            return False
    return True


# Đưa dữ liệu vào mỗi block để đọc
def PhanBlock(data, img):
    TapHop_num_block = []
    TapHop_PhamVi_Block = []

    TapHop_num_block.append(-1)
    TapHop_PhamVi_Block.append((0,0,0,0))

    for dulieu in data:
        block_num, left, top, width, height, text = dulieu
        block_num = int(block_num)
        left = int(left)
        top = int(top)
        width = int(width)
        height = int(height)
        right = left + width
        bot = top + height
        PhamVi_current = (left, right, top, bot)

        if KiemTra(PhamVi_current, img) is False:
            continue

        if TapHop_num_block[len(TapHop_num_block) - 1] !=  block_num:
            TapHop_num_block.append(block_num)
            TapHop_PhamVi_Block.append(PhamVi_current)

        else:
            PhamVi_old = TapHop_PhamVi_Block[len(TapHop_PhamVi_Block) - 1]
            PhamVi_new = CapNhat_PhamVi(PhamVi_current, PhamVi_old)
            TapHop_PhamVi_Block[len(TapHop_PhamVi_Block) - 1] = PhamVi_new


    return (TapHop_num_block, TapHop_PhamVi_Block)

def DocVanBan(VanBan):
    TapHop_img = Chuyen_PDF_img(VanBan)
    if TapHop_img is None:
        return
    for img in TapHop_img:

        img_docChu, img = TienXuLy(img)
        data = DocChu_TuHinh(img_docChu)

        TapHop_num_block, TapHop_PhamVi_Block = PhanBlock(data, img)
        #TapHop_PhamVi_Block = HieuChinh(TapHop_PhamVi_Block, img_docChu)


        for left, right, top, bot in TapHop_PhamVi_Block:
            if (right - left) ==0:
                continue
            if (bot - top ) ==0 :
                continue
            im_show = img.copy()
            cv2.rectangle(im_show, (left,top), (right,bot), (0,0,255), 1)
            img_reread = img_docChu[top:bot, left: right]
            cv2.imshow("img_reread", img_reread)
            data_reread = DocChu_TuHinh_2(img_reread)
            print("####################################")
            print(data_reread)

            im_show = resize_ingshow(im_show)
            cv2.imshow("im_show", im_show)
            cv2.waitKey(0)


DocVanBan(VanBan)
