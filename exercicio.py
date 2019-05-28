
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pytesseract

try:
    from PIL import Image
except ImportError:
    import Image


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

imagens = ['1.png']

for img_ref in imagens:
    img = cv2.imread(img_ref)
    imagemCinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imagemCinza = cv2.blur(imagemCinza, (13,13))

    imagemCinza = cv2.adaptiveThreshold(imagemCinza,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    imagemCinza = cv2.bitwise_not(imagemCinza)

    imagemCinza = cv2.dilate(imagemCinza, None, iterations=5)
    th3 = cv2.erode(imagemCinza, None, iterations = 5)

    imagemCinza, contours, hierarchy = cv2.findContours(imagemCinza, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # newContours = sorted(contours, key=lambda x: cv2.contourArea(x))

    areaImagem = imagemCinza.shape[0] * imagemCinza.shape[1]

    for ctn in contours:
        x, y, w, h = cv2.boundingRect(ctn)
        areaContorno = cv2.contourArea(ctn)

        proporcional = (100*areaContorno) / areaImagem

        if(proporcional > 50 or proporcional < 0.5):
            continue

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        imagemLeitura = img[y:h,x:w]

        if not (imagemLeitura.shape[0] and imagemLeitura.shape[1]):
            continue

        plt.imshow(imagemLeitura)
        plt.show()
        print(pytesseract.image_to_string(imagemLeitura))

    
    # print(pytesseract.image_to_string(th3))
    plt.imshow(img)
    plt.show()

