
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

def marcar(small, rgb):
        #threshold the image
        _, bw = cv2.threshold(small, 0.0, 255.0, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        bw = cv2.dilate(bw, None, iterations=4)
        # get horizontal mask of large size since text are horizontal components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

        # find all the contours
        _, contours, hierarchy,=cv2.findContours(connected.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        #Segment the text lines
        for idx in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[idx])            

            cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
            imagemLeitura = rgb[y:(y + h), x:(w + x)]
            # print(pytesseract.image_to_string(imagemLeitura))
            plt.imshow(imagemLeitura)
            plt.show()

for img_ref in imagens:
    # Executando tudo
    rgb = cv2.imread(img_ref)
    small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    marcar(small, rgb)

    plt.imshow(rgb)
    plt.show()
        

