

import cv2
import numpy as np

from utils.HUmomentsfromI import contour
from utils.hu_moments_generation import hu_moments_of_file

# Instrucciones en consola
from utils.label_converter import int_to_label

print("Pulse ESC para terminar.")

webcam = cv2.VideoCapture(0) # webcam 0, podría ser 1, 2... para abrir otra si hay más de una
f = cv2.getTickFrequency()   # tics del reloj por segundo

# Aquí se escribe el código de setup
#

def webCam(model):
    LastResponse = 'notstart'
    while True:
        ret, imWebcam = webcam.read()
        cv2.imshow('webcam', imWebcam)
        #hu_moments = contour(imWebcam)  # Genera los momentos de hu de la imagen en la webcam
        if cv2.waitKey(1) & 0xFF == ord('f'):

            hu_moments = contour(imWebcam) # Genera los momentos de hu de la imagen en la webcam
            sample = np.array([hu_moments], dtype=np.float32)  # numpy
            testResponse = model.predict(sample)[1]  # Predice la clase de cada file
            #tInicial = cv2.getTickCount()
            # Aquí se escribe el código para procesar la imagen imWebcam
            #imGris = cv.cvtColor(imWebcam, cv.COLOR_BGR2GRAY)
            #
            #
            # Lee la imagen y la imprime con un texto
            LastResponse=int_to_label(testResponse)
        cv2.putText(imWebcam, LastResponse, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        #cv2.imshow("result", image_with_text)
        #cv2.waitKey(0)


        # Tiempo de procesamiento
        #tFinal = cv2.getTickCount()
        #duracion = (tFinal - tInicial) / f  # Este valor se puede mostrar en consola o sobre la imagen

        # Aquí se escribe el código de visualización
        #cv.imshow('blancoYNegro', imGris)
        #
        #

        # Lee el teclado y decide qué hacer con cada tecla
        tecla = cv2.waitKey(30)  # espera 30 ms. El mínimo es 1 ms.
        # tecla == 0 si no se pulsó ninguna

        # tecla ESC para salir
        # ESC == 27 en ASCII
        if tecla == 27:
            break
        # aquí se pueden agregar else if y procesar otras teclas



cv2.destroyAllWindows()