import cv2
import csv
import glob
import numpy
import math


def write_hu_moments(label, writer):
    files = glob.glob('/Users/felicitas/PycharmProjects/pythonProject2/Shapes/' + label + '/*')  # label recibe el nombre de la carpeta
    hu_moments = []
    for file in files:
        hu_moments.append(hu_moments_of_file(file))
    for mom in hu_moments:
        flattened = mom.ravel()  # paso de un array de arrays a un array simple.
        row = numpy.append(flattened, label)  # le metes el flattened array y le agregas el label
        writer.writerow(row)  # Escribe una linea en el archivo.


def generate_hu_moments_file():
    with open('/Users/felicitas/PycharmProjects/pythonProject2/generated-files/shapes-hu-moments.csv', 'w', newline='') as file:  # Se genera un archivo nuevo (W=Write)
        writer = csv.writer(file)
        write_hu_moments("5-point-star", writer)
        write_hu_moments("Rectangle", writer)
        write_hu_moments("Triangle", writer)


# Encargada de generar los momentos de Hu para las imagenes
def hu_moments_of_file(filename):
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    bin = cv2.adaptiveThreshold(gray,120, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 67, 2)


    bin = 255 - bin

    kernel = numpy.ones((3, 3), numpy.uint8)  # Tamaño del bloque a recorrer
    # buscamos eliminar falsos positivos (puntos blancos en el fondo) para eliminar ruido.
    bin = cv2.morphologyEx(bin, cv2.MORPH_ERODE, kernel)

    contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)  # encuetra los contornos
    shape_contour = max(contours, key=cv2.contourArea)  # Agarra el contorno de area maxima

    # Calculate Moments
    moments = cv2.moments(shape_contour)  # momentos de inercia
    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments)  # momentos de Hu
    # Log scale hu moments
    for i in range(0, 7):
        huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))

    return huMoments