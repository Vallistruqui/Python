#Primer paso en machine learning es generar la muestra, la colección de características extraidas de una image
#que representan un objeto. En este caso van a hacer los invariantes de hu de las imagenes
# El primer paso sera generar y guardar los invariantes de las imagenes que son nuestra muestra

import cv2
import csv
import glob
import numpy
import math

#----------------------------------------------------------------------------------------------------
# Escribo los valores de los momentos de Hu en el archivo
#La funcion glob.glob permite descargar todas las imagenes en una carpeta, en este caso
#van a ser todas las imagenes en la carpeta shapes
#se crea el vector hu_moments
#para cada imagen en files, se guarda el momento de hu de la imagen en el vector con la funcion append
#se utiliza la funcion hu_moments_of_files que esta mas abajo
#la funcion mom.ravel pasa de un vector de vectores a un vector, esto es por que para cada imagen, el invariante de hu
#es de 7 elementos.

def write_hu_moments(label, writer):
    files = glob.glob('./Shapes/' + label + '/*')  # label recibe el nombre de la carpeta
    hu_moments = []
    for file in files:
        hu_moments.append(hu_moments_of_file(file))
    for mom in hu_moments:
        flattened = mom.ravel()  # paso de un array de arrays a un array simple.
        row = numpy.append(flattened, label)  # le metes el flattened array y le agregas el label
        writer.writerow(row)  # Escribe una linea en el archivo.

#----------------------------------------------------------------------------------------------------
# se crea la carpeta en la cual se van a guardar los invariantes de hu
def generate_hu_moments_file():
    with open('generated-files/shapes-hu-moments.csv', 'w',
              newline='') as file:  # Se genera un archivo nuevo (W=Write)
        writer = csv.writer(file)
        # Ahora escribo los momentos de Hu de cada uno de las figuras. Con el string "rectangle...etc" busca en la carpeta donde estan cada una de las imagenes
        # generar los momentos de Hu y los escribe sobre este archivo. (LOS DE ENTRENAMIENTO).
        write_hu_moments("5-point-star", writer)
        write_hu_moments("rectangle", writer)
        write_hu_moments("triangle", writer)

#-----------------------------------------------------------------------------------------------------
# Encargada de generar los momentos de Hu para las imagenes
#
def hu_moments_of_file(filename):
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 67, 2)

    # Invert the image so the area of the UAV is filled with 1's. This is necessary since
    # cv::findContours describes the boundary of areas consisting of 1's.
    bin = 255 - bin # como sabemos que las figuras son negras invertimos los valores binarios para que esten en 1.

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
        huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i])) # Mapeo para agrandar la escala.
    return huMoments
#--------------------------------------------------------------------------------

import numpy as np

from utils.label_converter import label_to_int

trainData = []
trainLabels = []

# Agarro las cosas en los archivos las guardo en variables y las mando a train data y labels
def load_training_set():
    global trainData
    global trainLabels
    with open('generated-files/shapes-hu-moments.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            class_label = row.pop() # saca el ultimo elemento de la lista
            floats = []
            for n in row:
                floats.append(float(n)) # tiene los momentos de Hu transformados a float.
            trainData.append(np.array(floats, dtype=np.float32)) # momentos de Hu
            trainLabels.append(np.array([label_to_int(class_label)], dtype=np.int32)) # Resultados
            #Valores y resultados se necesitan por separados
    trainData = np.array(trainData, dtype=np.float32)
    trainLabels = np.array(trainLabels, dtype=np.int32)
# transforma los arrays a arrays de forma numpy


# llama la funcion de arriba, se manda a entrenar y devuelve el modelo entrenado
def train_model():
    load_training_set()

    tree = cv2.ml.DTrees_create()
    tree.setCVFolds(1)
    tree.setMaxDepth(10)
    tree.train(trainData, cv2.ml.ROW_SAMPLE, trainLabels)


import numpy as np

from utils.label_converter import label_to_int

trainData = []
trainLabels = []

# Agarro las cosas en los archivos las guardo en variables y las mando a train data y labels
def load_training_set():
    global trainData
    global trainLabels
    with open('generated-files/shapes-hu-moments.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            class_label = row.pop() # saca el ultimo elemento de la lista
            floats = []
            for n in row:
                floats.append(float(n)) # tiene los momentos de Hu transformados a float.
            trainData.append(np.array(floats, dtype=np.float32)) # momentos de Hu
            trainLabels.append(np.array([label_to_int(class_label)], dtype=np.int32)) # Resultados
            #Valores y resultados se necesitan por separados
    trainData = np.array(trainData, dtype=np.float32)
    trainLabels = np.array(trainLabels, dtype=np.int32)
# transforma los arrays a arrays de forma numpy


# llama la funcion de arriba, se manda a entrenar y devuelve el modelo entrenado
def train_model():
    load_training_set()

    tree = cv2.ml.DTrees_create()
    tree.setCVFolds(1)
    tree.setMaxDepth(10)
    tree.train(trainData, cv2.ml.ROW_SAMPLE, trainLabels)

def load_and_test(model):
    files = glob.glob('../pythonProject2/Shapes/testing/*')
    for f in files:
        hu_moments = hu_moments_of_file(f) # Genera los momentos de hu de los files de testing
        sample = np.array([hu_moments], dtype=np.float32) # numpy
        testResponse = model.predict(sample)[1] # Predice la clase de cada file

        #Lee la imagen y la imprime con un texto
        image = cv2.imread(f)
        image_with_text = cv2.putText(image, int_to_label(testResponse), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("result", image_with_text)
        cv2.waitKey(0
