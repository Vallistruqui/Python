import cv2

def on_trackbar_change(val):
    # do nothing
    pass

def contour(frame):
    window_name = 'Binary'
    trackbar_name = 'Threshold'
    Amin = 'Area min'
    Amax ='Area max'
    cv2.namedWindow(window_name)
    slider_max = 200
    cv2.createTrackbar(trackbar_name, window_name, 0, slider_max, on_trackbar_change)
    cv2.createTrackbar(Amin, window_name, 1, 24999, on_trackbar_change)
    cv2.createTrackbar(Amax, window_name, 25000, 50000, on_trackbar_change)

        # Obtenemos el valor del trackbar
    trackbar_val = cv2.getTrackbarPos(trackbar_name, window_name)
    areamax = cv2.getTrackbarPos(Amax, window_name)
    areamin = cv2.getTrackbarPos(Amin, window_name)


    # Llevamos la imagen a monocromatica
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Aplicamos threshold con el valor del trackbar
    _, thresh1 = cv2.threshold(gray, trackbar_val, 255, cv2.THRESH_BINARY)
    # Eliminamos el ruido de la imagen
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    contours_denoise, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #cv2.drawContours(frame, contornos_filtrados, -1, (255, 255, 0), 2)

    contornos_filtrados = []

    for cont in contours_denoise:
            # print(cv2.contourArea(cont))
            if areamax > cv2.contourArea(cont) > areamin:
                contornos_filtrados.append(cont)

        # cv2.drawContours(frame, contornos_filtrados, -1, (255, 255, 0), 2)

    cntmax = contours_denoise[0]

        # Busco contorno mas grande

    for cnt in contornos_filtrados:

        if cv2.contourArea(cnt) > cv2.contourArea(cntmax):
                cntmax = cnt

        cv2.drawContours(frame, [cntmax], -1, (255, 255, 255), 10)


    #if cv2.waitKey(1) & 0xFF == ord('p'):
    moments = cv2.moments(cntmax)
    humoments = cv2.HuMoments(moments)
    for i in range(0, 7):
        print("letter s2 -> h", str(i), str(humoments[i]))
        M = cv2.moments(cntmax)
        cX = int(M["m10"] / M["m00"])  # calcula la coordenada en x, int es entero. M00 es el area total
        cY = int(M["m01"] / M["m00"])
        print("cX was: {}".format(cX))
        print("cY was: {}".format(cY))

    return humoments




    cv2.imshow(window_name, cv2.flip(frame, 1))
    cv2.imshow('den', cv2.flip(closing, 1))