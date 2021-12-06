import cv2
import numpy as np

def main():
    imga = cv2.imread("hw3_coins2.jpg")
    cv2.imshow('a', imga)

    #gray
    imgb = cv2.cvtColor(imga, cv2.COLOR_BGR2GRAY)
    cv2.imshow('b', imgb)

    imgt = cv2.GaussianBlur(imgb, (5, 5), 0)

    #OTSU
    ret, imgc = cv2.threshold(imgt, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow('c', imgc)
    
    #opening
    kernel = np.ones((3,3), np.uint8)
    imgd = cv2.morphologyEx(imgc, cv2.MORPH_OPEN, kernel, iterations = 2)
    cv2.imshow('d', imgd)
    
    #sure background
    imge = cv2.dilate(imgd, kernel, iterations=2)
    cv2.imshow('e', imge)
    
    #distance transform
    imgf = cv2.distanceTransform(imge, cv2.DIST_L2, 5)
    minp = np.amin(imgf)
    maxp = np.amax(imgf)
    imgf = cv2.convertScaleAbs(imgf, alpha=255/maxp, beta=-255*minp/maxp)
    cv2.imshow('f', imgf)
    
    #sure foreground
    ret, imgg = cv2.threshold(imgf, 0.5*imgf.max(), 255, cv2.THRESH_BINARY)
    cv2.imshow('g', imgg)
    
    #unknow
    imgh = np.uint8(imgg)
    imgh = cv2.subtract(imge, imgh)
    cv2.imshow('h', imgh)

    #markers
    ret, markers = cv2.connectedComponents(imgg)
    markers = markers+1
    markers[imgh==255] = 0
    maxp = np.amax(markers)
    imgi = cv2.convertScaleAbs(markers, alpha=255/maxp)
    imgi = cv2.applyColorMap(imgi, cv2.COLORMAP_JET)
    cv2.imshow('i', imgi)
    
    #watershed
    imgk = imga
    markers = cv2.watershed(imgk, markers)
    maxp = np.amax(markers)
    imgk = cv2.convertScaleAbs(markers, alpha=255/maxp)
    imgk = cv2.applyColorMap(imgk, cv2.COLORMAP_JET)
    cv2.imshow('k',imgk)
    imgj = imga
    imgj[markers == -1] = [0,0,255]
    cv2.imshow('j', imgj)
    
    cv2.imwrite("a.png", imga)
    cv2.imwrite("b.png", imgb)
    cv2.imwrite("c.png", imgc)
    cv2.imwrite("d.png", imgd)
    cv2.imwrite("e.png", imge)
    cv2.imwrite("f.png", imgf)
    cv2.imwrite("g.png", imgg)
    cv2.imwrite("h.png", imgh)
    cv2.imwrite("i.png", imgi)
    cv2.imwrite("j.png", imgj)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()