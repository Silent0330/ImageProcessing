import cv2
from PIL import Image
import numpy as np

def main():
    imga = cv2.imread("bonescan.tif", cv2.IMREAD_GRAYSCALE)

    laplacianKernel = np.array([[-1, -1, -1],
                                [-1,  8, -1],
                                [-1, -1, -1]], dtype=np.float32)
    imgb = cv2.filter2D(imga, ddepth=cv2.CV_16S, kernel=laplacianKernel)
    minp = np.amin(imgb)
    maxp = np.amax(imgb)
    imgb = cv2.convertScaleAbs(imgb, alpha=255/(maxp - minp), beta=(-minp)*255/(maxp - minp))

    imgc = cv2.add(imga/2, imgb/2)
    imgc = cv2.convertScaleAbs(imgc)

    #kernels is fliped
    sobelKernel_x = np.array([  [-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype = np.float32)
    sobelKernel_y = np.array([  [-1,  0,  1],
                                [-2,  0,  2],
                                [-1,  0,  1]], dtype = np.float32)
    sobel_x = cv2.filter2D(imga, ddepth=cv2.CV_16S, kernel=sobelKernel_x)
    sobel_y = cv2.filter2D(imga, ddepth=cv2.CV_16S, kernel=sobelKernel_y)
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    imgd = cv2.add(sobel_x, sobel_y)
    minp = np.amin(imgd)
    maxp = np.amax(imgd)
    print(minp)
    imgd = cv2.convertScaleAbs(imgd, alpha=255/(maxp - minp), beta=(-minp)*255/(maxp - minp))

    boxKernel = np.ones((5,5))/25
    imge = cv2.filter2D(imgd, ddepth=-1, kernel=boxKernel)

    imgf = cv2.multiply(imgb, imge, scale=1.0/255)

    imgg = cv2.add(imga, imgf)

    imgh = np.array(255*(imgg/255)**0.5,dtype='uint8')

    cv2.imwrite("a.png", imga)
    cv2.imwrite("b.png", imgb)
    cv2.imwrite("c.png", imgc)
    cv2.imwrite("d.png", imgd)
    cv2.imwrite("e.png", imge)
    cv2.imwrite("f.png", imgf)
    cv2.imwrite("g.png", imgg)
    cv2.imwrite("h.png", imgh)

    b = c = d = e = f = g = h = True
    cv2.imshow("(a)", imga)
    while(True):
        key = cv2.waitKey(1)
        
        if key == ord('b'):
            if b:
                cv2.imshow("(b)", imgb)
            else:
                cv2.destroyWindow("(b)")
            b = not b
        if key == ord('c'):
            if c:
                cv2.imshow("(c)", imgc)
            else:
                cv2.destroyWindow("(c)")
            c = not c
        if key == ord('d'):
            if d:
                cv2.imshow("(d)", imgd)
            else:
                cv2.destroyWindow("(d)")
            d = not d
        if key == ord('e'):
            if e:
                cv2.imshow("(e)", imge)
            else:
                cv2.destroyWindow("(e)")
            e = not e
        if key == ord('f'):
            if f:
                cv2.imshow("(f)", imgf)
            else:
                cv2.destroyWindow("(f)")
            f = not f
        if key == ord('g'):
            if g:
                cv2.imshow("(g)", imgg)
            else:
                cv2.destroyWindow("(g)")
            g = not g
        if key == ord('h'):
            if h:
                cv2.imshow("(h)", imgh)
            else:
                cv2.destroyWindow("(h)")
            h = not h
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()