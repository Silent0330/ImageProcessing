import cv2
import numpy as np

def main():
    imga = cv2.imread("integrated-ckt-damaged.tif", cv2.IMREAD_GRAYSCALE)
    cv2.namedWindow('a',cv2.WINDOW_NORMAL)
    cv2.imshow('a', imga)
    
    M = imga.shape[1]   #width
    N = imga.shape[0]   #height
    P = 2*M             #padding width
    Q = 2*N             #padding height

    imgb = cv2.copyMakeBorder(imga, top=0, bottom=N, left=0, right=M, borderType=cv2.BORDER_CONSTANT, value=0)
    cv2.namedWindow('b',cv2.WINDOW_NORMAL)
    cv2.imshow('b', imgb)

    imgc = imgb.astype('int16')
    for x in range(M):
        for y in range(N):
            if (x+y) % 2 == 1:
                imgc[y][x] = -imgc[y][x]
    cv2.namedWindow('c',cv2.WINDOW_NORMAL)
    cv2.imshow('c', np.clip(imgc,0,255).astype('uint8'))
    
    F = np.fft.fft2(imgc)
    imgd = np.log(abs(F)) / 30
    minp = np.amin(imgd)
    maxp = np.amax(imgd)
    imgd = cv2.convertScaleAbs(imgd, alpha=255/(maxp - minp), beta=(-minp)*255/(maxp - minp))
    cv2.namedWindow('d',cv2.WINDOW_NORMAL)
    cv2.imshow('d', imgd)
    
    H = np.zeros((Q,P), dtype=np.float32)
    D0 = 30
    a = 2 * (D0 ** 2)
    for x in range(P):
        for y in range(Q):
            D = (x - (M)) ** 2 + (y - (N)) ** 2 #squre of distance of (x,y) to (M,N)
            H[y][x] = np.exp(-D/a)
    imge = H
    minp = np.amin(imge)
    maxp = np.amax(imge)
    imge = cv2.convertScaleAbs(imge, alpha=255/(maxp - minp), beta=(-minp)*255/(maxp - minp))
    cv2.namedWindow('e',cv2.WINDOW_NORMAL)
    cv2.imshow('e', imge)

    HF = H * F
    imgf = 20 * np.log(abs(HF))
    cv2.namedWindow('f',cv2.WINDOW_NORMAL)
    cv2.imshow('f', imgf)

    imgg = np.fft.ifft2(HF)
    imgg = imgg.real
    for x in range(P):
        for y in range(Q):
            if (x+y)%2 == 1:
                imgg[y][x] = -imgg[y][x]
    imgg = np.uint8(imgg)
    cv2.namedWindow('g',cv2.WINDOW_NORMAL)
    cv2.imshow('g', imgg)
    
    imgh = imgg[:N, :M]
    cv2.namedWindow('h',cv2.WINDOW_NORMAL)
    imgh = np.uint8(imgh)
    cv2.imshow('h', imgh)

    cv2.imwrite("a.png", imga)
    cv2.imwrite("b.png", imgb)
    cv2.imwrite("c.png", imgc)
    cv2.imwrite("d.png", imgd)
    cv2.imwrite("e.png", imge)
    cv2.imwrite("f.png", imgf)
    cv2.imwrite("g.png", imgg)
    cv2.imwrite("h.png", imgh)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()