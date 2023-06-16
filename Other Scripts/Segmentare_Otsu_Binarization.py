import cv2
from matplotlib import pyplot as plt

img = cv2.imread(r"C:\Users\Alfred\Desktop\Importante\Licenta\Imagini\0X1B65D8D908C385B_sis_1.jpg",cv2.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read"

alpha = 0 # Contrast control
beta = 10 # Brightness control
enh_img = cv2.convertScaleAbs(img, alpha, beta)

# global thresholding
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# Otsu's thresholding
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# operations on enhanced image
# global thresholding
enh_ret1,enh_th1 = cv2.threshold(enh_img,127,255,cv2.THRESH_BINARY)
# Otsu's thresholding
enh_ret2,enh_th2 = cv2.threshold(enh_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
enh_blur = cv2.GaussianBlur(enh_img,(5,5),0)
enh_ret3,enh_th3 = cv2.threshold(enh_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3,
          enh_img, 0, enh_th1,
          enh_img, 0, enh_th2,
          enh_blur, 0, enh_th3,]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding",
          'Enhanced Noisy Image','Histogram','Global Thresholding (v=127)',
          'Enhanced Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered enhanced Image','Histogram',"Otsu's Thresholding"]
for i in range(6):
    plt.subplot(6,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(6,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(6,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()