# https://rosettacode.org/wiki/Image_convolution
# http://www.songho.ca/dsp/convolution/convolution.html
# http://colah.github.io/posts/2014-07-Understanding-Convolutions/
# http://machinelearninguru.com/computer_vision/basics/convolution/convolution_layer.html

# http://setosa.io/ev/image-kernels/
# https://docs.gimp.org/2.8/en/plug-in-convmatrix.html
# https://en.wikipedia.org/wiki/Kernel_(image_processing)

from PyVutils import Cv
import Kernels

im = Cv.Load("data/59643893_2262072140712829_2661529050794688512_n.jpg")
im = Cv.Load("data/53340286_2322723884631610_6305872212347846656_n.jpg")
im = Cv.Load("data/57338813_2279331262305612_7257596330455859200_n.jpg")
im = Cv.Load("data/54424786_1196201740561013_3820028167435845632_n.jpg")
im = Cv.Load("data/59898632_551122431961932_2924826446008418304_n.jpg")

kernel = Kernels.Identity

# for any number of channel
# import cv2
# filtered = cv2.filter2D(im, -1, kernel)

# for normalized gray image only
# from scipy.ndimage.filters import convolve
# im = Cv.ToGray(im)
# im = Cv.Normalize(im)
# filtered = convolve(im, kernel)

# for normalized gray image only
# from scipy import signal
# im = Cv.ToGray(im)
# im = Cv.Normalize(im)
# filtered = signal.convolve2d(im, kernel)

filtered = Cv.Conv2D(im, kernel)

Cv.Display(filtered)