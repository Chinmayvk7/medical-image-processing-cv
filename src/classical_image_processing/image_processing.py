'''
    1) we start with a image.
    2) apply high pass filter to it.
        -> this performs smtg called edge detection to get the edges.
            -> it convolves each row of the image with the highpass filter.
            -> we take the output of this and convolved with the highpass filter with the coloumns.


    Image is composed of pixels = "picture element" 

    3)     edge detection - using highpass filter.
           Noise removal - using lowpass and median filters. 

           Noise in images are usually  -> random, pixel to pixel variation and of high freq.
           Noise = sudden jumps between neighboring pixels
           
           Clean image pixel values:
                    100 101 102 103
           Noisy image:
                    100 145  87 130

            Blurring = averaging neighboring pixels ( this removes the noise ) 
            New pixel = average of neighbors

            100, 145, 87
            (100 + 145 + 87) / 3 ≈ 111

    Images = signals
    Low frequency → smooth regions, objects
    High frequency → edges + noise
    
    Blurring = low-pass filter
    
    Low-pass filter: Keeps low frequencies & Removes high frequencies.

    Noise lives mostly in high frequencies
    So blur kills it.


    4) line = [100, 150, 200, 180, 120]
       filter = [1, -1]

    # Convolution process: output[n] = input[n] − input[n+1] ( Take the difference between two consecutive samples.)
        result[0] = 1×100 + (-1)×150 = -50
        result[1] = 1×150 + (-1)×200 = -50
        result[2] = 1×200 + (-1)×180 = 20
        result[3] = 1×180 + (-1)×120 = 60
... and so on

    # im = plt.imread(name) =  Loads the image file as a NumPy array.
    Image becomes a 2D array where each element is a pixel intensity (0-255 for grayscale)

        Why hpf detects edges:

            Smooth area:  [100, 101, 102, 103] → differences are small
            Edge:         [100, 100, 200, 200] → large difference at the edge!

        Apply [1, -1]:
            
            [100, 101, 102] → [1×100 + (-1)×101 = -1]  (smooth, small value)
            [100, 200, 200] → [1×100 + (-1)×200 = -100] (edge, large value!)

    
        plt.gray()- Sets the colormap to grayscale
        Without this, images might display in color (using default viridis colormap)

    
  # im = input image (2D array)
    hpf = 1D high-pass filter
    applyFilter() = function that applies a 1D filter along rows


   # Creates a 10-point averaging filter
        np.ones(10) → [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        Divide by 10 → [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    Output[i] = (Input[i] + Input[i+1] + ... + Input[i+9]) / 10
            Each pixel becomes the **average of 10 neighboring pixels**.
    
    Noise: random fluctuations
        [100, 250, 90, 240, 105] ← noisy pixels

    After averaging:
        Average = (100 + 250 + 90 + 240 + 105) / 5 = 157
            Smooths out the random spikes


    # Median Filter - Neighborhood of pixels:

Salt → very high value (white)

Pepper → very low value (black)
    
    [100, 102, 250, 98, 101,   ← 250 is noise (salt)
     99, 103, 255,  97, 100,
     101,  98, 100, 102,  99,
      ...]

    # Sort these 25 values:
    [97, 98, 98, 99, 99, 100, 100, 101, 101, 102, 103, 250, 255]
                          ↑
                     Median (middle value)

    # Replace center pixel with median (100)
    # Noise (250, 255) is ignored!


'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

'''
myFilter = 1d filter array. Ex- [1,-1] or [0.1, 0.1, 0.1, 0.1, 0.1]
data = 2d image( rows x coloumns)
convolution changes the output - output_size = input_size + filter_size - 1

nx = length of input signal
nh = length of filter
'''

def Convolve1D(x, h):
    x = np.asarray(x)
    h = np.asarray(h)

    nx = x.size
    nh = h.size

    y = np.zeros(nx + nh - 1)

    for i in range(nx):
        for j in range(nh):
            y[i + j] += x[i] * h[j]

    return y


def applyFilter(myFilter, data) :    # filtering process in which 1d filters to 2D images.
    filLength = myFilter.size
    rows, cols = data.shape         # Extract the dimensions from the image. ( ex - 400 x 600 = rows = 400 & cols = 600 ) 

    out = np.empty( ( 0, cols + filLength - 1 ) )   # Create an empty 2D array to store filtered results

    
    for line in data: # line = 1d Array representing a horizontal line of pixels. This loop iterates through each row of image.
        result= Convolve1D(line, myFilter)   
        out = np.append( out, [result], axis = 0)  
    return out
'''
  - 0 rows initially (we'll append rows in the loop)
  - ( cols + filLength - 1 ) columns (convolution output size)
'''

'''
Appends the filtered row to the output array
[result] wraps the 1D result in a list to make it a row
axis=0 means "append along rows" (stack vertically)
'''

def edgeDetection(name):
    im = plt.imread(name)          # Loads the image file as a 2d NumPy array 
    plt.gray()

    hpf = np.array([1, -1]) # high pass filter for edge detection
    plt.imshow(im)              # shows 2D array as an image
    plt.title("original")
    plt.show()

    #apply convolution both directions
    outPicHorz = applyFilter(hpf, im)
    outPic = applyFilter(hpf, outPicHorz.T)    # applyFilter() only works row-wise. so  we need to transpose it to apply convolution to coloumn.
    plt.imshow(outPic.T)
    plt.title("highpass filter both directions")
    plt.show()

    #apply convolution Horizontally
    outPic = applyFilter(hpf, im)
    plt.imshow(outPic)
    plt.title("highpass filter horizontally")
    plt.show()

    #apply convolution Vertically
    outPicHorz = applyFilter(hpf, im.T)
    plt.imshow(outPicHorz.T)
    plt.title("highpass filter Vertically")
    plt.show()

    plt.show()


def noiseRemoval(name):
    im = plt.imread(name)
    plt.gray()

    plt.imshow(im)
    plt.title("original")
    plt.show()  

    length = 10
    lpf = np.ones(length)/length        # lowpass filter for blurring ( noise removal )

    outPic= applyFilter( lpf, im)
    plt.imshow(outPic)
    plt.title("noise removal using LPF")       
    plt.show()                         # Blurred image with reduced noise


    outMed = ndimage.median_filter(im, 5)  # median filter with 5x5 window
    plt.imshow(outMed)
    plt.title("noise removal using median filter")
    plt.show()

if __name__ == "__main__":

    n = "darinGray.jpg"
    edgeDetection(n)

    n = "darinGrayNoise.jpg"
    noiseRemoval(n)

    





    