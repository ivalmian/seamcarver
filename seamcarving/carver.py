import numpy as np
import numba
from scipy.ndimage import filters
from skimage import color

@numba.jit(nopython=True)
def _forward_pass(e:np.ndarray,b:np.ndarray):
    r, c = e.shape
    min_energy=0
    for i in range(1, r):
        for j in range(0, c):
            # Handle the left edge of the image, to ensure we don't index a -1
            if j == 0:
                idx = np.argmin(e[i-1, j:j + 2])
                b[i, j] = idx + j
                min_energy = e[i-1, idx + j]
            else:
                idx = np.argmin(e[i - 1, j - 1:j + 1])
                b[i, j] = idx + j - 1
                min_energy = e[i - 1, idx + j - 1]

            e[i, j] += min_energy

    return 

@numba.jit(nopython=True)
def _seam_mask(e:np.ndarray, b:np.ndarray, m:np.ndarray):
    r = e.shape[0]
    j = np.argmin(e[-1])
    for i in np.arange(r)[::-1]:
        m[i, j] = False
        j = b[i, j]
    return   

class fastCarver:
     
    #stores initial image + previous carves
    def __init__(self, img:np.ndarray, filt:np.ndarray = None):
        self.img = img
        self.bw = color.rgb2gray(img) if len(img.shape)==3 else img
        self.bw = self.bw.astype('float32')
        self.filtx = filt if filt is not None else np.array([[1.0, 2.0, 1.0],[0.0, 0.0, 0.0],[-1.0, -2.0, -1.0]])
        self.filty = self.filtx.T
        self.carved = np.zeros_like(self.bw,dtype='int')
    
    #compute the energy across the full image
    def calc_energy(self):
        return np.absolute(filters.convolve(self.bw, self.filtx)) + \
               np.absolute(filters.convolve(self.bw, self.filty))

    #forward pass of the DP problem
    def forward_pass(self):
        energy = self.calc_energy()
        btrack = np.zeros_like(energy,dtype=np.uint32)             
        _forward_pass(energy,btrack) #updates in place 
        return energy,btrack          
    
    #find the lowest cumulative energy seam
    def seam_mask(self):
        energy,btrack = self.forward_pass()
        mask = np.ones_like(energy,dtype=np.bool)  
        _seam_mask(energy,btrack,mask) #updates in place
        return mask
    
            