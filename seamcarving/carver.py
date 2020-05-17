"""
Implements a seamcarving solution and several approximations for multiple seams.

The single seam DP solution is based on
https://karthikkaranth.me/blog/implementing-seam-carving-with-python/

The following numba.jit accelerated functions are implemented:

    _forward_pass       
        Forward pass of single seam DP problem

    _seam_mask          
        Backtrack through forward pass output to find lowest energy seam


The following classes are implemented:

    fastCarver          
        Uses numda.jit to accelerate the single seam DP solution

    baseMultiSeamCarver 
        Base class for facilitating multiseam removal

    directSeamCarver    
        Implements baseMultiSeamCarver and uses fastCarver to individually 
        carve seams. Slow for large images

    roiSeamCarver       
        Implements baseMultiSeamCarver and uses the following tricks:
        * If a small number of seams needs to be removed then use fastCarver but 
          reuse the initial energy calculation for each seam
        * If a large number of seams needs to be removed then downsample image 
          to find ROI every N seams and only find seams in ROI

"""

import numpy as np
import numba
from scipy.ndimage import filters
from skimage import color, transform


@numba.jit(nopython=True)
def _forward_pass(e: np.ndarray,
                  b: np.ndarray):
    """
    Forward pass in the seam carving DP problem. 

    Args
    ---
    e: np.ndarray
        NxM matrix of local energy. Will be updated in place to calculate cumultive energy when 
        going forward
    b:np.ndarray
        NxM matrix of backtrack indecies. Can start with 0s, updated in place to find backtrack 
        indecies of lowest energy.

    Returns
    ---
    NoneType. e and b are updated in place
    """

    r, c = e.shape
    min_energy = 0
    for i in range(1, r):
        for j in range(0, c):
            # Handle the left edge of the image, to ensure we don't index a -1
            if j == 0:
                idx = np.argmin(e[i-1, j:j + 2])  # lowest energy neighbor
                b[i, j] = idx + j  # backtrack index
                # energy of the lowest energy neighbor
                min_energy = e[i-1, idx + j]
            else:
                # lowest energy neighbor
                idx = np.argmin(e[i - 1, j - 1:j + 1])
                b[i, j] = idx + j - 1  # backtrack index
                # energy of the lowest energy neighbor
                min_energy = e[i - 1, idx + j - 1]

            e[i, j] += min_energy  # add lowest energy to update cumulative energy

    return


@numba.jit(nopython=True)
def _seam_mask(e: np.ndarray, b: np.ndarray, m: np.ndarray):
    """
    Backtrack through the forward pass to find lowest energy seam. 

    Args
    ---
    e: np.ndarray
        NxM matrix of lowest energies to get to a given pixel. Used to find lowest energy terminal pixel
    b:np.ndarray
        NxM matrix of backtrack indecies. Used to backtrack
    m:np.ndarray
        NxM matrix starts as all True, then updated in place to False for lowest energy seam elements

    Returns
    ---
    NoneType. m is updated in place
    """

    r = e.shape[0]
    j = np.argmin(e[-1])  # lowest energy terimnal index
    # go back from lowest energy terminal index one row at a time
    for i in np.arange(r)[::-1]:
        m[i, j] = False
        j = b[i, j]  # use backtrack matrix to find next index
    return


class fastCarver:
    """
    Implements single seam seam carving DP problem. Utilizes _forward_pass and _seam_mask functions.

    Methods
    ---
    __init__        
        Initializes image + filter for computing local energy
    calc_energy     
        Computes energy by convolving image with filter
    forward_pass    
        Wrapper for _forward_pass
    seam_mask       
        Wrapper for _seam_mask
    """

    def __init__(self,
                 img: np.ndarray,
                 filt: np.ndarray = np.array([[1.0, 2.0, 1.0],
                                              [0.0, 0.0, 0.0],
                                              [-1.0, -2.0, -1.0]])):
        """
        fastCarver initializer

        Args
        ---
        img: np.ndarray    
            Input image, must have 3 color channels
        filt: np.ndarray   (Optional) 
            Filter for computing energy (by convolving filter with grayscale of the image)
        """

        assert isinstance(img, np.ndarray) and \
            len(img.shape) == 3 and img.shape[2] == 3

        self.img = img
        self.bw = color.rgb2gray(img).astype('float32')
        self.filtx = filt
        self.filty = filt.T

        return

    def calc_energy(self,bw=None):
        """
        Computes local energy by convolving the black and white image with the filters.
        Can be overwritten by other energy implementations
        """
        if bw is None:
            bw=self.bw
        return np.absolute(filters.convolve(bw, self.filtx)) + \
            np.absolute(filters.convolve(bw, self.filty))

    def forward_pass(self, energy: np.ndarray = None) -> (np.ndarray, np.ndarray):
        """
        Computes forward pass using _forward_pass

        Args
        ---
        energy: np.ndarray  (Optional)
            Local energy matrix on which to do forward/backward pass. 
            If none uses fastCarver.calc_energy to compute the energy using stored bw image and filters

        Returns
        ---
        energy: nd.array
            Minimum cumulative energy needed to get to a given pixel
        btrack: nd.array
            Matrix of backtrack indecies
        """
        if energy is None:
            energy = self.calc_energy()
        btrack = np.zeros_like(energy, dtype=np.uint32)
        _forward_pass(energy, btrack)  # updates in place
        return energy, btrack

    def seam_mask(self, energy: np.ndarray = None, btrack: np.ndarray = None) -> np.ndarray:
        """
        Computes the boolean matrix for the min energy seam using _seam_mask

        Args
        ---
        energy: np.ndarray  (Optional)
            Local energy matrix on which to do forward/backward pass. 
            If none uses fastCarver.calc_energy to compute the energy using stored bw image and filters
        btrack: np.ndarray  (Optional)
            Matrix of backtrack indecies.
            If none uses fastCarver.forward_pass to compute both cumulative energy and backtrack indecies

        Returns
        ---
        mask: nd.array
            Binary mask where minimum energy seam has pixels labeled False and all other pixels labeled True 
        """

        if btrack is None or energy is None:
            energy, btrack = self.forward_pass(energy)
        mask = np.ones_like(energy, dtype=np.bool)
        _seam_mask(energy, btrack, mask)  # updates in place
        return mask


class baseMultiSeamCarver:
    """
    API for carving multiple rows/columns from an image using seamcarving

    Methods
    ---
    __init__        
        Initializes image
    carve     
        Computes carved image with give numbers of rows and columns removed by
        applying baseMultiSeamCarver._carve_columns then rotating the image and applying again.
    _carve_columns    
        To be implemented by children classes. Actual multi column seamcarving strategy 
    """

    def __init__(self, img:  np.ndarray):
        self.img = img
        self.min_remaining_pixels = 10  # fewest pixels allowed to carve down to
        return

    def _carve_columns(self, img: np.ndarray, ncols: int = 0) -> np.ndarray:
        raise NotImplementedError(
            f'{type(self)}._carve_columns is not implemented')

    def carve(self, nrows: int = 0, ncols: int = 0) -> np.ndarray:
        """
        Carve nrows and ncols from self.img. Uses baseMultiSeamCarver._carve_columns for actual
        seamcarving strategy

        Args
        ---
        nrows: int
            Number of rows to remove. Must be at least self.min_remaining_pixels smaller that img height
        ncols: int
            Number of columns to remove. Must be at least self.min_remaining_pixels smaller that img width

        Returns:
        ---
        img: np.ndarray
            Carved image
        """

        assert self.img.shape[0]-nrows >= self.min_remaining_pixels
        assert self.img.shape[1]-ncols >= self.min_remaining_pixels

        img = self.img.copy()
        if ncols > 0:
            img = self._carve_columns(img, ncols)
        if nrows > 0:
            img = np.rot90(img, k=1)  # rotate 90 degrees
            img = self._carve_columns(img, nrows)
            img = np.rot90(img, k=3)  # rotate back by rotating 270 degrees
        return img


class directSeamCarver(baseMultiSeamCarver):
    """
    Implements baseMultiSeamCarver API by repeatedly computing single minimum seams and removing them

    Methods
    ---
    _carve_columns        
        Repeatedly applied fastCarver one seam at a time
    """

    def _carve_columns(self, img: np.ndarray, ncols:  int) -> np.ndarray:
        """
        Repeatedly applied fastCarver one seam at a time

        Args
        ---
        img: np.ndarray
            Input image
        ncols: int
            Number of columnar seams to remove

        Returns
        ---
        img: np.ndarray
            Carved image
        """
        for _ in range(ncols):
            carver = fastCarver(img)
            mask = carver.seam_mask()
            r, c = mask.shape
            img = img[np.stack([mask]*3, axis=2)].reshape([r, c-1, 3])
        return img


class roiSeamCarver(baseMultiSeamCarver):
    """
    Implements baseMultiSeamCarver API by intelligently selecting between two heuristics:
    1. Energy reusing seamcarving
    2. Occasionally computing fixed ROI using massively downsampled image and finding seams only in ROI

    Methods
    ---
    _carve_columns        
        Choses a strategy for seamcarving by calling either _carve_columns_energy_trick or _carve_columns_roi_trick
    _carve_columns_energy_trick
        Computes energy once then uses fastCarver.forward_pass/fastCarver.seam_mask to find additional seams
    _carve_columns_roi_trick
        Downsamples image to roiSeamCarver._DOWNSAMPLE_SIZE and uses that to find ROI in which to find seams.
        Reuses the energy calcualtion for both the downsampled image and the raw image
    """

    _DOWNSAMPLE_SIZE = 10  # target downsampled size

    def _carve_columns(self, img: np.ndarray, ncols: int) -> np.ndarray:
        """
        Choses a strategy for seamcarving by calling either _carve_columns_energy_trick (if few seams) 
        or _carve_columns_roi_trick (if many seams)

        Args
        ---
        img: np.ndarray
            Input image
        ncols: int
            Number of columnar seams to remove

        Returns
        ---
        img: np.ndarray
            Carved image
        """

        if ncols < img.shape[1]//self._DOWNSAMPLE_SIZE:
            return self._carve_columns_energy_trick(img, ncols)
        return self._carve_columns_roi_trick(img, ncols)

    def _carve_columns_energy_trick(self, img: np.ndarray, ncols: int) -> np.ndarray:
        """
        Compute energy once then reuse it for multiple applications of fastCarver.seam_mask

        Args
        ---
        img: np.ndarray
            Input image
        ncols: int
            Number of columnar seams to remove

        Returns
        ---
        img: np.ndarray
            Carved image
        """

        carver = fastCarver(img)
        E = carver.calc_energy()
        for _ in range(ncols):
            mask = carver.seam_mask(E.copy())
            r, c = mask.shape
            E = E[mask].reshape((r, c-1))  # remove seam from energy estimate
            img = img[np.stack([mask]*3, axis=2)].reshape([r, c-1, 3])
        return img

    def _carve_columns_roi_trick(self, img: np.ndarray, ncols: int) -> np.ndarray:
        """
        Downsample image to create downsampled estimate seam. Use the estimate seam to select an ROI
        from which to carve. For both downsampled and full estimate compute energy once then reuse it 
        for multiple applications of fastCarver.seam_mask

        Args
        ---
        img: np.ndarray
            Input image
        ncols: int
            Number of columnar seams to remove

        Returns
        ---
        img: np.ndarray
            Carved image
        """

        # full sized image params
        r, c, _ = img.shape
        # we want target size self._DOWNSAMPLE_SIZE
        downsample = c//self._DOWNSAMPLE_SIZE
        carver = fastCarver(img)
        E = carver.calc_energy()
        # roi bounds that will be established based on low rez image
        lb, rb = 0, img.shape[1]

        # low rez image params
        low_res_img = transform.resize(img, (r//downsample, c//downsample, 3))
        lr_carver = fastCarver(low_res_img)
        E_lr = lr_carver.calc_energy()

        for i in range(ncols):

            r, c, _ = img.shape

            # recalculate the ROI on the downsampled
            if i % downsample == 0:

                lr_mask = lr_carver.seam_mask(E_lr.copy())
                rlr, rlc = E_lr.shape
                E_lr = E_lr[lr_mask].reshape((rlr, rlc-1))
                roi_boundaries = np.argwhere(~lr_mask.all(axis=0)).flatten()
                lb, rb = max(0, (roi_boundaries[0]-1)*downsample, 0),\
                    min((roi_boundaries[-1]+1)*downsample, c)

            mask = carver.seam_mask(E[:, lb:rb].copy())
            mask = np.concatenate([np.ones((r, lb), dtype=bool),
                                   mask,
                                   np.ones((r, c-rb), dtype=bool)], axis=1)
            # adjust right boundary to be one pixel smaller (since we removed one seam)
            rb -= 1
            E = E[mask].reshape((r, c-1))
            img = img[np.stack([mask]*3, axis=2)].reshape([r, c-1, 3])

        return img
