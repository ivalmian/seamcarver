import unittest
from skimage import io
from seamcarving import carver

       
class carverTest(unittest.TestCase):

    test_image = io.imread('test_seamcarving/img/unsplash.jpg')

    def test_initCarver(self):  
        c = carver.fastCarver(self.test_image)
        assert c is not None
    
    def test_calcEnergy(self):
        c = carver.fastCarver(self.test_image)
        c.calc_energy()

    def test_calcForwardPass(self):
        c = carver.fastCarver(self.test_image)
        c.forward_pass()

    def test_calcSeamMask(self):
        c = carver.fastCarver(self.test_image)
        c.seam_mask()

    def test_directSeamCarver(self):
        c = carver.directSeamCarver(self.test_image)
        c.carve(10,10)

    def test_roiSeamCarver(self):
        c = carver.roiSeamCarver(self.test_image)
        c.carve(1,1)
    
    def test_roiSeamCarver_large(self):
        c = carver.roiSeamCarver(self.test_image)
        c.carve(200,200)

