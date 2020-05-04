## Numba accelerated seam carving python package + some tricks



### Currently implemented carvers
* seamcarving.carver.directSeamCarver
* seamcarving.carver.roiSeamCarver

### Usage
```
crv = seamcarving.carver.<carver_type>(image)
carved_image = crv.carve(nrows,ncols)
```

For example in notebook see notebooks/

### For test run

```
coverage run -m test_seamcarving
```
or 
```
run_test.sh
```

