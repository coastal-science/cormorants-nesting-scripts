# RAW to TIF
Most ortho-mosaics are exported from stitching as TIF images. However,
occasionally the ortho-mosaics are exported as other file formats such as RAW
and JPG. These images must be converted into TIFs before you can proceed with
tiling and annotation.

Unfortunately, we haven't found a way to reliably do this conversion 
programmatically (particularly from RAW images). Instead, we use image editing
tools such as Photoshop and GIMP to do these conversions manually. 

If using Photoshop, you can use the following steps to do the conversion from
RAW to TIF: 
1. Opening the `.raw` image in Photoshop
   * Include the Dimensions of the image (should be in the .raw file name)
   * Choose 3 interleaved channels

2. Convert the image to a `.tif`
   * File > Save As ... 
   * Select TIF
   * Choose the LZW compression method
   * Save
