# HeightGridNetwork - HGNet

Height grid location of objects from lidar data. Allied with a 2D camera detector, it allows the identification of 3D bounding boxes for objects.

## Dependencies

This was built with:

+ [Anaconda](https://anaconda.org) **v4.3.30**
+ [Python](https://anaconda.org/anaconda/python) **v3.5.2**

This was built under Windows, usage in Ubuntu is not tested at the moment. Still, usage on Ubuntu should not have cause any troubles.

Make sure you have an environment with the following packages installed.

+ [Numpy](http://www.numpy.org/) **v1.13.1**
+ [Opencv](https://opencv.org/) **v3.3.0**
+ [Vtk](https://www.vtk.org/) **v8.1.0**

## Point Cloud Data

Data used to experiment on is from the 3D object detection KITTI dataset.

The KITTI Vision Benchmark dataset for 3D Object Detection is available at [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php) for academic use.

Paper:

```
@ARTICLE{Geiger2013IJRR,
  author = {Andreas Geiger and Philip Lenz and Christoph Stiller and Raquel Urtasun},
  title = {Vision meets Robotics: The KITTI Dataset},
  journal = {International Journal of Robotics Research (IJRR)},
  year = {2013}
} 
```