# handical

Hand-eye extrinsic calibration of multiple fixed cameras and a robot arm.

A chessboard is mounted on the robot end-effector (ee). We move the ee around to different locations within the cameras view
and assume that the ee pose in the robot's base frame is known via forward kinematic. (Future versions can address noises
on those ee poses.) This script takes corner point measurements and compute the camera poses wrt the robot's base
and also the chessboard's pose wrt the ee.

## Installation
- Dependencies:
  + OpenCV2 (with python support)
    ```
        pip install opencv-python
    ```
  + PyYAML
    ```
        pip install pyyaml
    ```
  + gtsam: (tested with tag 4.0.2)
    * Download:
    ``` bash
       git clone git@github.com:borglab/gtsam.git
       git checkout 4.0.2 -b 4.0.2
    ```
    * Build and install: you need to build gtsam with cython wrapper support.
    Please follow the instructions [here](https://github.com/borglab/gtsam/blob/4.0.2/cython/README.md)
    for the dependencies and which compile flag to set.

- Build:
  + First, make sure your `PATH` env contains gtsam's `install/lib` folder. This is necessary for our cmake script to be able to find gtsam's cmake scripts.
  + Also, make sure your `PYTHONPATH` contains gtsam's Cython install path (normally in `CMAKE_INSTALL_PREFIX/cython` when you build gtsam).
  This is so that it can find gtsam Cython header: `gtsam/gtsam.pxd`.
  + Then just build and install the project using cmake.
