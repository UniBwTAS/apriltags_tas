# apriltags_tas

`apriltags_tas` is a Robot Operating System (ROS) wrapper of the [AprilTags C++ visual fiducial detector](https://github.com/Humhu/apriltags) based on the AprilTag fiducial marker system https://april.eecs.umich.edu/software/apriltag.html.
It also includes the improvements proposed in the __ICRA 2020__ Paper __Determining and Improving the Localization Accuracy of AprilTag Detection__.

**Authors**: Jan Kallwies, Bianca Forkel

## Quickstart

Starting with a working ROS installation (Kinetic and Melodic are supported):
```
export ROS_DISTRO=melodic               # Set this to your distro, e.g. kinetic or melodic
source /opt/ros/$ROS_DISTRO/setup.bash  # Source your ROS distro 
mkdir -p ~/catkin_ws/src                # Make a new workspace 
cd ~/catkin_ws/src                      # Navigate to the source space
git clone https://github.com/UniBwTAS/apriltags_tas.git      # Clone the git repository
cd ~/catkin_ws                          # Navigate to the workspace
rosdep install --from-paths src --ignore-src -r -y  # Install any missing packages
catkin build    # Build all packages in the workspace (catkin_make_isolated will work also)

# Start it by:
roslaunch apriltags_tas apriltag_detection.launch
```

In order to use the detector with your own image, please change the example image path or the image topic to be subscribed in the launch file `apriltag_detection.launch`.

The configuration of the tag description is adopted from the ROS package [`apriltag_ros`](https://github.com/RIVeR-Lab/apriltags_ros). Please see [ROS wiki](http://wiki.ros.org/apriltag_ros) for details and tutorials.

## Reference

If you use this code, please cite:

- J. Kallwies, B. Forkel and H.-J. Wuensche, “[Determining and Improving the Localization Accuracy of AprilTag Detection](https://ieeexplore.ieee.org/document/9197427),” in Proceedings of IEEE International Conference on Robotics and Automation (ICRA), June 2020.

```
@InProceedings{Kallwies2020_AprilTagAccuracy,
  author    = {Jan Kallwies AND Bianca Forkel AND Hans-Joachim Wuensche},
  title     = {{Determining and Improving the Localization Accuracy of AprilTag Detection}},
  booktitle = {Proceedings of IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2020},
  address   = {Paris, France (Virtual Conference)},
  month     = jun,
}
```
