# pyaarapsi
Multi-purpose python package, for a variety of needs across the AARAPSI Project

We've got:
## core
- ajax_tools.py: Create and manage AJAX servers, requests for use with python ```Bokeh``` or inter-node data sharing.
- argparse_tools.py: Methods to parse a variety of data types, for use with python ```argparse``` module.
- enum_tools.py: Methods to ease use or improve readability of python ```enum``` enumerations.
- file_system_tools.py: Methods to help explore the file system.
- helper_tools.py: A variety of methods and classes to assist and accelerate python development.
- image_transforms.py: ```ROS```, ```cv2```, ```cv_bridge```, and ```numpy``` fusion methods.
- missing_pixel_filler.py: Code I ripped from an online github, realised was bad, then upgraded (see Owen's corrections). Does what name suggests.
- os_tools.py: Commands to ease python utilisation of Linux ```screen``` package.
- ros_tools.py: A variety of methods and classes to assist and accelerate ```ROS``` development in python.
- roslogger.py: Custom build of ```ROS```'s ```rospy``` logging system to add exposed boolean for whether text was printed. Also useful for compatibility with non-```ROS``` codebases.
- transforms.py: Python tools for homogeneous transformations as well as 2D and 3D rotation matrices in ```numpy```.
- vars.py: Some defined codes, currently for colouring terminal output

## pathing
- base.py: Base class(es) for ```aarapsi_robot_pack```'s path_follower.py.
- basic.py: Helper commands for ```aarapsi_robot_pack```'s path_follower.py.
- enums.py: enumerations for ```aarapsi_robot_pack```'s path_follower.py.

## vpr_classes
- base.py: A base class for ```aarapsi_robot_pack``` nodes.
- download_models.py: Helper commands to pull down NetVLAD and hybridnet models from google drive
- hybridnet.py Container for HybridNet, somewhat accelerated.
- netvlad.py: Container for NetVLAD, somewhat accelerated.
- netvlad_lib.py: Helper commands for netvlad.py

## vpr_simple
- npz_dataset_finder.py: Tool for searching up npz files, to aid in use of vpr_dataset_tool.py
- svm_model_tool.py: All-in-one support vector machine generator, loader, and accessor.
- vpr_dataset_tool.py: All-in-one ```ROS``` bag to ```numpy``` npz generator, loader, fixer-upper.
- vpr_feature_tool.py: **Legacy**, please see vpr_dataset_tool.py
- vpr_helpers.py: Enumerations and tools for VPR feature extraction.
- vpr_image_methods.py: Tools for making, colouring, and operating on ```numpy``` images.
- vpr_plots.py: Might broken. Intended for Bokeh integration.
- vpr_plots_new.py: Broken. Intended for Bokeh integration.

## vpred
- robotmonitor.py:
- robotrun.py:
- robotvpr.py:
- vpred_factors.py:
- vpred_tools.py:
