# pyaarapsi
Multi-purpose python package, for a variety of needs across the AARAPSI Project

Sister ROS Package: https://github.com/QVPR/aarapsi_robot_pack

# Installation steps:
Grab the repo:
```
git clone https://github.com/QVPR/pyaarapsi.git
```
One of:
```
>>> pip install -e .[basic] # Only standard python dependencies
>>> pip install -e .[basic,ros] # Standard and ROS python dependencies
>>> pip install -e .[basic-versioned] # Only standard python dependencies with developer's flavour of package versions
>>> pip install -e .[basic-versioned,ros-versioned] # Standard and ROS python dependencies with developer's flavour of package versions
>>> pip install -e .[complete] # Same as [basic,ros]
>>> pip install -e .[complete-versioned] # Same as [basic-versioned,ros-versioned] (Recommended)
```

# Contents
## core
- ajax_tools.py: Create and manage AJAX servers, requests for use with python ```Bokeh``` or inter-node data sharing.
- argparse_tools.py: Methods to parse a variety of data types, for use with python ```argparse``` module.
- enum_tools.py: Methods to ease use or improve readability of python ```enum``` enumerations.
- file_system_tools.py: Methods to help explore the file system.
- helper_tools.py: A variety of methods and classes to assist and accelerate python development.
- image_transforms.py: ```ROS```, ```cv2```, ```cv_bridge```, and ```numpy``` fusion methods.
- lightglue.py: For use with LightGlue
- missing_pixel_filler.py: Code I ripped from an online github, realised was bad, then upgraded (see Owen's corrections). Does what name suggests.
- os_tools.py: Commands to ease python utilisation of Linux ```screen``` package.
- plotting_tools.py: Generic plotting helpers
- ros_tools.py: A variety of methods and classes to assist and accelerate ```ROS``` development in python.
- roslogger.py: Custom build of ```ROS```'s ```rospy``` logging system to add exposed boolean for whether text was printed. Also useful for compatibility with non-```ROS``` codebases.
- transforms.py: Python tools for homogeneous transformations as well as 2D and 3D rotation matrices in ```numpy```.
- vars.py: Some defined codes, currently for colouring terminal output

## core.classes
- defaultdict.py: For DefaultDict, which extends ```dict```, to provide a default key.
- object_storage_handler.py: For Object_Storage_Handler, to provide an easy loader/saver.

## examples
- extract_images_from_rosbags.py: Needs update.
- working_with_VPRDatasetProcessor_and_SVMModelProcessor.py: Needs update.

## nn
- classes.py:
- colours.py:
- enums.py:
- exceptions.py:
- general_helpers.py:
- modules.py:
- nn_factors.py:
- nn_helpers.py:
- param_helpers.py:
- params.py:
- visualize.py:
- vpr_helpers.py:

## pathing
- basic_rospy.py: Helper commands for ```aarapsi_robot_pack```'s path_follower.py; has ROS dependencies.
- basic.py: Helper commands for ```aarapsi_robot_pack```'s path_follower.py; no ROS dependencies.
- enums.py: enumerations for ```aarapsi_robot_pack```'s path_follower.py.
- extended_follower_base.py: Extended base class(es) for ```aarapsi_robot_pack```'s path_follower.py.
- simple_follower_base.py: Simple base class(es) for ```aarapsi_robot_pack```'s path_follower.py.

## vpr_classes
- apgem_lib.py: Helper commands for apgem.py.
- apgem.py: Container for AP-GeM, somewhat accelerated.
- base.py: A base class for ```aarapsi_robot_pack``` nodes.
- dataset_loader_base.py: A base class for ```aarapsi_robot_pack``` nodes.
- download_models.py: Helper commands to pull down models from cloud storage solution
- hybridnet.py Container for HybridNet, somewhat accelerated.
- netvlad_lib.py: Helper commands for netvlad.py
- netvlad.py: Container for NetVLAD, somewhat accelerated.
- salad_lib.py: Helper commands for salad.py
- salad.py: Container for DinoV2 SALAD, somewhat accelerated.

## vpr_simple
- config.py: Tool for setting up paths used in various ```pyaarapsi``` instances.
- npz_dataset_finder.py: Tool for searching up npz files, to aid in use of vpr_dataset_tool.py
- svm_model_tool.py: All-in-one support vector machine generator, loader, and accessor.
- vpr_dataset_tool.py: All-in-one ```ROS``` bag to ```numpy``` npz generator, loader, fixer-upper.
- vpr_helpers.py: Enumerations and tools for VPR feature extraction.
- vpr_image_methods.py: Tools for making, colouring, and operating on ```numpy``` images.
- vpr_plots.py: Might broken. Intended for Bokeh integration.
- vpr_plots_new.py: Broken. Intended for Bokeh integration.

## vpred
- gradseq_tools.py:
- robotmonitor.py:
- robotrun.py:
- robotvpr.py:
- visual_preproc.py:
- vpred_factors.py:
- vpred_tools.py:
