import os, sys
from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


normal_install_require_list = [
    'numpy', 'torch', 'torchvision',
    'tqdm', 'scipy', 'Pillow', 'scikit-learn', 
    'natsort', 'matplotlib', 'opencv-python',
    'bokeh', 'piexif', 'pandas', 'packaging',
    'pytorch-lightning', 'pydot', 'argparse',
    'pydbus', 'lightglue', 'fastdist', 'flask',
    'requests',
]

normal_install_versioned_require_list = [
    'numpy==1.21.0', 'torch==2.0.1', 'torchvision==0.15.2',
    'tqdm==4.64.1', 'scipy==1.3.3', 'Pillow==10.2.0', 'scikit-learn==0.22.2.post1', 
    'natsort==8.3.1', 'matplotlib==3.7.3', 'opencv-python==4.7.0.72',
    'bokeh==3.0.3', 'piexif==1.1.3', 'pandas==1.5.3', 
    'packaging==24.0', 'pytorch-lightning==1.5.10',
    'pydot==1.4.1', 'argparse==1.4.0', 'pydbus==0.6.0',
    'lightglue==0.0', 'fastdist==1.1.5', 'flask==2.3.2',
    'requests==2.22.0', 

]

# ROS Packages:
ros_install_require_list = [
    'cv-bridge', 'rospkg', 'rospy', 
    'rospy_message_converter', 'aarapsi_robot_pack',
    'genpy', 'rosbag', 'tf',

]

ros_install_versioned_require_list = [
    'cv-bridge==1.16.2', 'rospkg==1.5.0', 'rospy==1.16.0', 
    'rospy_message_converter==0.5.9', 'aarapsi_robot_pack',
    'genpy==0.6.15', 'rosbag==1.16.0', 'tf==1.13.2',

]

install_require_list = normal_install_require_list
# install_require_list = normal_install_versioned_require_list
# install_require_list = normal_install_require_list + ros_install_require_list
# install_require_list = normal_install_versioned_require_list + ros_install_versioned_require_list

# # workaround as opencv-python does not show up in "pip list" within a conda environment
# # we do not care as conda recipe has py-opencv requirement anyhow
# is_conda = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
# if not is_conda:
#     install_require_list.append('opencv-python')

setup(name='pyaarapsi',
      version='0.2.0',
      description='PyAARAPSI: Python Package for AARAPSI Project',
      author='Owen Claxton, Connor Malone, and Helen Carson',
      author_email='claxtono@qut.edu.au',
      url='https://github.com/QVPR/pyaarapsi',
      license='MIT',
      classifiers=[
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
      ],
      python_requires='>=3.6',
      install_requires=install_require_list,
      packages=find_packages(),
      keywords=[
          'python', 'place recognition', 'image retrieval', 'computer vision', 'robotics'
      ]
)
