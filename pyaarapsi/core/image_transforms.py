#!/usr/bin/env python3
'''
A collection of functions to help transform images
'''
from typing import Union, Callable, overload, Literal
# for some reason, CvBridge works more consistently if cv2 is imported (even if unused) beforehand:
import cv2 #pylint: disable=W0611
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from numpy.typing import NDArray
import rospy

__BRIDGE = CvBridge()

def compressed2np(msg: CompressedImage) -> NDArray:
    '''
    Convert a sensor_msgs/CompressedImage to a numpy array
    '''
    #buf     = np.ndarray(shape=(1, len(msg.data)), dtype=np.uint8, buffer=msg.data)
    #img_in  = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    # return img_in
    return __BRIDGE.compressed_imgmsg_to_cv2(msg, "rgb8")

def np2compressed(img_in: NDArray, add_stamp: bool = True) -> CompressedImage:
    '''
    Convert a numpy array to a sensor_msgs/CompressedImage
    '''
    # msg_out = CompressedImage()
    # msg_out.format = 'jpeg'
    # msg_out.data = np.array(cv2.imencode('.' + 'jpeg', img_in)[1]).tostring()
    msg_out = __BRIDGE.cv2_to_compressed_imgmsg(img_in[:,:,::-1], "jpeg")
    if add_stamp:
        msg_out.header.stamp = rospy.Time.now()
    return msg_out

def raw2np(msg: Image) -> NDArray:
    '''
    Convert a sensor_msgs/Image to a numpy array
    '''
    return __BRIDGE.imgmsg_to_cv2(msg, "rgb8")

def np2raw(img_in: NDArray, add_stamp: bool = True) -> Image:
    '''
    Convert a numpy array to a sensor_msgs/Image
    '''
    msg_out = __BRIDGE.cv2_to_imgmsg(img_in, "rgb8")
    if add_stamp:
        msg_out.header.stamp = rospy.Time.now()
    return msg_out

@overload
def imgmsgtrans(msg: CompressedImage, transform: Callable) -> CompressedImage:
    ...

@overload
def imgmsgtrans(msg: Image, transform: Callable) -> Image:
    ...

def imgmsgtrans(msg: Union[Image, CompressedImage], transform: Callable
                ) -> Union[Image, CompressedImage]:
    '''
    Transform ROS image data

    Inputs:
    - msg:          sensor_msgs/(Compressed)Image
    - transform:    handle to function to be applied
    Returns:
    - transformed image of type input msg
    '''
    assert isinstance(msg, CompressedImage) or isinstance(msg, Image)
    if isinstance(msg, CompressedImage):
        img = compressed2np(msg)
        img_trans = transform(img)
        msg_trans = np2compressed(img_trans)
    else:
        img = raw2np(msg)
        img_trans = transform(img)
        msg_trans = np2raw(img_trans)
    return msg_trans

def msg2img(msg: Union[Image, CompressedImage]) -> NDArray:
    '''
    Convert ROS msg to cv2 image

    Inputs:
    - msg:      sensor_msgs/(Compressed)Image
    Returns:
    - converted image as cv2 array
    '''
    assert isinstance(msg, CompressedImage) or isinstance(msg, Image)

    if isinstance(msg, CompressedImage):
        img = compressed2np(msg)
    else:
        img = raw2np(msg)
    return img

@overload
def img2msg(img: NDArray, mode: Literal['Image']) -> Image:
    ...

@overload
def img2msg(img: NDArray, mode: Literal['CompressedImage']) -> CompressedImage:
    ...

def img2msg(img: NDArray, mode: Literal['Image', 'CompressedImage']
            ) -> Union[Image, CompressedImage]:
    '''
    Convert cv2 img to ROS msg

    Inputs:
    - img:      cv2 image array
    - mode:     string, either 'Image' or 'CompressedImage'
    Returns:
    - sensor_msgs/(Compressed)Image
    '''
    assert mode in ['Image', 'CompressedImage']

    if mode == 'CompressedImage':
        msg = np2compressed(img)
    else:
        msg = np2raw(img)
    return msg
