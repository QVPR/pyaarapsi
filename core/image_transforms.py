import cv2 # for some reason, CvBridge works more consistently if cv2 is imported (even if unused) beforehand
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
import numpy as np
import rospy

__bridge = CvBridge()

def compressed2np(msg: CompressedImage) -> np.ndarray:
    #buf     = np.ndarray(shape=(1, len(msg.data)), dtype=np.uint8, buffer=msg.data)
    #img_in  = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    # return img_in
    global __bridge
    return __bridge.compressed_imgmsg_to_cv2(msg, "rgb8")

def np2compressed(img_in: np.ndarray, add_stamp: bool = True) -> CompressedImage:
    # msg_out = CompressedImage()
    # msg_out.format = 'jpeg'
    # msg_out.data = np.array(cv2.imencode('.' + 'jpeg', img_in)[1]).tostring()
    global __bridge
    msg_out = __bridge.cv2_to_compressed_imgmsg(img_in[:,:,::-1], "jpeg")
    if add_stamp:
        msg_out.header.stamp = rospy.Time.now()
    return msg_out

def raw2np(msg: Image) -> np.ndarray:
    global __bridge
    return __bridge.imgmsg_to_cv2(msg, "rgb8")

def np2raw(img_in: np.ndarray, add_stamp: bool = True) -> Image:
    global __bridge
    msg_out = __bridge.cv2_to_imgmsg(img_in, "rgb8")
    if add_stamp:
        msg_out.header.stamp = rospy.Time.now()
    return msg_out

def imgmsgtrans(msg, transform):
    '''
    Transform ROS image data

    Inputs:
    - msg:          sensor_msgs/(Compressed)Image
    - transform:    handle to function to be applied
    Returns:
    - transformed image of type input msg
    '''

    if isinstance(msg, CompressedImage):
        img         = compressed2np(msg)
        img_trans   = transform(img)
        msg_trans   = np2compressed(img_trans)
    elif isinstance(msg, Image):
        img         = raw2np(msg)
        img_trans   = transform(img)
        msg_trans   = np2raw(img_trans)
    else:
        raise Exception("Type not CompressedImage or Image.")
    return msg_trans

def msg2img(msg):
    '''
    Convert ROS msg to cv2 image

    Inputs:
    - msg:      sensor_msgs/(Compressed)Image
    Returns:
    - converted image as cv2 array
    '''

    if isinstance(msg, CompressedImage):
        img         = compressed2np(msg)
    elif isinstance(msg, Image):
        img         = raw2np(msg)
    else:
        raise Exception("Type not CompressedImage or Image.")
    return img

def img2msg(img, mode):
    '''
    Convert cv2 img to ROS msg

    Inputs:
    - img:      cv2 image array
    - mode:     string, either 'Image' or 'CompressedImage'
    Returns:
    - sensor_msgs/(Compressed)Image
    '''

    if mode == 'CompressedImage':
        msg = np2compressed(img)
    elif mode == 'Image':
        msg = np2raw(img)
    else:
        raise Exception("Mode not 'CompressedImage' or 'Image'.")
    return msg