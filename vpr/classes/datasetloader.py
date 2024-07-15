#!/usr/bin/env python3

import sys
import rospy
from rospy_message_converter import message_converter

from aarapsi_robot_pack.msg import ResponseDataset, RequestDataset

from pyaarapsi.core.roslogger import LogType
from pyaarapsi.vpr.vpr_dataset_tool import VPRDatasetProcessor
from pyaarapsi.vpr.classes.base import BaseROSClass

class DatasetLoader(BaseROSClass):
    '''
    Base class to handle dataset loading.
    '''
    def init_vars(self, *args, **kwargs) -> None:
        super().init_vars(*args, **kwargs)

        # Inter-loop variables for dataset loading control:
        self.dataset_queue = [] # List of dataset parameters pending construction

    def init_rospy(self, *args, **kwargs) -> None:
        super().init_rospy(*args, **kwargs)

        ds_requ = self.namespace + "/requests/dataset/"
        self.ds_requ_pub = self.add_pub(ds_requ + "request", RequestDataset, queue_size=1)
        self.ds_requ_sub = rospy.Subscriber(ds_requ + "ready",   ResponseDataset, self.ds_requ_cb,
                                            queue_size=1)

    def ds_requ_cb(self, msg: ResponseDataset):
        '''
        Dataset request callback; handle confirmation of dataset readiness
        '''
        if not msg.success:
            self.print(f'Dataset request processed, error. Parameters: {str(msg.params)}',
                       LogType.ERROR)
        try:
            # on separate line to try trigger ValueError failure:
            index = self.dataset_queue.index(msg.params)
            self.print('Dataset request processed, success. Removing from dataset queue.')
            self.dataset_queue.pop(index)
        except ValueError:
            pass

    def load_dataset(self, _ip: VPRDatasetProcessor, _dict: dict) -> None:
        '''
        Load in dataset to generate path and to utilise VPR index information
        '''
        # Try load in dataset:
        dataset_loaded = _ip.load_dataset(_dict) != ''
        if not dataset_loaded: # if the model failed to generate, the dataset is not ready:
            # Request dataset generation:
            dataset_msg = message_converter.convert_dictionary_to_ros_message(\
                                                'aarapsi_robot_pack/RequestDataset', _dict)
            self.dataset_queue.append(dataset_msg)
            self.ds_requ_pub.publish(self.dataset_queue[0])

            # Wait for news of dataset generation:
            wait_intervals = 0
            while len(self.dataset_queue) > 0: # while there is a dataset we are waiting on:
                if rospy.is_shutdown():
                    sys.exit()
                self.print('Waiting for dataset construction...', throttle=5)
                self.rate_obj.sleep()
                wait_intervals += 1
                if wait_intervals > 10 / (1/self.RATE_NUM.get()):
                    # Resend the oldest queue'd element every 10 seconds
                    self.ds_requ_pub.publish(self.dataset_queue[0])
                    wait_intervals = 0
            # Try load in the dataset again now that it is ready
            if _ip.load_dataset(_dict) == '':
                raise Exception('Dataset was constructed, but could not be loaded!')
