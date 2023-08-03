import rospy
import rospkg
import argparse as ap
import numpy as np
import sys
import os
import csv
import copy
import pydbus

from rospy_message_converter import message_converter

from nav_msgs.msg           import Path, Odometry
from std_msgs.msg           import String
from geometry_msgs.msg      import PoseStamped, Twist
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg        import Joy, CompressedImage
from aarapsi_robot_pack.msg import ControllerStateInfo, MonitorDetails, RequestDataset, ResponseDataset, xyw

from pyaarapsi.core.ros_tools               import LogType, pose2xyw
from pyaarapsi.core.helper_tools            import formatException
from pyaarapsi.core.enum_tools              import enum_name, enum_value
from pyaarapsi.core.vars                    import C_I_RED, C_I_GREEN, C_I_YELLOW, C_I_BLUE, C_I_WHITE, C_RESET, C_CLEAR, C_UP_N, C_DOWN_N

from pyaarapsi.vpr_simple.vpr_dataset_tool  import VPRDatasetProcessor
from pyaarapsi.vpr_simple.vpr_helpers       import FeatureType

from pyaarapsi.vpr_classes.base             import Base_ROS_Class
from pyaarapsi.core.argparse_tools          import check_positive_float, check_bool, check_string, check_float_list, check_enum, check_positive_int, check_float
from pyaarapsi.pathing.enums                import *
from pyaarapsi.pathing.make_paths           import *

class Main_ROS_Class(Base_ROS_Class):
    def init_params(self, rate_num: float, log_level: float, reset):
        super().init_params(rate_num, log_level, reset)

        self.ZONE_LENGTH            = self.params.add(self.namespace + "/path/zones/length",        None,               check_positive_int,                     force=False)
        self.ZONE_NUMBER            = self.params.add(self.namespace + "/path/zones/number",        None,               check_positive_int,                     force=False)
        self.SLOW_LIN_VEL_MAX       = self.params.add(self.namespace + "/limits/slow/linear",       None,               check_positive_float,                   force=False)
        self.SLOW_ANG_VEL_MAX       = self.params.add(self.namespace + "/limits/slow/angular",      None,               check_positive_float,                   force=False)
        self.FAST_LIN_VEL_MAX       = self.params.add(self.namespace + "/limits/fast/linear",       None,               check_positive_float,                   force=False)
        self.FAST_ANG_VEL_MAX       = self.params.add(self.namespace + "/limits/fast/angular",      None,               check_positive_float,                   force=False)
        self.COR_OFFSET             = self.params.add(self.namespace + "/cor_offset",               0.045,              check_float,                            force=False)
        self.CONTROLLER_MAC         = self.params.add(self.namespace + "/controller_mac",           None,               check_string,                           force=False)
        self.JOY_TOPIC              = self.params.add(self.namespace + "/joy_topic",                None,               check_string,                           force=False)
        self.CMD_TOPIC              = self.params.add(self.namespace + "/cmd_topic",                None,               check_string,                           force=False)
        self.ROBOT_ODOM_TOPIC       = self.params.add(self.namespace + "/robot_odom_topic",         None,               check_string,                           force=False)
        self.VPR_ODOM_TOPIC         = self.params.add(self.namespace + "/vpr_odom_topic",           None,               check_string,                           force=False)

        self.PUBLISH_ROLLMATCH      = self.params.add(self.nodespace + "/publish_rollmatch",        True,               check_bool,                             force=reset)
        self.REJECT_MODE            = self.params.add(self.nodespace + "/reject_mode",              Reject_Mode.OLD,    lambda x: check_enum(x, Reject_Mode),   force=reset)
        self.LOOP_PATH              = self.params.add(self.nodespace + "/loop_path",                True,               check_bool,                             force=reset)
        self.PRINT_DISPLAY          = self.params.add(self.nodespace + "/print_display",            True,               check_bool,                             force=reset)
        self.PUB_INFO               = self.params.add(self.nodespace + "/publish_info",             True,               check_bool,                             force=reset)
        self.LINSTOP_OVERRIDE       = self.params.add(self.nodespace + "/override/lin_error",       0.4,                check_positive_float,                   force=reset)
        self.ANGSTOP_OVERRIDE       = self.params.add(self.nodespace + "/override/ang_error",       80*np.pi/180,       check_positive_float,                   force=reset)
        self.SAFETY_OVERRIDE        = self.params.add(self.nodespace + "/override/safety",          Safety_Mode.UNSET,  lambda x: check_enum(x, Safety_Mode),   force=reset)
        self.AUTONOMOUS_OVERRIDE    = self.params.add(self.nodespace + "/override/autonomous",      Command_Mode.UNSET, lambda x: check_enum(x, Command_Mode),  force=reset)

    def init_vars(self):
        super().init_vars()

        self.vpr_ego            = []
        self.vpr_ego_hist       = []
        self.slam_ego           = []
        self.robot_ego          = []
        self.old_robot_ego      = []
        self.lookahead          = 1.0
        self.lookahead_mode     = Lookahead_Mode.DISTANCE
        self.dt                 = 1/self.RATE_NUM.get()
        self.print_lines        = 0

        self.old_linear         = 0.0
        self.old_angular        = 0.0
        self.zone_index         = None
        self.return_stage       = Return_Stage.UNSET

        self.plan_path          = Path()
        self.ref_path           = Path()
        self.state_msg          = MonitorDetails()

        self.ready              = False
        self.new_state_msg      = False
        self.new_robot_ego      = False
        self.new_slam_ego       = False
        self.new_vpr_ego        = False

        self.dataset_queue      = []
        self.dataset_loaded     = False

        self.command_mode       = Command_Mode.STOP

        self.safety_mode        = Safety_Mode.STOP

        self.vpr_mode_ind       = enum_value(PS4_Buttons.Square)
        self.stop_mode_ind      = enum_value(PS4_Buttons.X)
        self.slam_mode_ind      = enum_value(PS4_Buttons.O)
        self.zone_mode_ind      = enum_value(PS4_Buttons.Triangle)

        self.slow_mode_ind      = enum_value(PS4_Buttons.LeftBumper)
        self.fast_mode_ind      = enum_value(PS4_Buttons.RightBumper)

        self.netvlad_ind        = enum_value(PS4_Buttons.LeftArrow)
        self.hybridnet_ind      = enum_value(PS4_Buttons.RightArrow)
        self.raw_ind            = enum_value(PS4_Buttons.UpArrow)
        self.patchnorm_ind      = enum_value(PS4_Buttons.DownArrow)

        self.lin_cmd_ind        = enum_value(PS4_Triggers.LeftStickYAxis)
        self.ang_cmd_ind        = enum_value(PS4_Triggers.RightStickXAxis)

        self.feat_arr           = { self.raw_ind: FeatureType.RAW,           self.patchnorm_ind: FeatureType.PATCHNORM, 
                                    self.netvlad_ind: FeatureType.NETVLAD,   self.hybridnet_ind: FeatureType.HYBRIDNET }

        self.twist_msg          = Twist()

        # Set up bluetooth connection check variables:
        try:
            self.bus            = pydbus.SystemBus()
            self.adapter        = self.bus.get('org.bluez', '/org/bluez/hci0')
            self.mngr           = self.bus.get('org.bluez', '/')
            self.controller_ok  = self.check_controller()
        except:
            self.print('Unable to establish safe controller connection, exitting.', LogType.FATAL)
            self.print(formatException(), LogType.DEBUG)
            self.exit()

    def init_rospy(self):
        super().init_rospy()

        self.time               = rospy.Time.now().to_sec()
        ds_requ                 = self.namespace + "/requests/dataset/"
        self.path_pub           = self.add_pub(     self.namespace + '/path',       Path,                                       queue_size=1, latch=True, subscriber_listener=self.sublis)
        self.COR_pub            = self.add_pub(     self.namespace + '/cor',        PoseStamped,                                queue_size=1)
        self.goal_pub           = self.add_pub(     self.namespace + '/path_goal',  PoseStamped,                                queue_size=1)
        self.slam_pub           = self.add_pub(     self.namespace + '/slam_pose',  PoseStamped,                                queue_size=1)
        self.speed_pub          = self.add_pub(     self.namespace + '/speeds',     MarkerArray,                                queue_size=1, latch=True, subscriber_listener=self.sublis)
        self.zones_pub          = self.add_pub(     self.namespace + '/zones',      MarkerArray,                                queue_size=1, latch=True, subscriber_listener=self.sublis)
        self.cmd_pub            = self.add_pub(     self.CMD_TOPIC.get(),           Twist,                                      queue_size=1)
        self.info_pub           = self.add_pub(     self.nodespace + '/info',       ControllerStateInfo,                        queue_size=1)
        self.rollmatch_pub      = self.add_pub(     self.nodespace + '/rollmatch/compressed',  CompressedImage,                 queue_size=1)
        self.ds_requ_pub        = self.add_pub(     ds_requ + "request",            RequestDataset,                             queue_size=1)
        self.ds_requ_sub        = rospy.Subscriber( ds_requ + "ready",              ResponseDataset,        self.ds_requ_cb,    queue_size=1)
        self.state_sub          = rospy.Subscriber( self.namespace + '/state',      MonitorDetails,         self.state_cb,      queue_size=1)
        self.robot_odom_sub     = rospy.Subscriber( self.ROBOT_ODOM_TOPIC.get(),    Odometry,               self.robot_odom_cb, queue_size=1) # wheel encoders fused
        self.slam_odom_sub      = rospy.Subscriber( self.SLAM_ODOM_TOPIC.get(),     Odometry,               self.slam_odom_cb,  queue_size=1)
        self.joy_sub            = rospy.Subscriber( self.JOY_TOPIC.get(),           Joy,                    self.joy_cb,        queue_size=1)
        self.timer_chk          = rospy.Timer(rospy.Duration(2), self.check_controller)

        self.sublis.add_operation(self.namespace + '/path',     method_sub=self.path_peer_subscribe)
        self.sublis.add_operation(self.namespace + '/zones',    method_sub=self.path_peer_subscribe)
        self.sublis.add_operation(self.namespace + '/speeds',   method_sub=self.path_peer_subscribe)

    def state_cb(self, msg: MonitorDetails):
        
        self.vpr_ego                = [msg.data.vpr_ego.x, msg.data.vpr_ego.y, msg.data.vpr_ego.w]
        self.new_vpr_ego            = True

        if not self.ready:
            return

        self.vpr_ego_hist.append(self.vpr_ego)
        self.state_msg              = msg
        self.new_state_msg          = True
    
    def robot_odom_cb(self, msg: Odometry):
        if not self.ready:
            self.old_robot_ego  = pose2xyw(msg.pose.pose)
            self.new_robot_ego  = True
            return

        self.old_robot_ego          = self.robot_ego
        self.robot_ego              = pose2xyw(msg.pose.pose)
        self.new_robot_ego          = True

    def slam_odom_cb(self, msg: Odometry):
        self.slam_ego               = pose2xyw(msg.pose.pose)
        self.new_slam_ego           = True

    def joy_cb(self, msg: Joy):
        if not self.ready:
            return
        
        if abs(rospy.Time.now().to_sec() - msg.header.stamp.to_sec()) > 0.5: # if joy message was generated longer ago than half a second:
            self.safety_mode = Safety_Mode.STOP
            if not self.SIMULATION.get():
                self.print("Bad joy data.", LogType.WARN, throttle=5)
            else:
                self.print("Bad joy data.", LogType.DEBUG, throttle=5)
            return # bad data.

        # Toggle command mode:
        if msg.buttons[self.slam_mode_ind] > 0:
            if not self.command_mode == Command_Mode.SLAM:
                self.command_mode = Command_Mode.SLAM
                self.print("Autonomous Commands: SLAM", LogType.WARN)
        elif msg.buttons[self.vpr_mode_ind] > 0:
            if not self.command_mode == Command_Mode.VPR:
                self.command_mode = Command_Mode.VPR
                self.print("Autonomous Commands: VPR", LogType.ERROR)
        elif msg.buttons[self.zone_mode_ind] > 0:
            if not self.command_mode == Command_Mode.ZONE_RETURN:
                self.command_mode = Command_Mode.ZONE_RETURN
                self.zone_index   = None
                self.return_stage = Return_Stage.UNSET
                self.print("Autonomous Commands: Zone Reset", LogType.WARN)
        elif msg.buttons[self.stop_mode_ind] > 0:
            if not self.command_mode == Command_Mode.STOP:
                self.command_mode = Command_Mode.STOP
                self.print("Autonomous Commands: Disabled", LogType.INFO)

        # Toggle feature type:
        try:
            for i in self.feat_arr.keys():
                if msg.buttons[i] and (not self.FEAT_TYPE.get() == self.feat_arr[i]):
                    rospy.set_param(self.namespace + '/feature_type', enum_name(FeatureType.RAW))
                    self.print("Switched to %s." % enum_name(self.FEAT_TYPE.get()), LogType.INFO)
                    break
        except IndexError:
            pass

        # Toggle speed safety mode:
        if msg.buttons[self.fast_mode_ind] > 0:
            if not self.safety_mode == Safety_Mode.FAST:
                self.safety_mode = Safety_Mode.FAST
                self.print('Fast mode enabled.', LogType.ERROR)
        elif msg.buttons[self.slow_mode_ind] > 0:
            if not self.safety_mode == Safety_Mode.SLOW:
                self.safety_mode = Safety_Mode.SLOW
                self.print('Slow mode enabled.', LogType.WARN)
        else:
            if not self.safety_mode == Safety_Mode.STOP:
                self.safety_mode = Safety_Mode.STOP
                self.print('Safety released.', LogType.INFO)

    def path_peer_subscribe(self, topic_name: str):
        if not self.ready:
            return
        if topic_name == self.namespace + '/path':
            self.path_pub.publish(self.viz_path)
        elif topic_name == self.namespace + '/zones':
            self.zones_pub.publish(self.viz_zones)
        elif topic_name == self.namespace + '/speeds':
            self.speed_pub.publish(self.viz_speeds)
        else:
            raise Exception('Unknown path_peer_subscribe topic: %s' % str(topic_name))
        
    def ds_requ_cb(self, msg: ResponseDataset):
        if msg.success == False:
            self.print('Dataset request processed, error. Parameters: %s' % str(msg.params), LogType.ERROR)
        try:
            index = self.dataset_queue.index(msg.params)
            self.print('Dataset request processed, success. Removing from dataset queue.')
            self.dataset_queue.pop(index)

        except ValueError:
            pass

    def param_helper(self, msg: String):
        if msg.data == self.AUTONOMOUS_OVERRIDE.name:
            if not self.AUTONOMOUS_OVERRIDE.get() == Command_Mode.UNSET:
                self.command_mode = self.AUTONOMOUS_OVERRIDE.get()
                self.return_stage = Return_Stage.UNSET
            else:
                self.command_mode = Command_Mode.STOP
        elif msg.data == self.SAFETY_OVERRIDE.name:
            if not self.SAFETY_OVERRIDE.get() == Safety_Mode.UNSET:
                self.safety_mode = self.SAFETY_OVERRIDE.get()
            else:
                self.safety_mode = Safety_Mode.STOP

    def load_dataset(self):
        # Process path data:
        self.ip  = VPRDatasetProcessor(None, try_gen=False, ros=True, printer=self.print)

        dataset_dict = self.make_dataset_dict()
        dataset_dict['ft_types'] = enum_name(FeatureType.RAW, wrap=True)

        self.try_load_dataset(dataset_dict)

        if not self.dataset_loaded: # if the model failed to generate, datasets not ready, therefore...
            dataset_msg = message_converter.convert_dictionary_to_ros_message('aarapsi_robot_pack/RequestDataset', dataset_dict)
            self.dataset_queue.append(dataset_msg)

            self.ds_requ_pub.publish(self.dataset_queue[0])

            wait_intervals = 0
            while len(self.dataset_queue):
                if rospy.is_shutdown():
                    sys.exit()
                self.print('Waiting for path dataset construction...', throttle=5)
                self.rate_obj.sleep()
                wait_intervals += 1
                if wait_intervals > 10 / (1/self.RATE_NUM.get()):
                    # Resend the oldest queue'd element every 10 seconds
                    try:
                        self.ds_requ_pub.publish(self.dataset_queue[0])
                    except:
                        pass
                    wait_intervals = 0

            self.try_load_dataset(dataset_dict)
            if not self.dataset_loaded:
                raise Exception('Datasets were constructed, but could not be loaded!')

    def try_load_dataset(self, dataset_dict):
        if not self.dataset_loaded:
            try:
                self.ip.load_dataset(dataset_dict)
                self.dataset_loaded = True
            except:
                self.dataset_loaded = False

    def publish_controller_info(self, current_ind: int, target_ind: int, current_yaw: float, zone: int, adj_lookahead: float):
        msg                         = ControllerStateInfo()
        msg.header.stamp            = rospy.Time.now()
        msg.header.frame_id         = 'map'
        msg.query_image             = self.state_msg.queryImage
        # Extract Label Details:
        msg.dvc                     = self.state_msg.data.dvc
        msg.group.gt_ego            = xyw(**{i: np.round(self.state_msg.data.gt_ego.__getattribute__(i),3) for i in ['x', 'y', 'w']})
        msg.group.vpr_ego           = xyw(**{i: np.round(self.state_msg.data.vpr_ego.__getattribute__(i),3) for i in ['x', 'y', 'w']})
        msg.group.matchId           = self.state_msg.data.matchId
        msg.group.trueId            = self.state_msg.data.trueId
        msg.group.gt_state          = self.state_msg.data.gt_state
        msg.group.gt_error          = np.round(self.state_msg.data.gt_error, 3)
        # Extract (remaining) Monitor Details:
        msg.group.mState            = np.round(self.state_msg.mState, 3)
        msg.group.prob              = np.round(self.state_msg.prob, 3)
        msg.group.mStateBin         = self.state_msg.mStateBin
        msg.group.factors           = [np.round(i,3) for i in self.state_msg.factors]

        msg.group.safety_mode       = enum_name(self.safety_mode)
        msg.group.command_mode      = enum_name(self.command_mode)

        msg.group.current_yaw       = np.round(current_yaw, 3)
        msg.group.target_yaw        = np.round(self.path_xyws[target_ind,2], 3)

        msg.group.current_ind       = current_ind
        msg.group.target_ind        = target_ind
        msg.group.reject_mode       = enum_name(self.REJECT_MODE.get())

        try:
            msg.group.true_yaw      = np.round(self.slam_ego[2], 3)
            msg.group.delta_yaw     = np.round(self.slam_ego[2] - self.vpr_ego[2], 3)
        except:
            pass
        self.new_slam_ego = False

        msg.group.lookahead         = adj_lookahead
        msg.group.lookahead_mode    = enum_name(self.lookahead_mode)

        msg.group.zone_indices      = self.zone_indices
        msg.group.zone_length       = self.zone_length
        msg.group.zone_count        = self.num_zones
        msg.group.zone_current      = zone

        self.info_pub.publish(msg)

    def check_controller(self, event=None):
        if self.SIMULATION.get():
            return True
        mngd_objs = self.mngr.GetManagedObjects()
        for path in mngd_objs:
            if mngd_objs[path].get('org.bluez.Device1', {}).get('Connected', False):
                if str(mngd_objs[path].get('org.bluez.Device1', {}).get('Address')) == self.CONTROLLER_MAC.get():
                    return True
        self.print('Bluetooth controller not found! Shutting down.', LogType.FATAL)
        sys.exit()

    def print_display(self, new_linear, new_angular, current_ind, error_v, error_y, error_yaw, zone, lin_path_err, ang_path_err):

        if self.command_mode == Command_Mode.STOP:
            command_mode_string = C_I_GREEN + 'STOPPED' + C_RESET
        elif self.command_mode == Command_Mode.SLAM:
            command_mode_string = C_I_YELLOW + 'SLAM mode' + C_RESET
        elif self.command_mode == Command_Mode.ZONE_RETURN:
            command_mode_string = C_I_YELLOW + 'Resetting zone...' + C_RESET
        elif self.command_mode == Command_Mode.VPR:
            command_mode_string = C_I_RED + 'VPR MODE' + C_RESET
        else:
            raise Exception('Bad command mode state, %s' % str(self.command_mode))

        if self.safety_mode == Safety_Mode.STOP:
            safety_mode_string = C_I_GREEN + 'STOPPED' + C_RESET
        elif self.safety_mode == Safety_Mode.SLOW:
            safety_mode_string = C_I_YELLOW + 'SLOW mode' + C_RESET
        elif self.safety_mode == Safety_Mode.FAST:
            safety_mode_string = C_I_RED + 'FAST mode' + C_RESET
        else:
            raise Exception('Bad safety mode state, %s' % str(self.safety_mode))

        base_pos_string = ''.join([C_I_YELLOW + i + ': ' + C_I_WHITE + '% 5.2f ' for i in 'xyw']) + C_RESET
        base_vel_string = ''.join([C_I_YELLOW + i + ': ' + C_I_WHITE + '% 5.2f ' for i in ['LIN','ANG']]) + C_RESET
        base_err_string = ''.join([C_I_YELLOW + i + ': ' + C_I_WHITE + '% 5.2f ' for i in ['VEL', 'C-T','YAW']]) + C_RESET
        base_ind_string = ''.join([C_I_YELLOW + i + ': ' + C_I_WHITE + '%4d ' for i in ['CUR']]) + C_RESET
        base_svm_string = ''.join([C_I_YELLOW + i + ': ' + C_I_WHITE + '%s ' for i in ['OVERRIDE','SVM']]) + C_RESET
        vpr_pos_string  = base_pos_string % tuple(self.vpr_ego)
        slam_pos_string = base_pos_string % tuple(self.slam_ego)
        speed_string    = base_vel_string % (new_linear, new_angular)
        errors_string   = base_err_string % (error_v, error_y, error_yaw)
        index_string    = base_ind_string % (current_ind)
        path_err_string = base_vel_string % (lin_path_err, ang_path_err)
        svm_string      = base_svm_string % (enum_name(self.REJECT_MODE.get()), str(self.state_msg.mStateBin))
        TAB = ' ' * 8
        lines = [
                 '',
                 TAB + '-'*20 + C_I_BLUE + ' STATUS INFO ' + C_RESET + '-'*20,
                 TAB + 'Autonomous Mode: %s' % command_mode_string,
                 TAB + '    Safety Mode: %s' % safety_mode_string,
                 TAB + '   VPR Position: %s' % vpr_pos_string,
                 TAB + '  SLAM Position: %s' % slam_pos_string,
                 TAB + ' Speed Commands: %s' % speed_string,
                 TAB + '         Errors: %s' % errors_string,
                 TAB + '     Index Info: %s' % index_string,
                 TAB + '    Zone Number: %d' % zone,
                 TAB + '     Path Error: %s' % path_err_string,
                 TAB + '     SVM Status: %s' % svm_string
                ]
        print(''.join([C_CLEAR + line + '\n' for line in lines]) + (C_UP_N%1)*(len(lines)), end='')
