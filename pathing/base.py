import rospy
import numpy as np
import sys
import pydbus

from rospy_message_converter import message_converter

from nav_msgs.msg           import Path, Odometry
from std_msgs.msg           import String
from geometry_msgs.msg      import PoseStamped, Twist
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg        import Joy, CompressedImage
from aarapsi_robot_pack.msg import ControllerStateInfo, Label, RequestDataset, ResponseDataset

from pyaarapsi.core.ros_tools               import LogType, pose2xyw, q_from_rpy
from pyaarapsi.core.helper_tools            import formatException
from pyaarapsi.core.enum_tools              import enum_name, enum_value
from pyaarapsi.core.vars                    import C_I_RED, C_I_GREEN, C_I_YELLOW, C_I_BLUE, C_I_WHITE, C_RESET, C_CLEAR, C_UP_N

from pyaarapsi.vpr_simple.vpr_dataset_tool  import VPRDatasetProcessor
from pyaarapsi.vpr_simple.vpr_helpers       import FeatureType

from pyaarapsi.vpr_classes.base             import Base_ROS_Class
from pyaarapsi.core.argparse_tools          import check_positive_float, check_bool, check_string, check_enum, check_positive_int, check_float
from pyaarapsi.pathing.enums                import *
from pyaarapsi.pathing.basic                import *

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
        self.AUTONOMOUS_OVERRIDE    = self.params.add(self.nodespace + "/override/autonomous",      Command_Mode.STOP,  lambda x: check_enum(x, Command_Mode),  force=reset)

    def init_vars(self):
        super().init_vars()

        self.vpr_ego            = []
        self.slam_ego           = []
        self.robot_ego          = []
        self.lookahead          = 1.0
        self.lookahead_mode     = Lookahead_Mode.DISTANCE

        self.old_lin            = 0.0
        self.old_ang            = 0.0
        self.zone_index         = None
        self.return_stage       = Return_Stage.UNSET

        self.plan_path          = Path()
        self.ref_path           = Path()
        self.label              = Label()

        self.ready              = False
        self.new_state_msg      = False
        self.new_robot_ego      = False
        self.new_slam_ego       = False
        self.new_vpr_ego        = False

        self.dataset_queue      = []
        self.dataset_loaded     = False

        self.set_command_mode(Command_Mode.STOP)
        self.set_safety_mode(Safety_Mode.STOP)

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
        
        self.lin_lim            = { Safety_Mode.SLOW: self.SLOW_LIN_VEL_MAX.get(), Safety_Mode.FAST: self.FAST_LIN_VEL_MAX.get() }
        self.ang_lim            = { Safety_Mode.SLOW: self.SLOW_ANG_VEL_MAX.get(), Safety_Mode.FAST: self.FAST_ANG_VEL_MAX.get() }

        self.reject_lambda      = { Reject_Mode.STOP: lambda x: 0.0,        Reject_Mode.OLD: lambda x: x * 1,
                                    Reject_Mode.OLD_50: lambda x: x * 0.5,  Reject_Mode.OLD_90: lambda x: x * 0.9 }

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
        self.state_sub          = rospy.Subscriber( self.namespace + '/state',      Label,                  self.state_cb,      queue_size=1)
        self.robot_odom_sub     = rospy.Subscriber( self.ROBOT_ODOM_TOPIC.get(),    Odometry,               self.robot_odom_cb, queue_size=1) # wheel encoders fused
        self.slam_odom_sub      = rospy.Subscriber( self.SLAM_ODOM_TOPIC.get(),     Odometry,               self.slam_odom_cb,  queue_size=1)
        self.joy_sub            = rospy.Subscriber( self.JOY_TOPIC.get(),           Joy,                    self.joy_cb,        queue_size=1)
        self.timer_chk          = rospy.Timer(rospy.Duration(2), self.check_controller)

        self.sublis.add_operation(self.namespace + '/path',     method_sub=self.path_peer_subscribe)
        self.sublis.add_operation(self.namespace + '/zones',    method_sub=self.path_peer_subscribe)
        self.sublis.add_operation(self.namespace + '/speeds',   method_sub=self.path_peer_subscribe)

    def state_cb(self, msg: Label):
        if not self.ready:
            return
        
        self.vpr_ego            = [msg.vpr_ego.x, msg.vpr_ego.y, msg.vpr_ego.w]
        self.label              = msg
        self.new_vpr_ego        = True
    
    def robot_odom_cb(self, msg: Odometry):
        if not self.ready:
            return

        self.robot_ego          = pose2xyw(msg.pose.pose)
        self.new_robot_ego      = True

    def slam_odom_cb(self, msg: Odometry):
        if not self.ready:
            return
        
        self.slam_ego               = pose2xyw(msg.pose.pose)
        self.new_slam_ego           = True

    def joy_cb(self, msg: Joy):
        if not self.ready:
            return
        
        if abs(rospy.Time.now().to_sec() - msg.header.stamp.to_sec()) > 0.5: # if joy message was generated longer ago than half a second:
            self.set_safety_mode(Safety_Mode.STOP)
            if not self.SIMULATION.get():
                self.print("Bad joy data.", LogType.WARN, throttle=5)
            else:
                self.print("Bad joy data.", LogType.DEBUG, throttle=5)
            return # bad data.

        # Toggle command mode:
        if msg.buttons[self.slam_mode_ind] > 0:
            if not self.command_mode == Command_Mode.SLAM:
                if self.set_command_mode(Command_Mode.SLAM):
                    self.print("Autonomous Commands: SLAM", LogType.WARN)
        elif msg.buttons[self.vpr_mode_ind] > 0:
            if not self.command_mode == Command_Mode.VPR:
                if self.set_command_mode(Command_Mode.VPR):
                    self.print("Autonomous Commands: VPR", LogType.ERROR)
        elif msg.buttons[self.zone_mode_ind] > 0:
            if not self.command_mode == Command_Mode.ZONE_RETURN:
                if self.set_command_mode(Command_Mode.ZONE_RETURN):
                    self.zone_index   = None
                    self.return_stage = Return_Stage.UNSET
                    self.print("Autonomous Commands: Zone Reset", LogType.WARN)
        elif msg.buttons[self.stop_mode_ind] > 0:
            if not self.command_mode == Command_Mode.STOP:
                if self.set_command_mode(Command_Mode.STOP):
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
                if self.set_safety_mode(Safety_Mode.FAST):
                    self.print('Fast mode enabled.', LogType.ERROR)
        elif msg.buttons[self.slow_mode_ind] > 0:
            if not self.safety_mode == Safety_Mode.SLOW:
                if self.set_safety_mode(Safety_Mode.SLOW):
                    self.print('Slow mode enabled.', LogType.WARN)
        else:
            if not self.safety_mode == Safety_Mode.STOP:
                if self.set_safety_mode(Safety_Mode.STOP):
                    self.print('Safety released.', LogType.INFO)

    def set_safety_mode(self, mode: Safety_Mode, override=False):
        if override or self.SAFETY_OVERRIDE.get() == Safety_Mode.UNSET:
            self.safety_mode = mode
            return True
        return False

    def set_command_mode(self, mode: Command_Mode, override=False):
        self.command_mode = mode
        if not override:
            self.AUTONOMOUS_OVERRIDE.set(mode)
        return True

    def check_for_safety_stop(self, lin_err, ang_err):
        if self.command_mode in [Command_Mode.VPR, Command_Mode.SLAM]:
            if lin_err > self.LINSTOP_OVERRIDE.get() and self.LINSTOP_OVERRIDE.get() > 0:
                self.set_command_mode(Command_Mode.STOP)
            elif ang_err > self.ANGSTOP_OVERRIDE.get() and self.ANGSTOP_OVERRIDE.get() > 0:
                self.set_command_mode(Command_Mode.STOP)

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
    
    def make_path(self):
        # generate an n row, 4 column array (x, y, yaw, speed) corresponding to each reference image (same index)
        self.path_xyws  = np.transpose(np.stack([self.ip.dataset['dataset']['px'].flatten(), 
                                                 self.ip.dataset['dataset']['py'].flatten(),
                                                 self.ip.dataset['dataset']['pw'].flatten(),
                                                 make_speed_array(self.ip.dataset['dataset']['pw'].flatten())]))
        
        # determine zone number, length, indices:
        path_sum, path_len               = calc_path_stats(self.path_xyws)
        self.zone_length, self.num_zones = calc_zone_stats(path_len, self.ZONE_LENGTH.get(), self.ZONE_NUMBER.get(), )
        _end                             = [self.path_xyws.shape[0] + (int(not self.LOOP_PATH.get()) - 1)]
        self.zone_indices                = [np.argmin(np.abs(path_sum-(self.zone_length*i))) for i in np.arange(self.num_zones)] + _end
        
        # generate stuff for visualisation:
        self.path_indices                = [np.argmin(np.abs(path_sum-(0.2*i))) for i in np.arange(int(5 * path_len))]
        self.viz_path, self.viz_speeds   = make_path_speeds(self.path_xyws, self.path_indices)
        self.viz_zones                   = make_zones(self.path_xyws, self.zone_indices)
        
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
            if not self.command_mode == self.AUTONOMOUS_OVERRIDE.get():
                self.set_command_mode(self.AUTONOMOUS_OVERRIDE.get(), True)
                self.return_stage = Return_Stage.UNSET
        elif msg.data == self.SAFETY_OVERRIDE.name:
            if not self.SAFETY_OVERRIDE.get() == Safety_Mode.UNSET:
                self.set_safety_mode(self.SAFETY_OVERRIDE.get(), True)
            else:
                self.set_safety_mode(Safety_Mode.STOP)

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
        msg.query_image             = self.label.query_image
        # Extract Label Details:
        msg.dvc                     = self.label.distance_vector
        msg.group.gt_ego            = self.label.gt_ego
        msg.group.vpr_ego           = self.label.vpr_ego
        msg.group.matchId           = self.label.match_index
        msg.group.trueId            = self.label.truth_index
        msg.group.gt_state          = self.label.gt_class
        msg.group.gt_error          = np.round(self.label.gt_error, 3)
        # Extract (remaining) Monitor Details:
        msg.group.mState            = np.round(self.label.svm_z, 3)
        msg.group.prob              = np.round(self.label.svm_prob, 3)
        msg.group.mStateBin         = self.label.svm_class
        msg.group.factors           = [np.round(i,3) for i in self.label.svm_factors]

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

    def print_display(self, new_linear, new_angular, current_ind, speed, error_yaw, zone, lin_path_err, ang_path_err):

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
        base_err_string = ''.join([C_I_YELLOW + i + ': ' + C_I_WHITE + '% 5.2f ' for i in ['VEL', 'eYAW']]) + C_RESET
        base_ind_string = ''.join([C_I_YELLOW + i + ': ' + C_I_WHITE + '%4d ' for i in ['CUR']]) + C_RESET
        base_svm_string = ''.join([C_I_YELLOW + i + ': ' + C_I_WHITE + '%s ' for i in ['OVERRIDE','SVM']]) + C_RESET
        vpr_pos_string  = base_pos_string % tuple(self.vpr_ego)
        slam_pos_string = base_pos_string % tuple(self.slam_ego)
        speed_string    = base_vel_string % (new_linear, new_angular)
        errors_string   = base_err_string % (speed, error_yaw)
        index_string    = base_ind_string % (current_ind)
        path_err_string = base_vel_string % (lin_path_err, ang_path_err)
        svm_string      = base_svm_string % (enum_name(self.REJECT_MODE.get()), str(self.label.svm_class))
        TAB = ' ' * 8
        lines = [
                 '',
                 TAB + '-'*15 + C_I_BLUE + ' STATUS INFO ' + C_RESET + '-'*15,
                 TAB + 'Autonomous Mode: %s' % command_mode_string,
                 TAB + '    Safety Mode: %s' % safety_mode_string,
                 TAB + '   VPR Position: %s' % vpr_pos_string,
                 TAB + '  SLAM Position: %s' % slam_pos_string,
                 TAB + '   Measurements: %s' % errors_string,
                 TAB + ' Speed Commands: %s' % speed_string,
                 TAB + '     Index Info: %s' % index_string,
                 TAB + '    Zone Number: %d' % zone,
                 TAB + '     Path Error: %s' % path_err_string,
                 TAB + '     SVM Status: %s' % svm_string
                ]
        print(''.join([C_CLEAR + line + '\n' for line in lines]) + (C_UP_N%1)*(len(lines)), end='')

    def publish_pose(self, goal_ind: int, pub: rospy.Publisher) -> None:
        # Update visualisation of current goal/target pose
        goal                    = PoseStamped(header=Header(stamp=rospy.Time.now(), frame_id='map'))
        goal.pose.position      = Point(x=self.path_xyws[goal_ind,0], y=self.path_xyws[goal_ind,1], z=0.0)
        goal.pose.orientation   = q_from_yaw(self.path_xyws[goal_ind,2])
        pub.publish(goal)

    def make_new_command(self, speed: float, error_yaw: float) -> Twist:
        # If we're not estimating where we are, we're ignoring the SVM, or we have a good point:
        if not (self.command_mode == Command_Mode.VPR) or self.REJECT_MODE.get() == Reject_Mode.NONE or self.label.svm_class:
            # ... then calculate based on the controller errors:
            new_lin             = np.sign(speed)     * np.min([abs(speed),     self.lin_lim.get(self.safety_mode, 0)])
            new_ang             = np.sign(error_yaw) * np.min([abs(error_yaw), self.ang_lim.get(self.safety_mode, 0)])
        
        # otherwise, if we want to reject the point we received:
        else:
            # ... then we must decide on what new command we should send:
            try:
                new_lin         = self.reject_lambda[self.REJECT_MODE.get()](self.old_lin)
                new_ang         = self.reject_lambda[self.REJECT_MODE.get()](self.old_ang)
            except KeyError:
                raise Exception('Unknown rejection mode %s' % str(self.REJECT_MODE.get()))

        # Update the last-sent-command to be the one we just made:
        self.old_lin            = new_lin
        self.old_ang            = new_ang

        # Generate the actual ROS command:
        new_msg                 = Twist()
        new_msg.linear.x        = new_lin
        new_msg.angular.z       = new_ang
        return new_msg

class Zone_Return_Class(Main_ROS_Class):
    def update_COR(self, ego):
        # Update centre-of-rotation for visualisation and precise alignment:
        COR_x                   = ego[0] + self.COR_OFFSET.get() * np.cos(ego[2])
        COR_y                   = ego[1] + self.COR_OFFSET.get() * np.sin(ego[2])
        pose                    = PoseStamped(header=Header(stamp=rospy.Time.now(), frame_id='map'))
        pose.pose.position      = Point(x=COR_x, y=COR_y)
        pose.pose.orientation   = q_from_rpy(0, -np.pi/2, 0)
        self.COR_pub.publish(pose)
        return [COR_x, COR_y]
    
    def zone_return(self, ego, current_ind):
        '''
        Handle an autonomous return-to-zone:
        - Picks nearest target
        - Drives to target
        - Turns on-spot to face correctly

        Stages:
        Return_STAGE.UNSET: Stage 0 - New request: identify zone target.
        Return_STAGE.DIST:  Stage 1 - Distance from target exceeds 5cm: head towards target.
        Return_STAGE.TURN:  Stage 2 - Heading error exceeds 1 degree: turn on-the-spot towards target heading.
        Return_STAGE.DONE:  Stage 3 - FINISHED.

        '''

        if self.return_stage == Return_Stage.DONE:
            return
        
        # If stage 0: determine target and move to stage 1
        if self.return_stage == Return_Stage.UNSET:
            self.zone_index  = self.zone_indices[np.argmin(m2m_dist(current_ind, np.transpose(np.matrix(self.zone_indices))))] % self.path_xyws.shape[0]
            self.return_stage = Return_Stage.DIST

        self.publish_pose(self.zone_index, self.goal_pub)

        yaw_err     = calc_yaw_error(ego, self.path_xyws[:,0], self.path_xyws[:,1], target_ind=self.zone_index)
        ego_cor     = self.update_COR(ego) # must improve accuracy of centre-of-rotation as we do on-the-spot turns
        dist        = np.sqrt(np.square(ego_cor[0]-self.path_xyws[self.zone_index, 0]) + np.square(ego_cor[1]-self.path_xyws[self.zone_index, 1]))
        head_err    = self.path_xyws[self.zone_index, 2] - ego[2]

        # If stage 1: calculate distance to target (lin_err) and heading error (ang_err)
        if self.return_stage == Return_Stage.DIST:
            if abs(yaw_err) < np.pi/6:
                ang_err = np.sign(yaw_err) * np.max([0.1, -0.19*abs(yaw_err)**2 + 0.4*abs(yaw_err) - 0.007])
                lin_err = np.max([0.1, -(1/3)*dist**2 + (19/30)*dist - 0.06])
            else:
                lin_err = 0
                ang_err = np.sign(yaw_err) * 0.2

            # If we're within 5 cm of the target, stop and move to stage 2
            if dist < 0.05:
                self.return_stage = Return_Stage.TURN

        # If stage 2: calculate heading error and turn on-the-spot
        elif self.return_stage == Return_Stage.TURN:
            lin_err = 0
            # If heading error is less than 1 degree, stop and move to stage 3
            if abs(head_err) < np.pi/180:
                self.set_command_mode(Command_Mode.STOP)
                self.return_stage = Return_Stage.DONE
                ang_err = 0
                return # DONE! :)
            else:
                ang_err = np.sign(head_err) * np.max([0.1, abs(head_err)])
        else:
            raise Exception('Bad return stage [%s].' % str(self.return_stage))

        new_msg     = self.make_new_command(speed=lin_err, error_yaw=ang_err)
        self.cmd_pub.publish(new_msg)
