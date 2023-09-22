import rospy
import numpy as np
import sys
import pydbus
import cv2
import matplotlib.pyplot as plt

from rospy_message_converter import message_converter

from nav_msgs.msg           import Path, Odometry
from std_msgs.msg           import String
from geometry_msgs.msg      import Twist, Pose, PoseStamped, PoseWithCovarianceStamped
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg        import Joy, CompressedImage
from aarapsi_robot_pack.msg import ControllerStateInfo, Label, RequestDataset, ResponseDataset, ExpResults, GoalExpResults, xyw

from gazebo_msgs.msg        import ModelState
from gazebo_msgs.srv        import SetModelState, SetModelStateRequest

from pyaarapsi.core.ros_tools               import LogType, pose2xyw, q_from_rpy, twist2xyw, np2compressed, compressed2np
from pyaarapsi.core.helper_tools            import formatException, p2p_dist_2d, roll
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

        # Zone set-up:
        self.ZONE_LENGTH            = self.params.add(self.namespace + "/path/zones/length",        None,                   check_positive_float,                       force=False)
        self.ZONE_NUMBER            = self.params.add(self.namespace + "/path/zones/number",        None,                   check_positive_int,                         force=False)
        
        # Vehicle speed limits:
        self.SLOW_LIN_VEL_MAX       = self.params.add(self.namespace + "/limits/slow/linear",       None,                   check_positive_float,                       force=False)
        self.SLOW_ANG_VEL_MAX       = self.params.add(self.namespace + "/limits/slow/angular",      None,                   check_positive_float,                       force=False)
        self.FAST_LIN_VEL_MAX       = self.params.add(self.namespace + "/limits/fast/linear",       None,                   check_positive_float,                       force=False)
        self.FAST_ANG_VEL_MAX       = self.params.add(self.namespace + "/limits/fast/angular",      None,                   check_positive_float,                       force=False)
        
        # Vehicle Config and Communication:
        self.COR_OFFSET             = self.params.add(self.namespace + "/cor_offset",               0.045,                  check_float,                                force=False)
        self.CONTROLLER_MAC         = self.params.add(self.namespace + "/controller_mac",           None,                   check_string,                               force=False)
        self.JOY_TOPIC              = self.params.add(self.namespace + "/joy_topic",                None,                   check_string,                               force=False)
        self.CMD_TOPIC              = self.params.add(self.namespace + "/cmd_topic",                None,                   check_string,                               force=False)

        self.PUBLISH_ROLLMATCH      = self.params.add(self.nodespace + "/publish_rollmatch",        True,                   check_bool,                                 force=reset)
        self.REJECT_MODE            = self.params.add(self.nodespace + "/reject_mode",              Reject_Mode.OLD,        lambda x: check_enum(x, Reject_Mode),       force=reset)
        self.LOOP_PATH              = self.params.add(self.nodespace + "/loop_path",                True,                   check_bool,                                 force=reset)
        self.PRINT_DISPLAY          = self.params.add(self.nodespace + "/print_display",            True,                   check_bool,                                 force=reset)
        
        # Command overrides:
        self.LINSTOP_OVERRIDE       = self.params.add(self.nodespace + "/override/lin_error",       0.4,                    check_positive_float,                       force=reset)
        self.ANGSTOP_OVERRIDE       = self.params.add(self.nodespace + "/override/ang_error",       80*np.pi/180,           check_positive_float,                       force=reset)
        self.SAFETY_OVERRIDE        = self.params.add(self.nodespace + "/override/safety",          Safety_Mode.UNSET,      lambda x: check_enum(x, Safety_Mode),       force=reset)
        self.AUTONOMOUS_OVERRIDE    = self.params.add(self.nodespace + "/override/autonomous",      Command_Mode.STOP,      lambda x: check_enum(x, Command_Mode),      force=reset)

        # Experiment-specific Parameters:
        self.SLICE_LENGTH           = self.params.add(self.nodespace + "/exp/slice_length",         1.5,                    check_positive_float,                       force=reset)
        self.APPEND_DIST            = self.params.add(self.nodespace + "/exp/append_dist",          0.05,                   check_positive_float,                       force=reset)
        self.APPEND_MAX             = self.params.add(self.nodespace + "/exp/append_max",           50,                     check_positive_int,                         force=reset)
        self.EXPERIMENT_MODE        = self.params.add(self.nodespace + "/exp/mode",                 Experiment_Mode.UNSET,  lambda x: check_enum(x, Experiment_Mode),   force=reset)
        self.TECHNIQUE              = self.params.add(self.nodespace + "/exp/technique",            Technique.VPR,          lambda x: check_enum(x, Technique),         force=reset)
    
    def init_vars(self):
        super().init_vars() # Call base class method

        # Vehicle state information:
        self.ego                = []                        # ego to be used
        self.vpr_ego            = []                        # ego from VPR
        self.slam_ego           = []                        # ego from SLAM gt
        self.robot_ego          = []                        # ego from robot odometry / wheel encoders
        self.robot_velocities   = []                        # velocities from robot odometry / wheel encoders
        self.lookahead          = 1.0                       # Lookahead amount
        self.adjusted_lookahead = 0.0                       # Speed-adjusted lookahead distance
        self.lookahead_mode     = Lookahead_Mode.DISTANCE   # Lookahead mode

        # Inter-loop variables for velocity control:
        self.old_lin            = 0.0                       # Last-made-command's linear velocity
        self.old_ang            = 0.0                       # Last-made-command's angular velocity

        # Inter-loop variables for Zone-Return Features:
        self.zone_index         = -1                        # Index in path dataset corresponding to target zone
        self.return_stage       = Return_Stage.UNSET        # Enum to control flow of autonomous zone return commands
        self.saved_index        = -1                        # Index in path dataset user has selected to store
        self.saved_pose         = None                      # Saved position; in simulation, can teleport here.
        self.save_request       = Save_Request.NONE         # Enum to command setting/clearing of self.saved_index

        # Inter-loop variables for historical data management:
        self.match_hist         = []                        # Core historical data array
        self.new_history        = False                     # Whether new historical information has been appended
        self.current_hist_pos   = [None]*3                  # Most recent position estimate from historical data

        # Inter-loop variables for dataset loading control:
        self.dataset_queue      = []                        # List of dataset parameters pending construction
        self.dataset_loaded     = False                     # Whether datasets are ready

        # Empty structures to initialise memory requirements:
        self.viz_path           = Path()
        self.viz_speeds         = MarkerArray()
        self.viz_zones          = MarkerArray()
        self.label              = Label()
        self.last_command       = Twist()

        # Flags for loop progression control:
        self.ready              = False
        self.new_label          = False

        # Initialise dataset processor:
        self.ip                 = VPRDatasetProcessor(None, try_gen=False, ros=True, printer=self.print) 
        self.dsinfo             = self.make_dataset_dict() # Get VPR pipeline's dataset dictionary
        self.dsinfo['ft_types'] = enum_name(FeatureType.RAW, wrap=True) # Ensure feature is raw because we do roll-matching

        # Set initial mode states:
        self.set_command_mode(Command_Mode.STOP)
        self.set_safety_mode(Safety_Mode.STOP)

        # Controller bind information:
        self.vpr_mode_ind       = PS4_Pressed.Square
        self.stop_mode_ind      = PS4_Pressed.X
        self.slam_mode_ind      = PS4_Pressed.O
        self.zone_mode_ind      = PS4_Pressed.Triangle

        self.slow_mode_ind      = PS4_Pressed.LeftBumper
        self.fast_mode_ind      = PS4_Pressed.RightBumper

        self.netvlad_ind        = PS4_Pressed.LeftArrow
        self.hybridnet_ind      = PS4_Pressed.RightArrow
        self.raw_ind            = PS4_Pressed.UpArrow
        self.patchnorm_ind      = PS4_Pressed.DownArrow

        self.lin_cmd_ind        = PS4_Axes.LeftStickYAxis
        self.ang_cmd_ind        = PS4_Axes.RightStickXAxis

        self.store_pose_ind     = PS4_Pressed.Options
        self.clear_pose_ind     = PS4_Pressed.Share

        self.teleport_ind       = PS4_Pressed.RightStickIn

        self.experiment_ind     = PS4_Pressed.PS

        # Hashed entries for compact & fast access:
        self.feature_hash        = { self.raw_ind: FeatureType.RAW,           self.patchnorm_ind: FeatureType.PATCHNORM, 
                                     self.netvlad_ind: FeatureType.NETVLAD,   self.hybridnet_ind: FeatureType.HYBRIDNET }
        
        self.lin_lim_hash       = { Safety_Mode.SLOW: self.SLOW_LIN_VEL_MAX.get(), Safety_Mode.FAST: self.FAST_LIN_VEL_MAX.get() }
        self.ang_lim_hash       = { Safety_Mode.SLOW: self.SLOW_ANG_VEL_MAX.get(), Safety_Mode.FAST: self.FAST_ANG_VEL_MAX.get() }

        self.reject_hash        = { Reject_Mode.STOP: lambda x: 0.0,        Reject_Mode.OLD: lambda x: x * 1,
                                    Reject_Mode.OLD_50: lambda x: x * 0.5,  Reject_Mode.OLD_90: lambda x: x * 0.9 }

        self.command_str_hash   = { Command_Mode.STOP: C_I_GREEN + 'STOPPED',       Command_Mode.SLAM: C_I_YELLOW + 'SLAM mode', 
                                    Command_Mode.VPR: C_I_RED + 'VPR mode',         Command_Mode.ZONE_RETURN: C_I_YELLOW + 'Returning to Zone', 
                                    Command_Mode.SPECIAL: C_I_BLUE + 'Special mode'}

        self.experi_str_hash    = { Experiment_Mode.UNSET: C_I_GREEN + 'STOPPED',           Experiment_Mode.DRIVE_PATH: C_I_YELLOW + 'Path Following', 
                                    Experiment_Mode.INIT:  C_I_YELLOW + 'Initialising',     Experiment_Mode.HALT1:      C_I_YELLOW + 'Halting', 
                                    Experiment_Mode.ALIGN: C_I_YELLOW + 'Align to Start',   Experiment_Mode.DRIVE_GOAL: C_I_RED + 'Heading to Goal', 
                                    Experiment_Mode.HALT2: C_I_YELLOW + 'Halting',          Experiment_Mode.DANGER:     C_I_RED + 'Halting',
                                    Experiment_Mode.DONE:  C_I_GREEN + 'Complete' }

        self.safety_str_hash    = { Safety_Mode.STOP: C_I_GREEN + 'STOPPED', Safety_Mode.SLOW: C_I_YELLOW + 'SLOW mode', 
                                    Safety_Mode.FAST: C_I_RED + 'FAST mode', }
        
        # Path variables:
        self.path_xyws          = None      # Large numpy array of n rows by four columns (x, y, yaw, speed)
        self.path_len           = None      # Circumferential Path length
        self.path_sum           = []        # List of n elements with running sum of distance since first path position (zero to self.path_len)
        self.path_indices       = []        # Downsampled selection of self.path_xyws for visualisation

        # Zone variables:
        self.zone_length        = -1        # In metres, length of a zone after adjustment
        self.num_zones          = -1        # Number of zones after adjustment
        self.zone_indices       = []        # List of (self.num_zones + 1) elements, path indices for zone boundaries
        self.slam_zone          = -1        # Current zone as per SLAM

        # Experiment variables:
        self.exp_start_SLAM     = None      # Index in path dataset experiment will start SLAM
        self.exp_stop_SLAM      = None      # Index in path dataset experiment will stop SLAM
        self.exp_dist           = None      # Adjusted length of experiment (in case user starts too close to reference set edges)
        self.new_goal           = False     # Whether a new 2D Nav Goal has been requested
        self.goal_pose          = [0]*3     # Position of the 2D Nav Goal
        self.exp_transform      = None      # Numpy array Homogenous Transform
        self.exp_rotation       = 0         # 2D pose transformation correction
        self.point_shoot_stage  = Point_Shoot_Stage.UNSET
        self.point_shoot_start  = [None]*3  # Initial position to measure relative to
        self.point_shoot_point  = 0         # Angle to correct
        self.point_shoot_shoot  = 0         # Distance to cover
        self.current_results    = ExpResults()
        self.exp_results        = GoalExpResults()
        self.exp_count          = 0
        
        # General loop variables:
        self.est_current_ind    = -1        # Estimated path index closest to robot
        self.slam_current_ind   = -1        # SLAM ground truth index closest to robot
        self.target_ind         = -1        # Index at target; est_current_ind + adjusted lookahead distance.
        self.path_lin_err       = -1        # Path latitudinal error
        self.path_ang_err       = -1        # Path heading error

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
        self.path_pub           = self.add_pub(      self.namespace + '/path',                 Path,                                       queue_size=1, latch=True, subscriber_listener=self.sublis)
        self.speed_pub          = self.add_pub(      self.namespace + '/speeds',               MarkerArray,                                queue_size=1, latch=True, subscriber_listener=self.sublis)
        self.zones_pub          = self.add_pub(      self.namespace + '/zones',                MarkerArray,                                queue_size=1, latch=True, subscriber_listener=self.sublis)
        self.COR_pub            = self.add_pub(      self.namespace + '/cor',                  PoseStamped,                                queue_size=1)
        self.goal_pub           = self.add_pub(      self.namespace + '/path_goal',            PoseStamped,                                queue_size=1)
        self.slam_pub           = self.add_pub(      self.namespace + '/slam_pose',            PoseStamped,                                queue_size=1)
        self.cmd_pub            = self.add_pub(      self.CMD_TOPIC.get(),                     Twist,                                      queue_size=1)
        self.info_pub           = self.add_pub(      self.nodespace + '/info',                 ControllerStateInfo,                        queue_size=1)
        self.rollmatch_pub      = self.add_pub(      self.nodespace + '/rollmatch/compressed', CompressedImage,                            queue_size=1)
        self.init_pose_pub      = self.add_pub(      '/initialpose',                           PoseWithCovarianceStamped,                  queue_size=1)
        self.teleport_pose_pub  = self.add_pub(      self.nodespace + '/teleport/pose',        PoseStamped,                                queue_size=1, latch=True, subscriber_listener=self.sublis)
        self.teleport_index_pub = self.add_pub(      self.nodespace + '/teleport/index',       PoseStamped,                                queue_size=1, latch=True, subscriber_listener=self.sublis)
        self.experi_goal_pub    = self.add_pub(      self.nodespace + '/exp/goal',             PoseStamped,                                queue_size=1, latch=True, subscriber_listener=self.sublis)
        self.experi_start_pub   = self.add_pub(      self.nodespace + '/exp/start',            PoseStamped,                                queue_size=1, latch=True, subscriber_listener=self.sublis)
        self.experi_finish_pub  = self.add_pub(      self.nodespace + '/exp/finish',           PoseStamped,                                queue_size=1, latch=True, subscriber_listener=self.sublis)
        self.experi_pos_pub     = self.add_pub(      self.nodespace + '/exp/result',           ExpResults,                                 queue_size=1)
        self.experi_results_pub = self.add_pub(      self.nodespace + '/exp/goal_results',     GoalExpResults,                             queue_size=1)
        self.ds_requ_pub        = self.add_pub(      ds_requ + "request",                      RequestDataset,                             queue_size=1)
        self.ds_requ_sub        = rospy.Subscriber(  ds_requ + "ready",                        ResponseDataset,           self.ds_requ_cb, queue_size=1)
        self.goal_sub           = rospy.Subscriber(  '/move_base_simple/goal',                 PoseStamped,               self.goal_cb,    queue_size=1)
        self.state_sub          = rospy.Subscriber(  self.namespace + '/state',                Label,                     self.state_cb,   queue_size=1)
        self.joy_sub            = rospy.Subscriber(  self.JOY_TOPIC.get(),                     Joy,                       self.joy_cb,     queue_size=1)
        self.velo_sub           = rospy.Subscriber(  self.ROBOT_ODOM_TOPIC.get(),              Odometry,                  self.velo_cb,    queue_size=1)
        self.teleport_srv       = rospy.ServiceProxy('/gazebo/set_model_state',                SetModelState)
        self.timer_chk          = rospy.Timer(rospy.Duration(2), self.check_controller)

        self.sublis.add_operation(self.namespace + '/path',             method_sub=self.path_peer_subscribe)
        self.sublis.add_operation(self.namespace + '/zones',            method_sub=self.path_peer_subscribe)
        self.sublis.add_operation(self.namespace + '/speeds',           method_sub=self.path_peer_subscribe)
        self.sublis.add_operation(self.nodespace + '/exp/goal',         method_sub=self.path_peer_subscribe)
        self.sublis.add_operation(self.nodespace + '/teleport/pose',    method_sub=self.path_peer_subscribe)
        self.sublis.add_operation(self.nodespace + '/teleport/index',   method_sub=self.path_peer_subscribe)
        self.sublis.add_operation(self.nodespace + '/exp/start',        method_sub=self.path_peer_subscribe)
        self.sublis.add_operation(self.nodespace + '/exp/finish',       method_sub=self.path_peer_subscribe)

    def teleport(self, pos=None, vel=None, model='jackal', frame_id='map'):
        if not self.SIMULATION.get():
            self.print('Cannot teleport outside of a simulation environment!', LogType.ERROR)
            return
        
        twist = Twist()
        if not (vel is None):
            twist.linear.x  = vel[0]
            twist.angular.z = vel[1]

        pose  = Pose()
        if pos is None:
            pose.orientation = q_from_yaw(0)
        else:
            pose.position.x  = pos[0]
            pose.position.y  = pos[1]
            pose.orientation = q_from_yaw(pos[2])


        model_state                 = ModelState(model_name=model, pose=pose, twist=twist, reference_frame=frame_id)
        init_pose                   = PoseWithCovarianceStamped()
        init_pose.pose.pose         = pose
        init_pose.header.stamp      = rospy.Time.now()
        init_pose.header.frame_id   = frame_id
        resp = self.teleport_srv.call(SetModelStateRequest(model_state=model_state))
        if resp.success:
            self.init_pose_pub.publish(init_pose)

    def velo_cb(self, msg: Odometry):
        '''
        Callback to store robot wheel encoder velocities
        '''
        if not self.ready:
            return
        self.robot_velocities = twist2xyw(msg.twist.twist)
        
    def ds_requ_cb(self, msg: ResponseDataset):
        '''
        Dataset request callback; handle confirmation of dataset readiness
        '''
        if msg.success == False:
            self.print('Dataset request processed, error. Parameters: %s' % str(msg.params), LogType.ERROR)
        try:
            index = self.dataset_queue.index(msg.params) # on separate line to try trigger ValueError failure
            self.print('Dataset request processed, success. Removing from dataset queue.')
            self.dataset_queue.pop(index)
        except ValueError:
            pass

    def goal_cb(self, msg: PoseStamped):
        '''
        Callback to handle new 2D Nav Goal requests
        '''
        if not self.ready:
            return
        
        self.goal_pose      = pose2xyw(msg.pose)
        self.new_goal       = True
        publish_xyzrpy_pose([self.goal_pose[0], self.goal_pose[1], -0.5, 0, -np.pi/2, 0], self.experi_goal_pub)

    def state_cb(self, msg: Label):
        '''
        Callback to handle new labels from the VPR pipeline
        '''
        if not self.ready:
            return
        
        self.label              = msg
        
        self.vpr_ego            = [msg.vpr_ego.x, msg.vpr_ego.y, msg.vpr_ego.w]
        self.robot_ego          = [msg.robot_ego.x, msg.robot_ego.y, msg.robot_ego.w]
        self.slam_ego           = [msg.gt_ego.x, msg.gt_ego.y, msg.gt_ego.w]
        self.new_label          = True
        
        # Update historical data:
        # Order defined in enumeration HDi
        _append = [msg.match_index, msg.truth_index, msg.distance_vector[msg.match_index], msg.gt_class, msg.svm_class, *self.slam_ego, *self.robot_ego, *self.vpr_ego]
        if not len(self.match_hist): # If this is the first entry,
            self.match_hist.append(_append + [0])
            self.new_history = True
            return
        
        _dist = p2p_dist_2d(self.match_hist[-1][enum_value(HDi.robot_x):enum_value(HDi.robot_y)+1], self.robot_ego[0:2])
        if _dist > self.APPEND_DIST.get():
            self.match_hist.append(_append + [_dist])
            self.new_history = True

        while len(self.match_hist) > self.APPEND_MAX.get():
            self.match_hist.pop(0)

    def joy_cb(self, msg: Joy):
        '''
        Callback to handle user input on controller
        '''
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
        if self.slam_mode_ind(msg):
            if not self.command_mode == Command_Mode.SLAM:
                if self.set_command_mode(Command_Mode.SLAM):
                    self.print("Autonomous Commands: SLAM", LogType.WARN)
        elif self.vpr_mode_ind(msg):
            if not self.command_mode == Command_Mode.VPR:
                if self.set_command_mode(Command_Mode.VPR):
                    self.print("Autonomous Commands: VPR", LogType.ERROR)
        elif self.zone_mode_ind(msg):
            if not self.command_mode == Command_Mode.ZONE_RETURN:
                if self.set_command_mode(Command_Mode.ZONE_RETURN):
                    self.zone_index   = None
                    self.return_stage = Return_Stage.UNSET
                    self.print("Autonomous Commands: Zone Reset", LogType.WARN)
        elif self.stop_mode_ind(msg):
            if self.set_command_mode(Command_Mode.STOP):
                self.print("Autonomous Commands: Disabled", LogType.INFO)

        # Start experiment:
        if self.experiment_ind(msg):
            if self.EXPERIMENT_MODE.get() == Experiment_Mode.UNSET:
                self.EXPERIMENT_MODE.set(Experiment_Mode.INIT)
                self.print("Experiment: Starting.", LogType.WARN)
            elif not (self.EXPERIMENT_MODE.get() == Experiment_Mode.INIT):
                self.EXPERIMENT_MODE.set(Experiment_Mode.INIT)
                self.print("Experiment: Resetting.", LogType.WARN)

        # Toggle store/clear specific zone:
        if self.store_pose_ind(msg): # If a user wants to assign a zone:
            self.save_request = Save_Request.SET
        elif self.clear_pose_ind(msg): # If a user wants to clear a zone:
            self.save_request = Save_Request.CLEAR

        # Check if teleport request:
        if self.teleport_ind(msg): # If a user wants to teleport:
            self.teleport(pos=self.saved_pose)

        # Toggle feature type:
        try:
            for i in self.feature_hash.keys():
                if i(msg) and (not self.FEAT_TYPE.get() == self.feature_hash[i]):
                    rospy.set_param(self.namespace + '/feature_type', enum_name(FeatureType.RAW))
                    self.print("Switched to %s." % enum_name(self.FEAT_TYPE.get()), LogType.INFO)
                    break
        except IndexError:
            pass

        # Toggle speed safety mode:
        if self.fast_mode_ind(msg):
            if not self.safety_mode == Safety_Mode.FAST:
                if self.set_safety_mode(Safety_Mode.FAST):
                    self.print('Fast mode enabled.', LogType.ERROR)
        elif self.slow_mode_ind(msg):
            if not self.safety_mode == Safety_Mode.SLOW:
                if self.set_safety_mode(Safety_Mode.SLOW):
                    self.print('Slow mode enabled.', LogType.WARN)
        else:
            if not self.safety_mode == Safety_Mode.STOP:
                if self.set_safety_mode(Safety_Mode.STOP):
                    self.print('Safety released.', LogType.INFO)

    def path_peer_subscribe(self, topic_name: str):
        '''
        Callback to handle subscription events
        '''
        if not self.ready:
            return
        if topic_name == self.namespace + '/path':
            self.path_pub.publish(self.viz_path)

        elif topic_name == self.namespace + '/zones':
            self.zones_pub.publish(self.viz_zones)

        elif topic_name == self.namespace + '/speeds':
            self.speed_pub.publish(self.viz_speeds)

        elif topic_name == self.nodespace + '/exp/goal':
            publish_xyzrpy_pose([self.goal_pose[0], self.goal_pose[1], -0.5, 0, -np.pi/2, 0], self.experi_goal_pub)

        elif topic_name == self.nodespace + '/teleport/pose':
            if not (self.saved_pose is None):
                publish_xyzrpy_pose([self.saved_pose[0], self.saved_pose[1], -0.5, 0, -np.pi/2, 0], self.teleport_pose_pub)
            else:
                publish_xyzrpy_pose([0, 0, -0.5, 0, -np.pi/2, 0], self.teleport_pose_pub)

        elif topic_name == self.nodespace + '/teleport/index':
            if not (self.saved_index == -1):
                publish_xyzrpy_pose([self.path_xyws[self.saved_index,0], self.path_xyws[self.saved_index,1], -0.5, 0, -np.pi/2, 0], self.teleport_index_pub)
            else:
                publish_xyzrpy_pose([0, 0, -0.5, 0, -np.pi/2, 0], self.teleport_index_pub)

        elif topic_name == self.nodespace + '/exp/start':
            if not (self.exp_start_SLAM is None):
                publish_xyzrpy_pose([self.path_xyws[self.exp_start_SLAM,0], self.path_xyws[self.exp_start_SLAM,1], -0.5, 0, -np.pi/2, 0], self.experi_start_pub)
            else:
                publish_xyzrpy_pose([0, 0, -0.5, 0, -np.pi/2, 0], self.experi_start_pub)

        elif topic_name == self.nodespace + '/exp/finish':
            if not (self.exp_stop_SLAM is None):
                publish_xyzrpy_pose([self.path_xyws[self.exp_stop_SLAM,0], self.path_xyws[self.exp_stop_SLAM,1], -0.5, 0, -np.pi/2, 0], self.experi_finish_pub)
            else:
                publish_xyzrpy_pose([0, 0, -0.5, 0, -np.pi/2, 0], self.experi_finish_pub)
        else:
            raise Exception('Unknown path_peer_subscribe topic: %s' % str(topic_name))

    def set_safety_mode(self, mode: Safety_Mode, override=False):
        '''
        Manage changes to safety mode and sync with ROS parameter server
        '''
        if override or self.SAFETY_OVERRIDE.get() == Safety_Mode.UNSET:
            self.safety_mode = mode
            return True
        return False

    def set_command_mode(self, mode: Command_Mode, override=False):
        '''
        Manage changes to command mdoe and sync with ROS parameter server
        '''
        if mode == Command_Mode.STOP:
            if self.EXPERIMENT_MODE.get() == Experiment_Mode.DRIVE_GOAL:
                self.EXPERIMENT_MODE.set(Experiment_Mode.DANGER)
            else:
                self.EXPERIMENT_MODE.set(Experiment_Mode.UNSET)
        self.command_mode = mode
        if not override:
            self.AUTONOMOUS_OVERRIDE.set(mode)
        return True

    def check_for_safety_stop(self):
        '''
        Ensure vehicle is safely on-path
        '''
        self.path_lin_err, self.path_ang_err = calc_path_errors(self.slam_ego, self.slam_current_ind, self.path_xyws)
        if self.path_lin_err > self.LINSTOP_OVERRIDE.get() and self.LINSTOP_OVERRIDE.get() > 0:
            self.set_command_mode(Command_Mode.STOP)
            return False
        elif self.path_ang_err > self.ANGSTOP_OVERRIDE.get() and self.ANGSTOP_OVERRIDE.get() > 0:
            self.set_command_mode(Command_Mode.STOP)
            return False
        return True
    
    def make_path(self):
        '''
        Generate:
        - Zone information 
        - Downsampled list of path points
        - ROS structures for visualising path, speed along path, and zone boundaries
        '''
        # generate an n-row, 4 column array (x, y, yaw, speed) corresponding to each reference image (same index)
        self.path_xyws  = np.transpose(np.stack([self.ip.dataset['dataset']['px'].flatten(), 
                                                 self.ip.dataset['dataset']['py'].flatten(),
                                                 self.ip.dataset['dataset']['pw'].flatten(),
                                                 make_speed_array(self.ip.dataset['dataset']['pw'].flatten())]))
        
        # determine zone number, length, indices:
        self.path_sum, self.path_len     = calc_path_stats(self.path_xyws)
        self.zone_length, self.num_zones = calc_zone_stats(self.path_len, self.ZONE_LENGTH.get(), self.ZONE_NUMBER.get(), )
        _end                             = [self.path_xyws.shape[0] + (int(not self.LOOP_PATH.get()) - 1)]
        self.zone_indices                = [np.argmin(np.abs(self.path_sum-(self.zone_length*i))) for i in np.arange(self.num_zones)] + _end
        
        # generate stuff for visualisation:
        self.path_indices                = [np.argmin(np.abs(self.path_sum-(0.2*i))) for i in np.arange(int(5 * self.path_len))]
        self.viz_path, self.viz_speeds   = make_path_speeds(self.path_xyws, self.path_indices)
        self.viz_zones                   = make_zones(self.path_xyws, self.zone_indices)

    def param_helper(self, msg: String):
        '''
        Callback subfunction to provide hook-in to parameter server updates
        '''
        # If Autonomous Override ROS parameter has been updated:
        if msg.data == self.AUTONOMOUS_OVERRIDE.name: 
            if not self.command_mode == self.AUTONOMOUS_OVERRIDE.get():
                self.set_command_mode(self.AUTONOMOUS_OVERRIDE.get(), True)
                self.return_stage = Return_Stage.UNSET
        
        # If Safety Override ROS parameter has been updated
        elif msg.data == self.SAFETY_OVERRIDE.name:
            if not self.SAFETY_OVERRIDE.get() == Safety_Mode.UNSET:
                self.set_safety_mode(self.SAFETY_OVERRIDE.get(), True)
            else:
                self.set_safety_mode(Safety_Mode.STOP)

    def try_load_dataset(self, dataset_dict):
        '''
        Try-except wrapper for load_dataset
        '''
        if not self.dataset_loaded:
            try:
                self.ip.load_dataset(dataset_dict)
                self.dataset_loaded = True
            except:
                self.dataset_loaded = False

    def load_dataset(self):
        '''
        Load in dataset to generate path and to utilise VPR index information
        '''

        # Try load in dataset:
        self.try_load_dataset(self.dsinfo)

        if not self.dataset_loaded: # if the model failed to generate, the dataset is not ready, therefore...
            # Request dataset generation:
            dataset_msg = message_converter.convert_dictionary_to_ros_message('aarapsi_robot_pack/RequestDataset', self.dsinfo)
            self.dataset_queue.append(dataset_msg)
            self.ds_requ_pub.publish(self.dataset_queue[0])

            # Wait for news of dataset generation:
            wait_intervals = 0
            while len(self.dataset_queue): # while there is a dataset we are waiting on:
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

            # Try load in the dataset again now that it is ready
            self.try_load_dataset(self.dsinfo)
            if not self.dataset_loaded:
                raise Exception('Datasets were constructed, but could not be loaded!')

    def publish_controller_info(self):
        '''
        Publish controller information to ROS topic
        '''
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

        msg.group.current_yaw       = np.round(self.ego[2], 3)
        msg.group.target_yaw        = np.round(self.path_xyws[self.target_ind,2], 3)

        msg.group.current_ind       = self.est_current_ind
        msg.group.target_ind        = self.target_ind
        msg.group.reject_mode       = enum_name(self.REJECT_MODE.get())

        msg.group.true_yaw          = np.round(self.slam_ego[2], 3)
        msg.group.delta_yaw         = np.round(self.slam_ego[2] - self.vpr_ego[2], 3)

        msg.group.lookahead         = self.adjusted_lookahead
        msg.group.lookahead_mode    = enum_name(self.lookahead_mode)

        msg.group.zone_indices      = self.zone_indices
        msg.group.zone_length       = self.zone_length
        msg.group.zone_count        = self.num_zones
        msg.group.zone_current      = self.slam_zone

        self.info_pub.publish(msg)

    def check_controller(self, event=None):
        '''
        Check if controller is still connected via bluetooth
        '''
        if self.SIMULATION.get():
            return True
        mngd_objs = self.mngr.GetManagedObjects()
        for path in mngd_objs:
            if mngd_objs[path].get('org.bluez.Device1', {}).get('Connected', False):
                if str(mngd_objs[path].get('org.bluez.Device1', {}).get('Address')) == self.CONTROLLER_MAC.get():
                    return True
        self.print('Bluetooth controller not found! Shutting down.', LogType.FATAL)
        sys.exit()

    def print_display(self):
        '''
        Construct and print a Human-Machine Interface
        '''
        if self.EXPERIMENT_MODE.get() == Experiment_Mode.UNSET:
            automation_mode_string = '  Command Mode: ' + self.command_str_hash[self.command_mode] + C_RESET
        else:
            automation_mode_string = '  [' + C_I_BLUE + 'Experiment' + C_RESET + ']: ' + self.experi_str_hash[self.EXPERIMENT_MODE.get()] + C_RESET
        safety_mode_string  = self.safety_str_hash[self.safety_mode] + C_RESET

        base_pos_string = ''.join([C_I_YELLOW + i + ': ' + C_I_WHITE + '% 4.1f ' for i in 'xyw']) + C_RESET
        base_vel_string = ''.join([C_I_YELLOW + i + ': ' + C_I_WHITE + '% 4.1f ' for i in ['LIN','ANG']]) + C_RESET
        base_svm_string = ''.join([C_I_YELLOW + i + ': ' + C_I_WHITE + '%s ' for i in ['REJ','ON']]) + C_RESET
        vpr_pos_string  = base_pos_string % tuple(self.vpr_ego)
        slam_pos_string = base_pos_string % tuple(self.slam_ego)
        speed_string    = base_vel_string % (self.last_command.linear.x, self.last_command.angular.z)
        path_err_string = base_vel_string % (self.path_lin_err, self.path_ang_err)
        svm_string      = base_svm_string % (enum_name(self.REJECT_MODE.get()), str(self.label.svm_class))
        lines = ['',
                 ' ' + '-'*13 + C_I_BLUE + ' STATUS INFO ' + C_RESET + '-'*13,
                 ' ' + automation_mode_string,
                 ' ' + '   Safety Mode: %s' % safety_mode_string,
                 ' ' + '  VPR Position: %s' % vpr_pos_string,
                 ' ' + ' SLAM Position: %s' % slam_pos_string,
                 ' ' + '      Commands: %s' % speed_string,
                 ' ' + '     VPR Index: %d' % self.est_current_ind,
                 ' ' + '   Zone Number: %d' % self.slam_zone,
                 ' ' + '    Path Error: %s' % path_err_string,
                 ' ' + '    SVM Status: %s' % svm_string]
        print(''.join([C_CLEAR + line + '\n' for line in lines]) + (C_UP_N%1)*(len(lines)), end='')

    def try_send_command(self, error_lin: float, error_ang: float) -> Twist:
        '''
        Generate linear and angular commands; send to ROS if a safety is enabled. 
        '''
        # If we're not estimating where we are, we're ignoring the SVM, or we have a good point:
        if not (self.command_mode == Command_Mode.VPR) or self.REJECT_MODE.get() == Reject_Mode.NONE or self.label.svm_class:
            # ... then calculate based on the controller errors:
            new_lin             = np.sign(error_lin) * np.min([abs(error_lin), self.lin_lim_hash.get(self.safety_mode, 0)])
            new_ang             = np.sign(error_ang) * np.min([abs(error_ang), self.ang_lim_hash.get(self.safety_mode, 0)])
        
        # otherwise, if we want to reject the point we received:
        else:
            new_lin             = self.reject_hash[self.REJECT_MODE.get()](self.old_lin)
            new_ang             = self.reject_hash[self.REJECT_MODE.get()](self.old_ang)

        # Update the last-sent-command to be the one we just made:
        self.old_lin            = new_lin
        self.old_ang            = new_ang

        # Generate the actual ROS command:
        new_msg                 = Twist()
        new_msg.linear.x        = new_lin
        new_msg.angular.z       = new_ang

        # Store for reference    
        self.last_command       = new_msg

        # If a safety is 'enabled':
        if self.safety_mode in [Safety_Mode.SLOW, Safety_Mode.FAST]:
            self.cmd_pub.publish(new_msg)
            return True
        
        return False

    def roll_match(self, ind: int):
        '''
        Sliding match of a downsampled image to generate a yaw correction estimate
        '''
        resize          = [int(self.IMG_HFOV.get()), 8]
        img_dims        = self.IMG_DIMS.get()
        query_raw       = cv2.cvtColor(compressed2np(self.label.query_image), cv2.COLOR_BGR2GRAY)
        img             = cv2.resize(query_raw, resize)
        img_mask        = np.ones(img.shape)

        _b              = int(resize[0] / 2)
        sliding_options = range((-_b) + 1, _b)

        against_image   = cv2.resize(np.reshape(self.ip.dataset['dataset']['RAW'][ind], [img_dims[1], img_dims[0]]), resize)
        options_stacked = np.stack([roll(against_image, i).flatten() for i in sliding_options])
        img_stacked     = np.stack([(roll(img_mask, i)*img).flatten() for i in sliding_options])
        matches         = np.sum(np.square(img_stacked - options_stacked),axis=1)
        yaw_fix_deg     = sliding_options[np.argmin(matches)]
        yaw_fix_rad     = normalize_angle(yaw_fix_deg * np.pi / 180.0)

        if self.PUBLISH_ROLLMATCH.get():
            fig, ax = plt.subplots()
            ax.plot(sliding_options, matches)
            ax.plot(sliding_options[np.argmin(matches)], matches[np.argmin(matches)])
            ax.set_title('%s' % str([sliding_options[np.argmin(matches)], matches[np.argmin(matches)]]))
            fig.canvas.draw()
            img_np_raw_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_np_raw      = img_np_raw_flat.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img_np          = np.flip(img_np_raw, axis=2) # to bgr format, for ROS
            plt.close('all') # close matplotlib
            plt.show()
            img_msg = np2compressed(img_np)
            img_msg.header.stamp = rospy.Time.now()
            img_msg.header.frame_id = 'map'
            self.rollmatch_pub.publish(img_msg)

        return yaw_fix_rad

    def update_historical_data(self):
        '''
        Use historical data to generate new localisation estimate
        '''
        if not self.new_history: # ensure there is a new point to process
            return
        self.new_history = False

        _match_arr      = np.array(self.match_hist) # cast list of lists to array

        # Find index that is slice_length distance away from most recent position
        _ind            = -1
        for i in range(-2, -_match_arr.shape[0] - 1, -1):
            if np.sum(_match_arr[i:,enum_value(HDi.dist)]) > self.SLICE_LENGTH.get():
                _ind    = i
                break
        if _ind         == -1:
            return
        
        _distances      = np.array([np.sum(_match_arr[_ind+i+1:,enum_value(HDi.dist)]) for i in range(abs(_ind)-1)] + [0])
        
        _results        = ExpResults()
        _results.gt_pos = xyw(*self.match_hist[-1][enum_value(HDi.slam_x):enum_value(HDi.slam_w)+1])

        # Calculate VPR-only position estimate:
        _vpr_matches        = _match_arr[_ind:,:]
        _best_vpr           = np.argmin(_vpr_matches[:,enum_value(HDi.mDist)])
        _vpr_ind            = int(_vpr_matches[_best_vpr, enum_value(HDi.mInd)])
        _vpr_sum_so_far     = _distances[_best_vpr]
        _vpr_now_ind        = np.argmin(abs(np.array(self.path_sum) - (self.path_sum[_vpr_ind] + _vpr_sum_so_far)))
        _vpr_pos_now        = self.path_xyws[_vpr_now_ind, 0:3]
        _results.vpr_pos    = xyw(*_vpr_pos_now)

        # Calculate SVM position estimate:
        _svm_matches        = _vpr_matches[np.asarray(_vpr_matches[:,enum_value(HDi.svm_class)],dtype=bool),:]
        if not _svm_matches.shape[0]:
            _svm_pos_now        = np.nan
            _svm_ind            = np.nan
            _results.svm_state  = _results.FAIL
        else:
            _best_svm           = np.argmin(_svm_matches[:,enum_value(HDi.mDist)])
            _svm_ind            = int(_svm_matches[_best_svm, enum_value(HDi.mInd)])
            _svm_sum_so_far     = _distances[_best_svm]
            _svm_now_ind        = np.argmin(abs(np.array(self.path_sum) - (self.path_sum[_svm_ind] + _svm_sum_so_far))) # This will fail if a path is a loop
            _svm_pos_now        = self.path_xyws[_svm_now_ind, 0:3]
            _results.svm_pos    = xyw(*_svm_pos_now)
            _results.svm_state  = _results.SUCCESS

        # Pick which value to store:
        if self.TECHNIQUE.get() == Technique.VPR:
            self.current_hist_pos = list(_vpr_pos_now)
        elif self.TECHNIQUE.get() == Technique.SVM:
            self.current_hist_pos = list(_svm_pos_now)
        else:
            raise Exception('Bad technique: %s' % enum_name(self.TECHNIQUE.get()))
        
        self.current_results = _results

        self.experi_pos_pub.publish(_results) # Publish results to ROS

    def update_zone_target(self):
        '''
        Update zone return's target index
        '''
        if self.save_request        == Save_Request.SET:
            self.saved_index        = self.slam_current_ind
            self.saved_pose         = self.slam_ego
            publish_xyzrpy_pose([self.saved_pose[0], self.saved_pose[1], -0.5, 0, -np.pi/2, 0], self.teleport_pose_pub)
            publish_xyzrpy_pose([self.path_xyws[self.saved_index,0], self.path_xyws[self.saved_index,1], -0.5, 0, -np.pi/2, 0], self.teleport_index_pub)
            self.print('Saved current position.')
        elif self.save_request      == Save_Request.CLEAR:
            self.saved_index        = -1
            self.saved_pose         = None
            publish_xyzrpy_pose([0, 0, -0.5, 0, -np.pi/2, 0], self.teleport_pose_pub)
            publish_xyzrpy_pose([0, 0, -0.5, 0, -np.pi/2, 0], self.teleport_index_pub)
            self.print('Cleared saved position.')
        self.save_request           = Save_Request.NONE

    def update_COR(self, ego):
        '''
        Update centre-of-rotation for visualisation and precise alignment
        '''
        COR_x                   = ego[0] + self.COR_OFFSET.get() * np.cos(ego[2])
        COR_y                   = ego[1] + self.COR_OFFSET.get() * np.sin(ego[2])
        publish_xyzrpy_pose([COR_x, COR_y, -0.5, 0, -np.pi/2, 0], self.COR_pub)
        return [COR_x, COR_y]