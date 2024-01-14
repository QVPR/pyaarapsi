#! /usr/bin/env python3

import rospy
import copy
import numpy as np

from geometry_msgs.msg      import Twist, Pose, PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg        import Joy
from aarapsi_robot_pack.msg import Label, ExpResults, GoalExpResults, xyw

from gazebo_msgs.msg        import ModelState
from gazebo_msgs.srv        import SetModelState, SetModelStateRequest

from pyaarapsi.core.ros_tools               import LogType, pose2xyw, NodeState
from pyaarapsi.core.helper_tools            import p2p_dist_2d, formatException
from pyaarapsi.core.vars                    import C_I_RED, C_I_GREEN, C_I_YELLOW, C_I_BLUE, C_I_WHITE, C_RESET, C_CLEAR, C_UP_N

from pyaarapsi.vpr_simple.vpr_helpers       import FeatureType
from pyaarapsi.core.argparse_tools          import check_positive_float, check_enum, check_positive_int
from pyaarapsi.pathing.enums                import *
from pyaarapsi.pathing.basic                import *
from pyaarapsi.pathing.simple_follower_base import Simple_Follower_Class

class Extended_Follower_Class(Simple_Follower_Class):
    def init_params(self, rate_num: float, log_level: float, reset):
        super().init_params(rate_num, log_level, reset)

        # Experiment-specific Parameters:
        self.SLICE_LENGTH           = self.params.add(self.nodespace + "/exp/slice_length",         1.5,                    check_positive_float,                       force=reset)
        self.APPEND_DIST            = self.params.add(self.nodespace + "/exp/append_dist",          0.05,                   check_positive_float,                       force=reset)
        self.APPEND_MAX             = self.params.add(self.nodespace + "/exp/append_max",           50,                     check_positive_int,                         force=reset)
        self.EXPERIMENT_MODE        = self.params.add(self.nodespace + "/exp/mode",                 Experiment_Mode.UNSET,  lambda x: check_enum(x, Experiment_Mode),   force=reset)
        self.TECHNIQUE              = self.params.add(self.nodespace + "/exp/technique",            Technique.VPR,          lambda x: check_enum(x, Technique),         force=reset)

    def init_vars(self, simple=True):
        super().init_vars(simple) # Call base class method

        # Inter-loop variables for Zone-Return Features:
        self.zone_index         = -1                        # Index in path dataset corresponding to target zone
        self.return_stage       = Return_Stage.UNSET        # Enum to control flow of autonomous zone return commands
        self.saved_index        = -1                        # Index in path dataset user has selected to store
        self.saved_pose         = []                        # Saved position; in simulation, can teleport here.
        self.save_request       = Save_Request.NONE         # Enum to command setting/clearing of self.saved_index

        # Inter-loop variables for historical data management:
        self.match_hist         = []                        # Core historical data array
        self.new_history        = False                     # Whether new historical information has been appended
        self.current_hist_pos   = []                        # Most recent position estimate from historical data

        # Controller bind information:
        self.vpr_mode_ind       = PS4_Pressed.Square
        self.zone_mode_ind      = PS4_Pressed.Triangle

        self.store_pose_ind     = PS4_Pressed.Options
        self.clear_pose_ind     = PS4_Pressed.Share

        self.teleport_ind       = PS4_Pressed.RightStickIn

        self.experiment_ind     = PS4_Pressed.PS

        # Hashed entries for compact & fast access:
        self.experi_str_hash    = { Experiment_Mode.UNSET: C_I_GREEN + 'STOPPED',           Experiment_Mode.DRIVE_PATH: C_I_YELLOW + 'Path Following', 
                                    Experiment_Mode.INIT:  C_I_YELLOW + 'Initialising',     Experiment_Mode.HALT1:      C_I_YELLOW + 'Halting', 
                                    Experiment_Mode.ALIGN: C_I_YELLOW + 'Align to Start',   Experiment_Mode.DRIVE_GOAL: C_I_RED + 'Heading to Goal', 
                                    Experiment_Mode.HALT2: C_I_YELLOW + 'Halting',          Experiment_Mode.DANGER:     C_I_RED + 'Halting',
                                    Experiment_Mode.DONE:  C_I_GREEN + 'Complete' }

        # Experiment variables:
        self.exp_start_SLAM     = -1                        # Index in path dataset experiment will start SLAM
        self.exp_stop_SLAM      = -1                        # Index in path dataset experiment will stop SLAM
        self.exp_dist           = -1                        # Adjusted length of experiment (in case user starts too close to reference set edges)
        self.new_goal           = False                     # Whether a new 2D Nav Goal has been requested
        self.goal_pose          = [0]*3                     # Position of the 2D Nav Goal
        self.exp_transform      = np.array(None)            # Numpy array Homogenous Transform
        self.exp_rotation       = 0                         # 2D pose transformation correction
        self.point_shoot_stage  = Point_Shoot_Stage.UNSET
        self.point_shoot_start  = []                        # Initial position to measure relative to
        self.point_shoot_point  = 0                         # Angle to correct
        self.point_shoot_shoot  = 0                         # Distance to cover
        self.current_results    = ExpResults()
        self.exp_results        = GoalExpResults()
        self.exp_count          = 0

    def init_rospy(self):
        super().init_rospy()

        self.teleport_pose_pub  = self.add_pub(      self.nodespace + '/teleport/pose',        PoseStamped,                                queue_size=1, latch=True, subscriber_listener=self.sublis)
        self.teleport_index_pub = self.add_pub(      self.nodespace + '/teleport/index',       PoseStamped,                                queue_size=1, latch=True, subscriber_listener=self.sublis)
        self.experi_goal_pub    = self.add_pub(      self.nodespace + '/exp/goal',             PoseStamped,                                queue_size=1, latch=True, subscriber_listener=self.sublis)
        self.experi_start_pub   = self.add_pub(      self.nodespace + '/exp/start',            PoseStamped,                                queue_size=1, latch=True, subscriber_listener=self.sublis)
        self.experi_finish_pub  = self.add_pub(      self.nodespace + '/exp/finish',           PoseStamped,                                queue_size=1, latch=True, subscriber_listener=self.sublis)
        self.experi_pos_pub     = self.add_pub(      self.nodespace + '/exp/result',           ExpResults,                                 queue_size=1)
        self.experi_results_pub = self.add_pub(      self.nodespace + '/exp/goal_results',     GoalExpResults,                             queue_size=1)
        self.goal_sub           = rospy.Subscriber(  '/move_base_simple/goal',                 PoseStamped,               self.goal_cb,    queue_size=1)
        self.teleport_srv       = rospy.ServiceProxy('/gazebo/set_model_state',                SetModelState)
        
        self.sublis.add_operation(self.nodespace + '/exp/goal',         method_sub=self.path_peer_subscribe)
        self.sublis.add_operation(self.nodespace + '/teleport/pose',    method_sub=self.path_peer_subscribe)
        self.sublis.add_operation(self.nodespace + '/teleport/index',   method_sub=self.path_peer_subscribe)
        self.sublis.add_operation(self.nodespace + '/exp/start',        method_sub=self.path_peer_subscribe)
        self.sublis.add_operation(self.nodespace + '/exp/finish',       method_sub=self.path_peer_subscribe)

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
        if not self.ready:
            return
        
        super().state_cb(msg)
        
        # Update historical data:
        # Order defined in enumeration HDi
        _append = [msg.match_index, msg.truth_index, msg.distance_vector[msg.match_index], msg.gt_class, msg.svm_class, *self.slam_ego, *self.robot_ego, *self.vpr_ego]
        if not len(self.match_hist): # If this is the first entry,
            self.match_hist.append(_append + [0])
            self.new_history = True
            return
        
        _dist = p2p_dist_2d(self.match_hist[-1][(HDi.robot_x.value):(HDi.robot_y.value+1)], self.robot_ego[0:2])
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
                    rospy.set_param(self.namespace + '/feature_type', FeatureType.RAW.name)
                    self.print("Switched to %s." % self.FEAT_TYPE.get().name, LogType.INFO)
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

        try:
            super().path_peer_subscribe(topic_name)
            return
        except:
            pass

        if topic_name == self.nodespace + '/exp/goal':
            publish_xyzrpy_pose([self.goal_pose[0], self.goal_pose[1], -0.5, 0, -np.pi/2, 0], self.experi_goal_pub)

        elif topic_name == self.nodespace + '/teleport/pose':
            if self.saved_pose:
                publish_xyzrpy_pose([self.saved_pose[0], self.saved_pose[1], -0.5, 0, -np.pi/2, 0], self.teleport_pose_pub)
            else:
                publish_xyzrpy_pose([0, 0, -0.5, 0, -np.pi/2, 0], self.teleport_pose_pub)

        elif topic_name == self.nodespace + '/teleport/index':
            if not (self.saved_index == -1):
                publish_xyzrpy_pose([self.path_xyws[self.saved_index,0], self.path_xyws[self.saved_index,1], -0.5, 0, -np.pi/2, 0], self.teleport_index_pub)
            else:
                publish_xyzrpy_pose([0, 0, -0.5, 0, -np.pi/2, 0], self.teleport_index_pub)

        elif topic_name == self.nodespace + '/exp/start':
            if not (self.exp_start_SLAM == -1):
                publish_xyzrpy_pose([self.path_xyws[self.exp_start_SLAM,0], self.path_xyws[self.exp_start_SLAM,1], -0.5, 0, -np.pi/2, 0], self.experi_start_pub)
            else:
                publish_xyzrpy_pose([0, 0, -0.5, 0, -np.pi/2, 0], self.experi_start_pub)

        elif topic_name == self.nodespace + '/exp/finish':
            if not (self.exp_stop_SLAM == -1):
                publish_xyzrpy_pose([self.path_xyws[self.exp_stop_SLAM,0], self.path_xyws[self.exp_stop_SLAM,1], -0.5, 0, -np.pi/2, 0], self.experi_finish_pub)
            else:
                publish_xyzrpy_pose([0, 0, -0.5, 0, -np.pi/2, 0], self.experi_finish_pub)
        else:
            raise Exception('Unknown path_peer_subscribe topic: %s' % str(topic_name))

    def update_COR(self, ego):
        '''
        Update centre-of-rotation for visualisation and precise alignment
        '''
        COR_x                   = ego[0] + self.COR_OFFSET.get() * np.cos(ego[2])
        COR_y                   = ego[1] + self.COR_OFFSET.get() * np.sin(ego[2])
        publish_xyzrpy_pose([COR_x, COR_y, -0.5, 0, -np.pi/2, 0], self.COR_pub)
        return [COR_x, COR_y]

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
            if np.sum(_match_arr[i:,HDi.dist.value]) > self.SLICE_LENGTH.get():
                _ind    = i
                break
        if _ind         == -1:
            return
        
        _distances      = np.array([np.sum(_match_arr[_ind+i+1:,HDi.dist.value]) for i in range(abs(_ind)-1)] + [0])
        
        _results        = ExpResults()
        _results.gt_pos = xyw(*self.match_hist[-1][HDi.slam_x.value:HDi.slam_w.value+1])

        # Calculate VPR-only position estimate:
        _vpr_matches        = _match_arr[_ind:,:]
        _best_vpr           = np.argmin(_vpr_matches[:,HDi.mDist.value])
        _vpr_ind            = int(_vpr_matches[_best_vpr, HDi.mInd.value])
        _vpr_sum_so_far     = _distances[_best_vpr]
        _vpr_now_ind        = np.argmin(abs(np.array(self.path_sum) - (self.path_sum[_vpr_ind] + _vpr_sum_so_far)))
        _vpr_pos_now        = self.path_xyws[_vpr_now_ind, 0:3]
        _results.vpr_pos    = xyw(*_vpr_pos_now)

        # Calculate SVM position estimate:
        _svm_matches        = _vpr_matches[np.asarray(_vpr_matches[:,HDi.svm_class.value],dtype=bool),:]
        if not _svm_matches.shape[0]:
            _svm_pos_now        = [np.nan, np.nan, np.nan]
            _svm_ind            = np.nan
            _results.svm_state  = _results.FAIL
        else:
            _best_svm           = np.argmin(_svm_matches[:,HDi.mDist.value])
            _svm_ind            = int(_svm_matches[_best_svm, HDi.mInd.value])
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
            raise Exception('Bad technique: %s' % self.TECHNIQUE.get().name)
        
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
            self.saved_pose         = []
            publish_xyzrpy_pose([0, 0, -0.5, 0, -np.pi/2, 0], self.teleport_pose_pub)
            publish_xyzrpy_pose([0, 0, -0.5, 0, -np.pi/2, 0], self.teleport_index_pub)
            self.print('Cleared saved position.')
        self.save_request           = Save_Request.NONE

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
        svm_string      = base_svm_string % (self.REJECT_MODE.get().name, str(self.label.svm_class))
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

    def zone_return(self, ego, target, ignore_heading=False):
        '''

        Handle an autonomous movement to a target
        - Picks nearest target
        - Drives to target
        - Turns on-spot to face correctly

        Stages:
        Return_STAGE.UNSET: Stage 0 - New request: identify zone target.
        Return_STAGE.DIST:  Stage 1 - Distance from target exceeds 5cm: head towards target.
        Return_STAGE.TURN:  Stage 2 - Heading error exceeds 1 degree: turn on-the-spot towards target heading.
        Return_STAGE.DONE:  Stage 3 - FINISHED.

        '''

        publish_xyw_pose(target, self.goal_pub)

        if self.return_stage == Return_Stage.DONE:
            return True

        yaw_err     = calc_yaw_error(ego, target)
        ego_cor     = self.update_COR(ego) # must improve accuracy of centre-of-rotation as we do on-the-spot turns
        dist        = np.sqrt(np.square(ego_cor[0] - target[0]) + np.square(ego_cor[1] - target[1]))
        head_err    = angle_wrap(target[2] - ego[2], 'RAD')

        # If stage 1: calculate distance to target (lin_err) and heading error (ang_err)
        if self.return_stage == Return_Stage.DIST:
            if abs(yaw_err) < np.pi/18:
                ang_err = np.sign(yaw_err) * np.min([np.max([0.1, -0.19*abs(yaw_err)**2 + 0.4*abs(yaw_err) - 0.007]),1])
                lin_err = np.max([0.1, 2 * np.log10(dist + 0.7)])
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
            if (abs(head_err) < np.pi/180) or ignore_heading:
                self.set_command_mode(Command_Mode.STOP)
                self.return_stage = Return_Stage.DONE
                ang_err = 0

                return True
            else:
                ang_err = np.sign(head_err) * np.max([0.1, abs(head_err)])
        else:
            raise Exception('Bad return stage [%s].' % str(self.return_stage))

        self.try_send_command(lin_err,ang_err)
        return False
    
    def point_and_shoot(self, start, ego, point, shoot):
        '''

        Handle an autonomous movement to a target
        - Aims at target
        - Drives to target

        Stages:
        Point_Shoot_Stage.INIT:  Stage 0 - New request: start. 
        Point_Shoot_Stage.POINT: Stage 1 - Use point amount to correct within 2 degrees of error
        Point_Shoot_Stage.SHOOT: Stage 2 - Use shoot amount to correct witihn 10cm of error
        Point_Shoot_Stage.DONE:  Stage 3 - FINISHED.

        '''

        if self.point_shoot_stage == Point_Shoot_Stage.DONE:
            return True
        
        elif self.point_shoot_stage == Point_Shoot_Stage.INIT:
            self.point_shoot_stage = Point_Shoot_Stage.POINT

        elif self.point_shoot_stage == Point_Shoot_Stage.POINT:
            point_err = normalize_angle(point - (ego[2] - start[2]))
            if np.abs(point_err) > np.pi/90:
                self.try_send_command(0, np.sign(point_err) * 0.3)
            else:
                self.point_shoot_stage = Point_Shoot_Stage.SHOOT

        elif self.point_shoot_stage == Point_Shoot_Stage.SHOOT:
            shoot_err = shoot - p2p_dist_2d(start, ego)
            if np.abs(shoot_err) > 0.1:
                self.try_send_command(0.3,0)
            else:
                self.point_shoot_stage = Point_Shoot_Stage.DONE

        return False
    
    def experiment(self):

        if self.new_goal: # If 2D Nav Goal is used to request a goal
            _closest_goal_ind   = calc_current_ind(self.goal_pose, self.path_xyws)
            _stopping           = 0.5 # provide 50cm for vehicle to come to a halt
            self.exp_stop_SLAM  = np.argmin(np.abs(self.path_sum - (self.path_sum[_closest_goal_ind] - _stopping))) 
            # Some magic numbers, 0.10m and 0.05m, to ensure the historical data gets cleaned out between experiments
            self.exp_start_SLAM = np.argmin(np.abs(self.path_sum - (self.path_sum[self.exp_stop_SLAM] - (self.SLICE_LENGTH.get()+0.10)))) 
            self.exp_dist       = self.path_sum[self.exp_stop_SLAM] - self.path_sum[self.exp_start_SLAM]
            if self.exp_dist < (self.SLICE_LENGTH.get()+0.05): # Warn if the historical data may not clear
                self.print('Proposed experiment length: %0.2f [m]: Historical data may be retained!' % self.exp_dist, LogType.WARN)
            else:
                self.print('Proposed experiment length: %0.2f [m].' % self.exp_dist)
            self.new_goal       = False
            self.EXPERIMENT_MODE.set(Experiment_Mode.INIT)
            self.print('[Experiment] Initialisation phase.')

            publish_xyzrpy_pose([self.path_xyws[self.exp_start_SLAM,0], self.path_xyws[self.exp_start_SLAM,1], -0.5, 0, -np.pi/2, 0], self.experi_start_pub)
            publish_xyzrpy_pose([self.path_xyws[self.exp_stop_SLAM,0], self.path_xyws[self.exp_stop_SLAM,1], -0.5, 0, -np.pi/2, 0], self.experi_finish_pub)

        elif self.exp_dist is None:
            self.print('Experiment pending 2D Nav Goal ...', LogType.WARN, throttle=30)
            return
        
        elif self.EXPERIMENT_MODE.get() == Experiment_Mode.INIT:
            self.exp_results                    = GoalExpResults()
            self.exp_results.header.stamp       = rospy.Time.now()
            self.exp_results.header.frame_id    = 'map'
            self.exp_results.id                 = self.exp_count
            self.exp_count                      = self.exp_count + 1
            self.exp_results.path_start_pos     = xyw(*self.path_xyws[self.exp_start_SLAM,0:3])
            self.exp_results.path_finish_pos    = xyw(*self.path_xyws[self.exp_stop_SLAM,0:3])
            self.exp_results.mode               = self.TECHNIQUE.get().name
            self.exp_results.goal_position      = xyw(*self.goal_pose)
            self.return_stage                   = Return_Stage.DIST
            self.EXPERIMENT_MODE.set(Experiment_Mode.ALIGN)
            self.print('[Experiment] Align phase.')
        
        elif self.EXPERIMENT_MODE.get() == Experiment_Mode.ALIGN:
            if self.zone_return(self.slam_ego, self.path_xyws[self.exp_start_SLAM]):
                self.EXPERIMENT_MODE.set(Experiment_Mode.DRIVE_PATH)
                self.print('[Experiment] Driving along path.')
                return

        elif self.EXPERIMENT_MODE.get() == Experiment_Mode.DRIVE_PATH:
            if (self.path_sum[self.slam_current_ind] - self.path_sum[self.exp_start_SLAM]) < self.exp_dist:
                self.path_follow(self.slam_ego, self.slam_current_ind)
            else:
                self.EXPERIMENT_MODE.set(Experiment_Mode.HALT1)
                self.print('[Experiment] Halting ... (1)')
                return

        elif self.EXPERIMENT_MODE.get() == Experiment_Mode.HALT1:
            if len(self.robot_velocities):
                if np.sum(self.robot_velocities) < 0.05:
                    self.EXPERIMENT_MODE.set(Experiment_Mode.DRIVE_GOAL)
                    self.print('[Experiment] Driving to goal.')
                    self.point_shoot_stage                  = Point_Shoot_Stage.INIT
                    self.point_shoot_point                  = calc_yaw_error(self.current_hist_pos, self.goal_pose)
                    self.point_shoot_shoot                  = p2p_dist_2d(self.current_hist_pos, self.goal_pose)
                    self.point_shoot_start                  = self.robot_ego
                    self.exp_results.point                  = self.point_shoot_point
                    self.exp_results.shoot                  = self.point_shoot_shoot
                    self.exp_results.localisation           = copy.deepcopy(self.current_results)
                    self.exp_results.robot_goal_start_pos   = xyw(*self.robot_ego)
                    self.exp_results.slam_goal_start_pos    = xyw(*self.slam_ego)
                    return
            self.try_send_command(0,0)
        
        elif self.EXPERIMENT_MODE.get() == Experiment_Mode.DRIVE_GOAL:
            if self.point_and_shoot(start=self.point_shoot_start, ego=self.robot_ego, point=self.point_shoot_point, shoot=self.point_shoot_shoot):
                self.EXPERIMENT_MODE.set(Experiment_Mode.HALT2)
                self.exp_results.success = True
                self.print('[Experiment] Halting ... (2)')
                return
            
        elif self.EXPERIMENT_MODE.get() == Experiment_Mode.DANGER:
            self.EXPERIMENT_MODE.set(Experiment_Mode.HALT2)
            self.exp_results.success = False
            self.print('[Experiment] Halting due to potential danger')
            return

        elif self.EXPERIMENT_MODE.get() == Experiment_Mode.HALT2:
            if len(self.robot_velocities):
                if np.sum(self.robot_velocities) < 0.05:
                    self.exp_results.robot_goal_finish_pos   = xyw(*self.robot_ego)
                    self.exp_results.slam_goal_finish_pos    = xyw(*self.slam_ego)
                    self.experi_results_pub.publish(self.exp_results)
                    self.EXPERIMENT_MODE.set(Experiment_Mode.DONE)
                    return
            self.try_send_command(0,0)

        elif self.EXPERIMENT_MODE.get() == Experiment_Mode.DONE:
            self.print('[Experiment] Complete.', throttle=30)

    def loop_contents(self):
        '''
        
        Main Loop

        '''

        # Calculate current SLAM position and zone:
        self.slam_current_ind       = calc_current_ind(self.slam_ego, self.path_xyws)
        self.slam_zone              = calc_current_zone(self.slam_current_ind, self.num_zones, self.zone_indices)

        self.update_zone_target()

        # Calculate/estimate current ego:
        if self.command_mode == Command_Mode.VPR: # If we are estimating pose, calculate via VPR:
            self.est_current_ind    = self.label.match_index
            self.heading_fixed      = normalize_angle(angle_wrap(self.vpr_ego[2] + self.roll_match(self.est_current_ind), 'RAD'))
            self.ego                = [self.vpr_ego[0], self.vpr_ego[1], self.heading_fixed]

        else: # If we are not estimating pose, use everything from the ground truth:
            self.est_current_ind    = self.slam_current_ind
            self.heading_fixed      = normalize_angle(angle_wrap(self.slam_ego[2] + self.roll_match(self.est_current_ind), 'RAD'))
            self.ego                = self.slam_ego
        
        publish_xyw_pose(self.path_xyws[self.slam_current_ind], self.slam_pub) # Visualise SLAM nearest position on path
        self.update_historical_data() # Manage storage of historical data

        # Denest; check if stopped:
        if self.command_mode in [Command_Mode.STOP]:
            pass

        # Else: if the current command mode is a path-following exercise:
        elif self.command_mode in [Command_Mode.VPR, Command_Mode.SLAM]:
            self.path_follow(self.ego, self.est_current_ind)

        # Else: if the current command mode is set to return-to-nearest-zone-boundary:
        elif self.command_mode in [Command_Mode.ZONE_RETURN]:
        
            # If stage 0: determine target and move to stage 1
            if self.return_stage == Return_Stage.UNSET:
                if self.saved_index == -1: # No saved zone
                    self.zone_index  = calc_nearest_zone(self.zone_indices, self.est_current_ind, self.path_xyws.shape[0])
                else:
                    self.zone_index  = self.saved_index
                self.return_stage = Return_Stage.DIST
            
            self.zone_return(self.ego, self.path_xyws[self.zone_index,:])

        # Else: if the current command mode is set to special functions (experiments, testing):
        if not (self.EXPERIMENT_MODE.get() == Experiment_Mode.UNSET):
            self.experiment()

        # Print HMI:
        if self.PRINT_DISPLAY.get():
            self.print_display()
        self.publish_controller_info()