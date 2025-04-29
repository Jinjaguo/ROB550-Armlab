"""!
Implements the RXArm class.

The RXArm class contains:

* last feedback from joints
* functions to command the joints
* functions to get feedback from joints
* functions to do FK and IK
* A function to read the RXArm config file

You will upgrade some functions and also implement others according to the comments given in the code.
"""
import numpy as np
from functools import partial
from kinematics import FK_dh, FK_pox, get_pose_from_T, get_euler_angles_from_T
import time
import csv
import sys, os
from pdb import set_trace

from builtins import super
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from resource.config_parse import parse_dh_param_file
from sensor_msgs.msg import JointState
import rclpy
from rclpy.executors import SingleThreadedExecutor

sys.path.append('../../interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/interbotix_xs_modules/xs_robot') 
from arm import InterbotixManipulatorXS
from mr_descriptions import ModernRoboticsDescription as mrd

from kinematics import IK

"""
TODO: Implement the missing functions and add anything you need to support them
"""
""" Radians to/from  Degrees conversions """
D2R = np.pi / 180.0
R2D = 180.0 / np.pi


def _ensure_initialized(func):
    """!
    @brief      Decorator to skip the function if the RXArm is not initialized.

    @param      func  The function to wrap

    @return     The wraped function
    """
    def func_out(self, *args, **kwargs):
        if self.initialized:
            return func(self, *args, **kwargs)
        else:
            print('WARNING: Trying to use the RXArm before initialized')

    return func_out


class RXArm(InterbotixManipulatorXS):
    """!
    @brief      This class describes a RXArm wrapper class for the rx200
    """
    def __init__(self, dh_config_file=None):
        """!
        @brief      Constructs a new instance.

                    Starts the RXArm run thread but does not initialize the Joints. Call RXArm.initialize to initialize the
                    Joints.

        @param      dh_config_file  The configuration file that defines the DH parameters for the robot
        """
        super().__init__(robot_model="rx200")
        self.joint_names = self.arm.group_info.joint_names
        self.num_joints = 5
        # Gripper
        self.gripper_state = True
        # State
        self.initialized = False
        # Cmd
        self.position_cmd = None
        self.moving_time = 2.0
        self.accel_time = 0.5
        # Feedback
        self.position_fb = None
        self.velocity_fb = None
        self.effort_fb = None
        # DH Params
        # self.dh_params = np.array([[0.0, 0.0, 0.0, np.pi/2],
        #                 [0.0, 0.0, 0.0, 0.0],
        #                 [0.0, np.pi/2, 0.0, 0.0],
        #                 [205.73, 0.0, 0.0, 0.0],
        #                [200.0, 0.0, 0.0, 0.0],
        #                [174.15, np.pi/2, 0.0, 0.0]])
        self.servo_offsets_fk = D2R * np.array([90, -74.1412, +70, -3, 0])
        self.dh_params = np.array([
                # [0.0, np.pi, 0.0, 0.0], # world
                [0.0, -np.pi/2, 0.0, 0], # base
                [205.73, 0.0, 0.0, 0.0], # shoulder
                [200.0, 0.0, 0.0, 0.0], #elbow
                [0.0, -np.pi/2, 0.0, 0.0], #wrist pitch
                [0.0, np.pi/2, 0.0, np.pi/2], #wrist yaw
                [0.0, 0.0, 174.15, 0.0] #wrist roll
                ])

        self.dh_params_joint_indices = [
                                        # 0, # world      
                                         1,
                                           1,
                                             1,
                                               1,
                                                 0,
                                                   1
                                                   ]
        

        self.dh_config_file = dh_config_file
        if (dh_config_file is not None):
            self.dh_params = RXArm.parse_dh_param_file(dh_config_file)
        #POX params
        self.M_matrix = []
        self.S_list = []
        self.learned_waypoints = []

    def initialize(self):
        """!
        @brief      Initializes the RXArm from given configuration file.

                    Initializes the Joints and serial port

        @return     True is succes False otherwise
        """
        self.initialized = False
        # Wait for other threads to finish with the RXArm instead of locking every single call
        time.sleep(0.25)
        """ Commanded Values """
        self.position = [0.0] * self.num_joints  # radians
        """ Feedback Values """
        self.position_fb = [0.0] * self.num_joints  # radians
        self.velocity_fb = [0.0] * self.num_joints  # 0 to 1 ???
        self.effort_fb = [0.0] * self.num_joints  # -1 to 1

        # Reset estop and initialized
        self.estop = False
        self.enable_torque()
        self.moving_time = 2.0
        self.accel_time = 0.5
        self.arm.go_to_home_pose(moving_time=self.moving_time,
                             accel_time=self.accel_time,
                             blocking=False)
        self.gripper.release()
        self.initialized = True
        return self.initialized

    def go_to_ready_pose(self):
        self.moving_time = 2.0
        self.accel_time = 1.0
        pose = D2R * np.array([0.0, -60.0, 60.0, 0.0, 0.0])
        print("Go to ready pose", pose)
        move_time = self.compute_move_time(pose)
        self.set_moving_time(move_time)
        print("go to rdy move time", move_time)
        #self.set_positions(pose, is_blocking=True)
        self.arm.set_joint_positions(pose,
                            2.0,
                            1.0,
                            True)

    def sleep(self):
        self.moving_time = 2.0
        self.accel_time = 1.0
        self.arm.go_to_home_pose(moving_time=self.moving_time,
                             accel_time=self.accel_time,
                             blocking=True)
        self.arm.go_to_sleep_pose(moving_time=self.moving_time,
                              accel_time=self.accel_time,
                              blocking=False)
        self.initialized = False

    def grab_block(self, target_pos_world, size, phi):
        print("\nGrabbing block")

        target_pos_world[2] += size / 3

        roll_cmd = phi
        block_offset = size / 2
        joint_lengths = np.array([self.dh_params[1][0], self.dh_params[2][0], self.dh_params[-1][2]])

        target_reachable = False
        premove_reachable = False
        pitch_cmd = -np.pi/2
        pitch_cmd -= np.pi/8 # initialize one iteration before the loop
        
        while not(target_reachable and premove_reachable) and pitch_cmd < np.pi/4:
            pitch_cmd += np.pi/8 #iterate

            target_pos_cmd = target_pos_world - np.array([0, 0, block_offset]) #grabbing

            target_reachable, target_pose = IK(target_pos_cmd, roll_cmd, pitch_cmd, joint_lengths)

            premove_pos_cmd = target_pos_cmd + np.array([0, 0, block_offset*3])
            premove_reachable, premove_pose = IK(premove_pos_cmd, roll_cmd, pitch_cmd, joint_lengths)
            # print("While loop iteration for pi= ", pitch_cmd)
        
        if not(target_reachable and premove_reachable):
            return False, None
        premove_pose -= self.servo_offsets_fk
        target_pose -= self.servo_offsets_fk

        # print("target_pos_world", target_pos_world)
        # print("premove_pos_cmd", premove_pos_cmd)
        # print("target_pos_cmd", target_pos_cmd)

        time.sleep(0.1)

        print(f"pitch: {pitch_cmd}")
        # print("executing first move", premove_pose)
        move_time = self.compute_move_time(premove_pose)
        self.set_moving_time(move_time)
        print(self.set_positions(premove_pose, is_blocking=True))

        time.sleep(0.1)


        # print("executing second move", target_pose)
        move_time = self.compute_move_time(target_pose)
        self.set_moving_time(move_time)
        print(self.set_positions(target_pose, is_blocking=True))
        self.gripper.grasp()
        time.sleep(0.5)
        # set_trace()

        # print("executing last move", premove_pose)
        move_time = self.compute_move_time(premove_pose)
        self.set_moving_time(move_time)
        print(self.set_positions(premove_pose, is_blocking=True))

        time.sleep(0.1)

        return True, pitch_cmd

    def drop_block(self, target_pos_world, size, roll, pitch):
        print("\nDropping block")
        roll_cmd = 0
        block_offset = size / 2
        joint_lengths = np.array([self.dh_params[1][0], self.dh_params[2][0], self.dh_params[-1][2]])

        target_reachable = False
        premove_reachable = False
        postmove_reachable = False

        all_moves_reachable = target_reachable and premove_reachable and postmove_reachable
        pitch_cmd = pitch

        target_pos_cmd = target_pos_world + np.array([0, 0, block_offset + 10]) #placing
        #TODO: offset premove towards 0,0 by 50mm

        premove_pos_cmd = np.array([target_pos_cmd[0], target_pos_cmd[1]])
        #print("step 1, ", premove_pos_cmd)
        dist = np.hypot(premove_pos_cmd[0], premove_pos_cmd[1])
        assert (dist>100)
        premove_pos_cmd = (premove_pos_cmd / dist) * (dist - 75)
        #print("step 2, ", premove_pos_cmd)
        premove_pos_cmd = np.array([premove_pos_cmd[0], premove_pos_cmd[1], target_pos_cmd[2]])
        #print("step 3, ", premove_pos_cmd)

        premove_reachable, premove_pose = IK(premove_pos_cmd + np.array([0, 0, block_offset*4]), roll_cmd, pitch, joint_lengths)
        # print('premove pose preoffset', R2D * premove_pose)
        target_reachable, target_pose = IK(target_pos_cmd, roll_cmd, pitch_cmd, joint_lengths)
        postmove_reachable, postmove_pose = IK(target_pos_cmd + np.array([0, 0, block_offset*3]), roll_cmd, pitch_cmd, joint_lengths)        
        all_moves_reachable = target_reachable and premove_reachable and postmove_reachable
        # print('IK solve status', all_moves_reachable)


        

        if not(all_moves_reachable) and pitch != 0 and pitch != -np.pi/2:       
            flipmove_reachable, flip_pose_drop = IK(np.array([150, 0, 150]), roll_cmd, -np.pi/2, joint_lengths)
            if(flipmove_reachable):
                flip_pose_drop -= self.servo_offsets_fk
                move_time = self.compute_move_time(flip_pose_drop)
                self.set_moving_time(move_time)
                self.set_positions(flip_pose_drop, is_blocking=True)
                self.gripper.release()
                time.sleep(1.0)

                return -1 #flipmove_reachable and block dropped

                #_, flip_pose_grab = IK(np.array[0, 150, block_offset], roll_cmd, -np.pi/2, joint_lengths)
                #move_time = self.compute_move_time(flip_pose_grab)
                #self.set_moving_time(move_time)
                #self.set_positions(flip_pose_grab, is_blocking=True)
                #self.gripper.grab()
                #self.drop_block(target_pos_world, size, roll, 0)
            else:
                print("Move to reorient cube not reachable")
                return 0
        elif not(all_moves_reachable) and pitch == 0:
            print('Drop location not reachable [IK failed]')
            return 0
        elif not(all_moves_reachable) and pitch == -np.pi/2:
            #if -pi/2 not possible make it zero and try to place again
            return self.drop_block(target_pos_world, size, roll, 0) #recursive call with pitch 0
        else:
            #actually do the movements
            premove_pose -= self.servo_offsets_fk
            target_pose -= self.servo_offsets_fk
            postmove_pose -= self.servo_offsets_fk

            # print("base rotation for move 0", R2D * premove_pose[0])
            move_time = self.compute_move_time(premove_pose)
            self.set_moving_time(move_time)
            self.set_positions(premove_pose, is_blocking=True)

            # print("base rotation for move 1", R2D * postmove_pose[0])
            move_time = self.compute_move_time(postmove_pose)
            self.set_moving_time(move_time)
            self.set_positions(postmove_pose, is_blocking=True)

            # print("base rotation for move 2", R2D * target_pose[0])
            move_time = self.compute_move_time(target_pose) #fails when pitch=0 for some reason
            self.set_moving_time(move_time)
            self.set_positions(target_pose, is_blocking=True)
            self.gripper.release()
            time.sleep(1.0)

            # print("base rotation for move 3", R2D * postmove_pose[0])
            move_time = self.compute_move_time(postmove_pose)
            self.set_moving_time(move_time)
            self.set_positions(postmove_pose, is_blocking=True)

            # print("base rotation for move 4", R2D * premove_pose[0])
            move_time = self.compute_move_time(premove_pose)
            self.set_moving_time(move_time)
            self.set_positions(premove_pose, is_blocking=True)

            return 1
        
    def grab_fixed_point(self, target_pos_world, phi, theta):
        roll_cmd = phi
        joint_lengths = np.array([self.dh_params[1][0], self.dh_params[2][0], self.dh_params[-1][2]])
        pitch_cmd = theta
        fixed_bolck_reachable, fixed_block_pose = IK(target_pos_world, roll_cmd, pitch_cmd, joint_lengths)
        if fixed_bolck_reachable:
            fixed_block_pose -= self.servo_offsets_fk
            move_time = 1.5
            self.set_moving_time(move_time)
            self.set_positions(fixed_block_pose, is_blocking=True)
            time.sleep(0.2)
            return True, pitch_cmd
        


    def set_positions(self, joint_positions, is_blocking=False):
        """!
         @brief      Sets the positions.

         @param      joint_angles  The joint angles
         """
        return self.arm.set_joint_positions(joint_positions,
                                 moving_time=self.moving_time,
                                 accel_time=self.accel_time,
                                 blocking=is_blocking)

    def set_moving_time(self, moving_time):
        self.moving_time = moving_time

    def set_accel_time(self, accel_time):
        self.accel_time = accel_time

    def disable_torque(self):
        """!
        @brief      Disables the torque and estops.
        """
        self.core.robot_set_motor_registers('group', 'all', 'Torque_Enable', 0)

    def enable_torque(self):
        """!
        @brief      Disables the torque and estops.
        """
        self.core.robot_set_motor_registers('group', 'all', 'Torque_Enable', 1)

    def get_positions(self):
        """!
        @brief      Gets the positions.

        @return     The positions.
        """
        return self.position_fb

    def get_velocities(self):
        """!
        @brief      Gets the velocities.

        @return     The velocities.
        """
        return self.velocity_fb

    def get_efforts(self):
        """!
        @brief      Gets the loads.

        @return     The loads.
        """
        return self.effort_fb
    
    def save_position(self, gripper_state):
        """!
        @brief      Save the current position
        """
        self.learned_waypoints.append([self.get_positions() - self.servo_offsets_fk, (gripper_state)])
        print(self.learned_waypoints[-1])
        

    def clear_saved_positions(self):
        """!
        @brief      Clear the saved positions
        """
        self.learned_waypoints = []
        self.arm.go_to_sleep_pose(moving_time=self.moving_time,
                              accel_time=self.accel_time,
                              blocking=True)
        self.disable_torque()
        print("Starting new learned path")


#   @_ensure_initialized

    def get_ee_pose(self):
        """!
        @brief      TODO Get the EE pose. Distances should be in mm

        @return     The EE pose as [x, y, z, phi, theta, psi]
        """

        count = 0
        for i, row in enumerate(self.dh_params):
            if self.dh_params_joint_indices[i] == 1:
                self.dh_params[i, 3] = self.position_fb[count]
                count += 1

        #print(self.dh_params)

        Transform_out = FK_dh(self.dh_params)

        translation = Transform_out @ [0, 0, 0, 1] + [0, 0, 103.91, 1]

        angles = get_euler_angles_from_T(np.linalg.inv(Transform_out))
        return [translation[0], translation[1], translation[2], angles[0], angles[1], angles[2]]


    def parse_pox_param_file(self):
        """!
        @brief      TODO Parse a PoX config file

        @return     0 if file was parsed, -1 otherwise 
        """
        return -1

    def parse_dh_param_file(self):
        print("Parsing DH config file...")
        dh_params = parse_dh_param_file(self.dh_config_file)
        print("DH config file parse exit.")
        return dh_params

    def get_dh_parameters(self):
        """!
        @brief      Gets the dh parameters.

        @return     The dh parameters.
        """
        return self.dh_params
    
    def compute_move_time(self, goal, max_ang_vel=np.pi/2):
        """!
        @brief      Compute the time to move to a goal position

        @return     The time to move to the goal position
        """
        delta = self.position_fb - self.servo_offsets_fk - goal
        max_delta = np.max(np.abs(delta))
        time = max_delta / max_ang_vel
        if time < 1:
            time = 1
        elif time > 10:
            time = 10

        return time

class RXArmThread(QThread):
    """!
    @brief      This class describes a RXArm thread.
    """
    updateJointReadout = pyqtSignal(list)
    updateEndEffectorReadout = pyqtSignal(list)

    def __init__(self, rxarm, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      RXArm  The RXArm
        @param      parent  The parent
        @details    TODO: set any additional initial parameters (like PID gains) here
        """
        QThread.__init__(self, parent=parent)
        self.rxarm = rxarm
        self.node = rclpy.create_node('rxarm_thread')
        self.subscription = self.node.create_subscription(
            JointState,
            '/rx200/joint_states',
            self.callback,
            10
        )
        self.subscription  # prevent unused variable warning
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def callback(self, data):
        self.rxarm.position_fb = np.asarray(data.position)[0:5] + self.rxarm.servo_offsets_fk
        self.rxarm.velocity_fb = np.asarray(data.velocity)[0:5]
        self.rxarm.effort_fb = np.asarray(data.effort)[0:5]
        self.updateJointReadout.emit(self.rxarm.position_fb.tolist())
        self.updateEndEffectorReadout.emit(self.rxarm.get_ee_pose())
        #for name in self.rxarm.joint_names:
        #    print("{0} gains: {1}".format(name, self.rxarm.get_motor_pid_params(name)))
        if (__name__ == '__main__'):
            print(self.rxarm.position_fb)

    def run(self):
        """ Spin the executor """
        try:
            while rclpy.ok():
                self.executor.spin_once(timeout_sec=0.02)
        finally:
            self.node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    rclpy.init() # for test
    rxarm = RXArm()
    print(rxarm.joint_names)
    armThread = RXArmThread(rxarm)
    armThread.start()
    try:
        joint_positions = [-1.0, 0.5, 0.5, 0, 1.57]
        rxarm.initialize()

        rxarm.arm.go_to_home_pose()
        rxarm.set_gripper_pressure(0.5)
        rxarm.set_joint_positions(joint_positions,
                                  moving_time=2.0,
                                  accel_time=0.5,
                                  blocking=True)
        rxarm.gripper.grasp()
        rxarm.arm.go_to_home_pose()
        rxarm.gripper.release()
        rxarm.sleep()

    except KeyboardInterrupt:
        print("Shutting down")

    rclpy.shutdown()