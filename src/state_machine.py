"""!
The state machine that implements the logic.
"""
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
import time
import numpy as np
import rclpy

class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.teaching_waypoints = np.empty((0,8)) 
        self.gripper_state = 0 # 0 = open, 1 = closed
        self.moving_time = 0
        self.accel_time = 0
        self.waypoints = [
            [-np.pi/2,       -0.5,      -0.3,          0.0,        0.0],
            [0.75*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
            [0.5*-np.pi/2,   -0.5,      -0.3,      np.pi/2,        0.0],
            [0.25*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
            [0.0,             0.0,       0.0,          0.0,        0.0],
            [0.25*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
            [0.5*np.pi/2,     0.5,       0.3,     -np.pi/3,        0.0],
            [0.75*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
            [np.pi/2,         0.5,       0.3,     -np.pi/3,        0.0],
            [0.0,             0.0,       0.0,          0.0,        0.0]]

    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and funcitons as needed.
        """
        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()

        if self.next_state == "goto_waypoint":
            self.goto_waypoint()


    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """
        self.status_message = "State: Execute - Executing motion plan"
        self.current_state = "execute"

        moving_time = self.rxarm.set_moving_time(3)
        accel_time = self.rxarm.set_accel_time(5)
        for wp in self.waypoints:
            if self.next_state == "estop":
                self.status_message = "State: Execute - Estop signal detected, stopping motion"
                return
            self.rxarm.arm.set_joint_positions(wp,
                                                moving_time=moving_time,
                                                accel_time=accel_time,
                                                blocking=True
            )
        self.rxarm.sleep()
        self.next_state = "idle"

    def record(self):
        self.teaching_waypoints = np.empty((0,8))
        self.rxarm.arm.go_to_sleep_pose(moving_time=2,
                              accel_time=1,
                              blocking=True)
        self.gripper_state = 0
        self.rxarm.gripper.release()
        time.sleep(1)
        self.rxarm.disable_torque()
        print('Start recording waypoints')
    

    def record_waypoint(self):
        current_position = self.rxarm.get_positions()
        if len(self.teaching_waypoints) > 0:
            last_position = self.teaching_waypoints[-1][:5]
            angular_displacements = np.abs(current_position - last_position)
            max_angular_displacement = np.max(angular_displacements)
            self.move_time = np.clip(max_angular_displacement / (np.pi/6), 2, 6)
            self.accel_time = np.clip(self.move_time / 2, 1, 3)
        else:
            self.move_time = 2 
            self.accel_time = 0.5  
        current_position_with_state = np.append(current_position, [self.gripper_state, self.move_time, self.accel_time])
        print('******************************************************')
        print('current position with state',current_position_with_state)
        self.teaching_waypoints = np.vstack((self.teaching_waypoints, current_position_with_state))
        self.status_message = "State: Record - Recorded waypoint"
        print('======================================================')
        print('teaching waypoints', self.teaching_waypoints)

    def change_gripper_state(self):
        self.gripper_state = 1 - self.gripper_state
        print('---------------------------------------------------')
        print('the state of the gripper is now', self.gripper_state)
        
    def goto_waypoint(self):   
        self.current_state = "goto_waypoint"
        self.status_message = "State: Go to waypoint - Moving to waypoint"
        self.rxarm.enable_torque()
        self.rxarm.arm.go_to_sleep_pose(moving_time=2,
                              accel_time=1,
                              blocking=True)
        moving_time = self.rxarm.set_moving_time(3)
        accel_time = self.rxarm.set_accel_time(5)
        for wp in self.teaching_waypoints:
            joint_positions = wp[:5]  
            gripper_state = wp[5]     
            moving_time = wp[6]         
            accel_time = wp[7]        
            self.rxarm.arm.set_joint_positions(joint_positions,
                                                moving_time=moving_time,
                                                accel_time=accel_time,
                                                blocking=True)
            time.sleep(0.2)
            if gripper_state == 0:
                self.rxarm.gripper.release()
            else:
                self.rxarm.gripper.grasp()
            time.sleep(0.2)
        self.rxarm.arm.go_to_sleep_pose(moving_time=2,
                              accel_time=1,
                              blocking=True)
        print('===============================================================')
        print('Finished moving to all waypoints')
        self.next_state = "idle"
        


    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.next_state = "idle"

        """TODO Perform camera calibration routine here"""
        self.status_message = "Calibration - Completed Calibration"

    """ TODO """
    def detect(self):
        """!
        @brief      Detect the blocks
        """
        time.sleep(1)

    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            time.sleep(5)
        self.next_state = "idle"

class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            time.sleep(0.05)