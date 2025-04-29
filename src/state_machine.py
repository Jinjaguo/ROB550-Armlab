"""!
The state machine that implements the logic.
"""
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
import time
import numpy as np
import rclpy
from kinematics import IK
from pdb import set_trace

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
        self.status_message = "State: Init"
        self.current_state = "init"
        self.next_state = "init"
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
        
        self.learned_waypoints = [
            # [([ 0.53075737, -0.27611655,  0.52615541,  1.42506814,  0.55683506]), 0],
            # [([ 0.53535932, -0.18100974,  0.63506806,  1.25479627,  0.55530107]), 0],
            # [([ 0.53689331, -0.18100974,  0.63813603,  1.25479627,  0.55530107]), 1],
            # [([ 0.9510681 , -0.63200009,  1.11980605,  1.19957304,  0.16873789]), 1],
            # [([-0.65654379, -0.73937875,  1.19497108,  1.15508759, -0.18714567]), 1],
            # [([ 0.09357283, -0.41570881,  0.77926224,  1.24712646,  0.09817477]), 1],
            # [([ 0.54916513, -0.19788353,  0.71176708,  1.14588368,  0.54763114]), 1],
            # [([ 0.54916513, -0.19788353,  0.71176708,  1.14588368,  0.54763114]), 0],
            # [([ 0.54916513, -0.2561748 ,  0.20248547,  1.71959245,  0.52155346]), 0],
            [([ 1.73953426,  0.83879628, -1.01242733,  1.71345663, -0.56450492]), 1],
            [([ 1.57386434,  0.21015537, -0.86363119,  0.93419433, -0.15339808]), 1],
            [([ 1.26093221,  0.42184472, -0.74551469,  1.15968955, -0.08897088]), 0],
            [([ 1.70425272,  0.16566993, -0.79920399,  0.98481572, -0.09357283]), 0],
            [([ 1.4496119 ,  0.14266022,  0.1582719 ,  0.81914574, -1.00629139]), 0],

        ]
        
        self.block_grabbed = False
        self.click_point = None
        self.pitch_cmd = -np.pi/2

        #event 1
        self.event_1_z__large = 0
        self.event_1_z__small = 0

        # event2
        self.event_2_small_end = np.zeros(3)
        self.event_2_large_end = np.zeros(3)

        # event 3
        self.event_3_stacking_height = 5
        self.event_3_stacking_pos = np.array([-150, 250, self.event_3_stacking_height])

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

        if self.next_state == "init":
            self.init()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute_static":
            self.execute_static()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()

        if self.next_state == "teach":
            self.teach()

        if self.next_state == "execute_learned":
            self.execute_learned()

        if self.calibrate == "calibrating":
            self.calibrate()

        if self.next_state == "click_block":
            self.click_block()      
              
        if self.next_state == "detect":
            self.detect()

        if self.next_state == "event_1":
            self.event_1()

        if self.next_state == "event_2":
            self.event_2()

        if self.next_state == "event_3":
            self.event_3()

    """Functions run for each state"""

    def init(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Init - Waiting to initalize arm"
        self.current_state = "init"
        self.rxarm.set_moving_time(2)
        self.rxarm.set_accel_time(5)

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
        self.event_3_stacking_pos[2] = self.event_3_stacking_height
        self.event_1_z__small = self.event_1_z__large = 0

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def execute_static(self):
        """!
        @brief      Go through all waypoints
        """
        self.status_message = "State: execute_static - Executing motion plan TEST"

        if self.current_state != "execute_static": #do nothing on first run to update status message
            self.current_state = "execute_static"
            return

        self.rxarm.set_moving_time(2)
        self.rxarm.set_accel_time(1)

        self.rxarm.arm.go_to_home_pose(self.rxarm.moving_time, self.rxarm.accel_time, True)

        for wp in self.waypoints:
            self.rxarm.set_positions(wp,is_blocking=True)
            if self.current_state != self.next_state:
                return

        self.rxarm.sleep()
        self.next_state = "idle"
    
    def execute_learned(self):
        """!
        @brief      Execute the learned waypoints
        """
        self.status_message = "State: execute_learned - Executing learned waypoints"

        if self.current_state != "execute_learned":
            self.current_state = "execute_learned"
            # self.rxarm.arm.go_to_sleep_pose(self.rxarm.moving_time, self.rxarm.accel_time, False)
            self.rxarm.enable_torque()
            return
        
        self.rxarm.set_moving_time(1)
        self.rxarm.set_accel_time(5)


        # self.rxarm.arm.go_to_home_pose(self.rxarm.moving_time, self.rxarm.accel_time, True)
        self.rxarm.gripper.release()
        self.block_grabbed = False

        waypoints = self.learned_waypoints if len(self.rxarm.learned_waypoints) == 0 else self.rxarm.learned_waypoints

        for wp in waypoints:
            self.rxarm.set_moving_time(self.rxarm.compute_move_time(wp[0], 3*np.pi/2))

            self.rxarm.set_positions(wp[0],is_blocking=True)
            if wp[1] == 1 and not self.block_grabbed:
                self.rxarm.gripper.grasp()
                self.block_grabbed = True
                time.sleep(0.5)
            elif wp[1] == 0 and self.block_grabbed:
                self.rxarm.gripper.release()
                self.block_grabbed = False
                time.sleep(0.5)
            if self.current_state != self.next_state:
                return

        # self.rxarm.sleep()
        # self.next_state = "idle"
        

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
        self.current_state = "detect"
        self.status_message = "State: Detect - Detecting blocks"
        self.camera.blockDetector()
        self.next_state = "idle"
        
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
        self.block_grabbed = False

    def calibrate(self):
        """!
        @brief      Calibrate the camera
        """
        self.current_state = "calibrate"
        self.status_message = "State: Calibrate - Calibrating Camera"
        
        self.camera.calibrate()

        self.next_state = "idle"

    def click_block(self):
        if self.click_point is None:
            print("Pixel to World failed - click block")
            self.next_state = "idle"
            return
        
        if self.current_state != "click_block":
            self.current_state = "click_block"

            if not self.block_grabbed:
                self.status_message = "State: Grab - Moving to grab block"
                self.rxarm.gripper.release()
                self.pitch_cmd = -np.pi/2
            else:
                self.status_message = "Status: Release - Moving to release block"
                self.rxarm.gripper.grasp()
            return
        
        if self.block_grabbed:
            success = self.rxarm.drop_block(self.click_point, 35, 0, self.pitch_cmd)
            if not success:
                self.next_state = "idle"
                print("drop failed")
                return
        else:
            grab_angle = 0
            block_size = 38
            for (x,y,z,angle,size) in self.camera.detected_blocks.values():
                if abs(x - self.click_point[0]) < size and abs(y - self.click_point[1]) < size:
                    self.click_point = np.array([x,y,z])
                    grab_angle = angle
                    block_size = size

            success, self.pitch_cmd = self.rxarm.grab_block(self.click_point, block_size, grab_angle)
            if not success:
                self.next_state = "idle"
                print("grab failed")
                return
        
        print('executing move to ready pose')
        move_time = self.rxarm.compute_move_time(np.array([0, -0.6, 0.6, 0, 0]))
        self.rxarm.set_moving_time(move_time)
        print("move time = ", self.rxarm.moving_time)
        time.sleep(0.5)
        self.rxarm.set_positions(np.array([0, -0.6, 0.6, 0, 0]), is_blocking=True)


        self.block_grabbed = not self.block_grabbed
        self.next_state = "idle"

    def event_1(self):
        self.current_state = "event_1"
        self.status_message = "Event 1"

        block_found = False
        self.go_to_ready_pose()

        self.camera.save_detections = True            
        time.sleep(0.2)

        for key in sorted(self.camera.detected_blocks.keys(), key=lambda x: x[1]):
            value = self.camera.detected_blocks.get(key)

            if value is None:
                continue
            (x,y,z,angle,size) = value

            if y > 0:
                block_found = True
                break
        if not block_found:
            print("Empty dictionary\n")
            print("save_detections = ",self.camera.save_detections)
            return
        
        print(f"Key {key}, x {x}, y {y}, z {z}, angle {angle}, size {size}")

        self.camera.save_detections = False
        grab_success, pitch = self.rxarm.grab_block(np.array([x,y,z]), size, angle)

        if not grab_success:
            print("event 1 - Grab failed")
            return

        self.go_to_ready_pose()
        
        stacking_pos = np.array([250 if size > 30 else -250, -100, 0])
        tmp_stacking_pos = stacking_pos

        max_z = 0
        for (x,y,z,_,_) in self.camera.detected_blocks.values():
            print(f"x {x}, y {y}, z {z}")
            if z > max_z and abs(x - stacking_pos[0]) < size and abs(y - stacking_pos[1]) < size:
                tmp_stacking_pos = np.array([x,y,z])
                max_z = z
                print(f"max z: {max_z}")

        # stacking_pos = (tmp_stacking_pos + stacking_pos)/2

        stacking_pos[2] = self.event_1_z__large if size > 30 else self.event_1_z__small

        self.camera.save_detections = False
        drop_success = self.rxarm.drop_block(stacking_pos, size, angle, pitch)
        if size > 30:
            self.event_1_z__large += size
        else:
            self.event_1_z__small += size

        if drop_success == 0:
            self.rxarm.arm.release()
            self.next_state = "idle"
            print("Release failed - Going back to idle")  

    def event_2(self):
        self.current_state = 'event_2'
        self.status_message = "Event 2"

        block_found = False

        large_starting_point = np.array([200, -75, 0])
        small_starting_point = np.array([-200, -75, 0])
        self.go_to_ready_pose()

        self.camera.save_detections = True            
        time.sleep(0.2)

        for key in sorted(self.camera.detected_blocks.keys(), key=lambda x: x[1]):
            value = self.camera.detected_blocks.get(key)

            if value is None:
                continue
            (x,y,z,angle,size) = value
            if size > 30: # large
                y_range = [large_starting_point[1], self.event_2_large_end[1]]
                stacking_pos = large_starting_point
            else:
                y_range = [small_starting_point[1], self.event_2_small_end[1]]
                stacking_pos = small_starting_point

            if not(y_range[0] - size < y < y_range[1] + size and abs(x - stacking_pos[0]) < 50):
                block_found = True
                print("Y-range", y_range)
                print("x-range", stacking_pos[0])
                break

        if not block_found:
            print("Empty dictionary")
            return
        
        print(f"Key {key}, x {x}, y {y}, z {z}, angle {angle}, size {size}")

        self.camera.save_detections = False
        grab_success, pitch = self.rxarm.grab_block(np.array([x,y,z]), size, angle)

        if not grab_success:
            print("event 2 - Grab failed")
            return
        
        self.go_to_ready_pose()
        
        block_offset = 20

        stacking_pos[1] += (size)*key[1]

        max_z = 0
        for (x,y,z,_,_) in self.camera.detected_blocks.values():
            if z > max_z and abs(x - stacking_pos[0]) < size and abs(y - stacking_pos[1]) < size:
                stacking_pos = np.array([x,y,z])
                max_z = z
                print(f"max z: {max_z}")

        self.camera.save_detections = False
        drop_success = self.rxarm.drop_block(stacking_pos, size, angle, pitch)

        if size > 30:
            self.event_2_large_end = stacking_pos if stacking_pos[1] > self.event_2_large_end[1] else self.event_2_large_end
        else:
            self.event_2_small_end = stacking_pos if stacking_pos[1] > self.event_2_small_end[1] else self.event_2_small_end
        
        if drop_success == 0:
            self.rxarm.gripper.release()
            self.next_state = "idle"
            print("Release failed - Going back to idle")  

    def event_3(self):
        self.current_state = "event_3"
        self.status_message = "Event 3"
        #fixed_bolck_pos_top = np.array([250, 75, 50])
        #self.rxarm.grab_fixed_point(fixed_bolck_pos_top,phi=0,theta=-np.pi)



        
        self.go_to_ready_pose()
        self.camera.save_detections = True            
        time.sleep(0.2)
        block_found = False
        # set_trace()
            # key = next(iter(sorted(self.camera.detected_blocks.keys())))
            # print(sorted(self.camera.detected_blocks.keys()))
            # (x,y,z,angle,size) = self.camera.detected_blocks[key]
        # print("Keys", sorted(self.camera.detected_blocks.keys()))

        for key in sorted(self.camera.detected_blocks.keys()):
            value = self.camera.detected_blocks.get(key)
            if value is None:
                continue
            (x,y,z,angle,size) = value
            if size < 30:
                continue
            if (abs(x - self.event_3_stacking_pos[0]) > size or abs(y - self.event_3_stacking_pos[1]) > size):
                block_found = True
                break
                # print(sorted(self.camera.detected_blocks.keys()))


        if not block_found:
            print("Empty dictionary")
            return

        print(f"Key {key}, x {x}, y {y}, z {z}, angle {angle}, size {size}")

        self.camera.save_detections = False
        grab_success, pitch = self.rxarm.grab_block(np.array([x,y,z]), size, angle)
        
        self.go_to_ready_pose()
        time.sleep(0)

        #if not self.camera.gripper_poll():
        #    print("Block not grabbed - trying again")
        #    self.rxarm.gripper.release()
        #    return
        #else:
        #    print("block in gripper, proceding")

        stacking_pos = self.event_3_stacking_pos
        tmp_stacking_pos = stacking_pos

        max_z = 0
        for (x,y,z,_,size) in self.camera.detected_blocks.values():
            if size > 30 and z > max_z and abs(x - self.event_3_stacking_pos[0]) < size and abs(y - self.event_3_stacking_pos[1]) < size:
                stacking_pos = np.array([x,y,z])
                max_z = z
                print(f"max z: {max_z}")

        stacking_pos = (tmp_stacking_pos + stacking_pos)/2
        stacking_pos[2] = self.event_3_stacking_pos[2]
        # if max_z == 0:
            # input("Max Z zero: Press Enter to continue...")

        # time.sleep(0.5)

        if not grab_success:
            print("event 3 - Grab failed")
            return
        self.camera.save_detections = False

        if self.event_3_stacking_pos[2] < 100:
            drop_success = self.rxarm.drop_block(stacking_pos, size, angle, pitch)
        else :
            drop_success = self.rxarm.drop_block(stacking_pos - np.array([0, 0, 5]), size, angle, 0)
        self.rxarm.gripper.release()
        self.event_3_stacking_pos[2] += 40

        # if drop_success == 1:
            # self.event_3_stacking_pos[2] += 38
        if drop_success == 0:
            self.rxarm.gripper.release()
            self.next_state = "idle"
            print("Release failed - Going back to idle")    
        

    def go_to_ready_pose(self):
        # print("executing move to ready pose")
        # time.sleep(0.5)
        ready_pos = [0, -1.2, 0.8, 0, 0]
        move_time = self.rxarm.compute_move_time(ready_pos)
        self.rxarm.set_moving_time(move_time)
        # print("move time = ", move_time)
        self.rxarm.set_positions(ready_pos, is_blocking=True)
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