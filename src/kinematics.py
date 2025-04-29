"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm

import pdb


def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def FK_dh(dh_params,link=[0, 0, 0, 1]):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """

    # if joint_angles is None:
    #     joint_angles = np.zeros(len(dh_params))
    # else:
    #     joint_angles = np.append(dh_params[0,3], joint_angles)
    #     # np.insert(joint_angles, -2, dh_params[-2,3])

    T_out = np.eye(4)
    # pdb.set_trace()
    for i, row in enumerate(dh_params):
        # print(i, row)
        # print(i, joint_angles[i])

        T = get_transform_from_dh(row[0], row[1], row[2], row[3])
        T_out = T_out @ T
        

    # print(T_out)
    return T_out


def get_transform_from_dh(a, alpha, d, theta, twist_about_y=False):
    """!
    @brief      Gets the transformation matrix T from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transformation matrix.
    """
    T_theta = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                    [np.sin(theta), np.cos(theta), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    
    T_d = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, d],
                    [0, 0, 0, 1]])
    
    T_a_x = np.array([[1, 0, 0, a],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    
    T_a_y = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, -a],
                    [0, 0, 0, 1]])
    
    T_x = np.array([[1, 0, 0, 0],
                    [0, np.cos(alpha), -np.sin(alpha), 0],
                    [0, np.sin(alpha), np.cos(alpha), 0],
                    [0, 0, 0, 1]])
    
    T_y = np.array([[np.cos(alpha), 0, np.sin(alpha), 0],
                    [0, 1, 0, 0],
                    [-np.sin(alpha), 0, np.cos(alpha), 0],
                    [0, 0, 0, 1]])
    
    # if twist_about_y:
    #     T = T_theta @ T_d @ T_a_y @ T_y
    # else:
    T = T_theta @ T_d @ T_a_x @ T_x

    return T


def get_euler_angles_from_T(T):
    """!
    @brief      Gets the euler angles from a transformation matrix.

                TODO: Implement this function return the 3 Euler angles from a 4x4 transformation matrix T
                If you like, add an argument to specify the Euler angles used (xyx, zyz, etc.)

    @param      T     transformation matrix

    @return     The euler angles from T.
    """
    rot_M = T[0:3, 0:3]

    phi = np.arctan2(-rot_M[1, 2], rot_M[2, 2])
    theta = np.arcsin(rot_M[0, 2])
    psi = np.arctan2(-rot_M[0, 1], rot_M[0, 0])

    return np.array([phi, theta, psi])


def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the 6DOF pose vector from a 4x4 transformation matrix T

    @param      T     transformation matrix

    @return     The pose vector from T.
    """
    return T[0:3, 3]


def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a  representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4x4 homogeneous matrix representing the pose of the desired link

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4x4 homogeneous matrix representing the pose of the desired link
    """
    pass


def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """
    pass


def IK_geometric(dh_params, pose):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose vector as np.array to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose vector as np.array 

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    pass

def IK(input, phi, theta, joint_lengths, o_c_offset=0):
    
    input_pos = input.copy()
    # print("Starting IK solver with input pos:", input_pos)
    # print("input phi and theta", phi, theta)

    if input_pos[2] < 0:
        input_pos[2] = 0

    input_pos[2] -= 103.9 - 10

    joint_pos = np.zeros(5)

    psi = np.arctan2(input_pos[1], input_pos[0])
    R_z = np.array([[np.cos(psi), np.sin(psi), 0], [-np.sin(psi), np.cos(psi), 0], [0, 0, 1]])

    input_pos = R_z @ input_pos

    pos_2d = np.array([input_pos[0], input_pos[2]])

    if np.hypot(pos_2d[0], pos_2d[1]) > np.sum(joint_lengths):
        return False, joint_pos

    o_c = pos_2d - (joint_lengths[2] + o_c_offset)*np.array([np.cos(theta), np.sin(theta)])
    o_c[1] = o_c[1] + 0.035 * o_c[0]

    if np.hypot(o_c[0], o_c[1]) > np.sum(joint_lengths[:2]) or o_c[1] < 10:
        #print("IK Failed")
        return False, joint_pos

    joint_pos[0] = psi

    arccos_input = ((o_c[0]**2 + o_c[1]**2) - joint_lengths[0]**2 - joint_lengths[1]**2)/(2*joint_lengths[0]*joint_lengths[1])
    
    joint_pos[2] = -np.arccos(arccos_input)

    alpha = np.arctan2(joint_lengths[1]*np.sin(joint_pos[2]), joint_lengths[0] + joint_lengths[1]*np.cos(joint_pos[2]))
    joint_pos[1] = np.arctan2(o_c[1], o_c[0]) - alpha

    joint_pos[3] = theta - (joint_pos[1] + joint_pos[2])


    joint_pos[1] *= -1
    joint_pos[2] *= -1
    joint_pos[3] *= -1

    #TODO add roll angle value for joint_pos[4]
    joint_pos[4] = phi + joint_pos[0] if theta == -np.pi/2 else 0 #opposite of base rotation + roll cmd
    if(joint_pos[4] > np.pi/4):
        joint_pos[4] -= np.pi/2
    if(joint_pos[4] < -np.pi/4):
        joint_pos[4] += np.pi/2
    # print('joint_pos',180/np.pi*joint_pos)

    joint_pos[3] -= 0.1

    joint_pos[0] = clamp(joint_pos[0] - np.pi/2) + np.pi/2

    return True, joint_pos

