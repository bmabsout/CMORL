import pathlib
import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import pybullet
from collections import deque

# Constants to define training and visualisation.

BOUND_ANG = 110         # Joint maximum angle (deg)
STEP_ANGLE = 11           # Maximum angle (deg) delta per step
ANG_FACTOR = 0.1          # Improve angular velocity resolution before clip.

# Values for randomization, to improve sim to real transfer.
RANDOM_GYRO = 0           # Percent
RANDOM_JOINT_ANGS = 0      # Percent
RANDOM_MASS = 0           # Percent, currently inactive
RANDOM_FRICTION = 0       # Percent, currently inactive

LENGTH_RECENT_ANGLES = 3  # Buffer to read recent joint angles
LENGTH_HISTORY = 3 # Number of steps to state history

# SIZE_OBSERVATION = 3+3+8
OBSERVATION_HISTORY_ANGLES = 1
SIZE_OBSERVATION = 3+3 + 8*OBSERVATION_HISTORY_ANGLES
# TOTAL_OBSERVATION = SIZE_OBSERVATION*LENGTH_HISTORY
TOTAL_OBSERVATION = SIZE_OBSERVATION


class OpenCatGymEnv(gym.Env):
    """ Gymnasium environment (stable baselines 3) for OpenCat robots.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, render_mode=None):
        self.step_counter = 0
        self.state_history = deque(maxlen=LENGTH_HISTORY)
        self.recent_angles = deque([[0.0]*8]*LENGTH_RECENT_ANGLES, maxlen=LENGTH_RECENT_ANGLES)
        self.bound_ang = np.deg2rad(BOUND_ANG)

        if render_mode=="human":
            p.connect(p.GUI)
            # Uncommend to create a video.
            #video_options = ("--width=960 --height=540 
            #                + "--mp4=\"training.mp4\" --mp4fps=60")
            #p.connect(p.GUI, options=video_options) 
        else:
            # Use for training without visualisation (significantly faster).
            p.connect(p.DIRECT)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=0.5, 
                                     cameraYaw=-170, 
                                     cameraPitch=-40, 
                                     cameraTargetPosition=[0.4,0,0])

        # The action space are the 8 joint angles.
        self.action_space = gym.spaces.Box(np.array([-1]*8), np.array([1]*8))

        high = np.ones(TOTAL_OBSERVATION)
        self.observation_space = gym.spaces.Box(high = high, low = -high)

    def _get_obs(self, action):
        state_vel, state_angvel = map(np.asarray, p.getBaseVelocity(self.robot_id))
        state_vel = np.clip(state_vel, -1.0 , 1.0)
        state_angvel_clip = np.clip(state_angvel*ANG_FACTOR, -1.0, 1.0)
        # self.state_robot = np.concatenate((state_vel, state_angvel_clip, action))
        # time_obs = np.fmod(self.step_counter/100.0, 1.0)
        self.recent_angles.appendleft(self.get_observed_joints())
        self.state_robot = np.concatenate([state_vel, state_angvel_clip] + list(self.recent_angles)[0:OBSERVATION_HISTORY_ANGLES])#, [time_obs]))
        # self.state_history.appendleft(self.state_robot)
        # obs = np.array((self.state_history[0:2] + self.state_history[2::3])[:LENGTH_HISTORY]).flatten()
        # padded_obs = np.zeros(TOTAL_OBSERVATION)
        # padded_obs[:len(obs)] = obs
        # return padded_obs
        return self.state_robot

    def get_joint_angs(self):
        return np.asarray(p.getJointStates(self.robot_id, self.joint_ids),
                                                   dtype=object)[:,0]
    
    def low_resolution_angs(self, joint_angs):
        # Transform angle to degree and perform rounding, because 
        # OpenCat robot have only integer values.
        joint_angsDeg = np.rad2deg(joint_angs.astype(np.float64))
        joint_angsDegRounded = joint_angsDeg.round()
        return np.deg2rad(joint_angsDegRounded)

    def get_observed_joints(self):
        return self.low_resolution_angs(self.get_joint_angs())/self.bound_ang

    def control_motors(self, action):
        joint_angs = self.get_joint_angs()
        ds = np.deg2rad(STEP_ANGLE) # Maximum change of angle per step
        with_action = joint_angs + np.clip(action*self.bound_ang - joint_angs, -1.0, 1.0) * ds # Change per step including agent action
        new_joint_angs = np.clip(self.low_resolution_angs(with_action), -self.bound_ang, self.bound_ang)
        # # Simulate delay for data transfer. Delay has to be modeled to close 
        # # "reality gap").
        # p.stepSimulation()
        # Set new joint angles
        p.setJointMotorControlArray(
            self.robot_id, 
            self.joint_ids, 
            p.POSITION_CONTROL, 
            new_joint_angs, 
            forces=np.ones(8)*0.2
        )


    def step(self, action):
        self.step_counter += 1
        # p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        last_position = p.getBasePositionAndOrientation(self.robot_id)[0][0]
        
        self.control_motors(action)
        # p.stepSimulation() # Delay of data transfer
        p.stepSimulation() # Emulated delay of data transfer via serial port
        self.observation = self._get_obs(action)
        current_position = p.getBasePositionAndOrientation(self.robot_id)[0][0]
        movement_forward = current_position - last_position
        # joints = np.clip(np.mean(1.0 - np.abs(action)), 1e-3, 1.0)**0.5
        forward = np.clip(movement_forward*300, 1e-3, 1.0)
        angular_velocity = np.asarray(p.getBaseVelocity(self.robot_id)[1])
        body_stability = 1.0 - np.clip(angular_velocity*ANG_FACTOR, 0.0, 1.0)
        change_direction = np.sign(self.recent_angles[0]-self.recent_angles[1]) == np.sign(self.recent_angles[1]-self.recent_angles[2])
        # reward = (forward*change_direction*body_stability*joints)**(1.0/4.0)
        reward = 0.0
        # Set state of the current state.
        terminated = False
        truncated = False
        info = {"forward": forward, "change_direction": change_direction, "body_stability": body_stability}
        # Stop criteria of current learning episode: 
        if self.is_fallen(): # Robot fell
            terminated = True
            truncated = False

        return (np.array(self.observation).astype(np.float32), 
                        reward, terminated, truncated, info)


    def reset(self, seed=None, options=None):
        self.step_counter = 0
        self.arm_contact = 0
        p.resetSimulation()
        # Disable rendering during loading.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) 
        p.setGravity(0,0,-9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        start_pos = [0,0,0.08]
        start_orient = p.getQuaternionFromEuler([0,0,0])

        urdf_path = pathlib.Path(__file__).parent.resolve() / "models/"
        self.robot_id = p.loadURDF(str(urdf_path / "bittle_esp32.urdf"), 
                                   start_pos, start_orient, 
                                   flags=p.URDF_USE_SELF_COLLISION) 
        
        # Initialize urdf links and joints.
        self.joint_ids = []
        #paramIds = []
        for j in range(p.getNumJoints(self.robot_id)):
            joint_type = p.getJointInfo(self.robot_id, j)[2]

            if (joint_type == p.JOINT_PRISMATIC 
                or joint_type == p.JOINT_REVOLUTE):
                self.joint_ids.append(j)
                #paramIds.append(p.addUserDebugParameter(joint_name.decode("utf-8")))
                # Limiting motor dynamics. Although bittle's dynamics seem to 
                # be be quite high like up to 7 rad/s.
                p.changeDynamics(self.robot_id, j, maxJointVelocity = np.pi*10) 
        
        # Setting start position. This influences training.
        joint_angs = np.deg2rad(np.array([1, 0, 1, 0, 1, 0, 1, 0])*50) 

        for i, joint_id in enumerate(self.joint_ids):
            p.resetJointState(self.robot_id, joint_id, joint_angs[i])

        # Initialize robot state history with reset position
        state_joints = np.asarray(
            p.getJointStates(self.robot_id, self.joint_ids), dtype=object)[:,0]
        state_joints /= self.bound_ang 
        
        self.state_history = deque(maxlen=LENGTH_HISTORY)
        self.recent_angles = deque([[0.0]*8]*LENGTH_RECENT_ANGLES, maxlen=LENGTH_RECENT_ANGLES)
        self.observation = self._get_obs(self.action_space.low*0.0)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        info = {}
        return np.array(self.observation).astype(np.float32), info


    def render(self, mode='human'):
        pass


    def close(self):
        p.disconnect()


    def is_fallen(self):
        """ Check if robot is fallen. It becomes "True", 
            when pitch or roll is more than 1.3 rad.
        """
        pos, orient = p.getBasePositionAndOrientation(self.robot_id)
        orient = p.getEulerFromQuaternion(orient)
        is_fallen = (np.fabs(orient[0]) > 1.3 
                    or np.fabs(orient[1]) > 1.3)

        return is_fallen


    def randomize(self, value, percentage):
        """ Randomize value within percentage boundaries.
        """
        percentage /= 100
        value_randomized = value * (1 + percentage*(2*np.random.rand()-1))

        return value_randomized


gym.register("Bittle-custom", OpenCatGymEnv)