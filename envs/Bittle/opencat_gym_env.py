import pathlib
import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import pybullet

# Constants to define training and visualisation.

EPISODE_LENGTH = 250      # Number of steps for one training episode
MAXIMUM_LENGTH = 1.8e6    # Number of total steps for entire training

# Factors to weight rewards and penalties.
PENALTY_STEPS = 2e6       # Increase of penalty by step_counter/PENALTY_STEPS
FAC_MOVEMENT = 1000       # Reward movement in x-direction
FAC_STABILITY = 0.1       # Punish body roll and pitch velocities
FAC_Z_VELOCITY = 0.0      # Punish z movement of body
FAC_SLIP = 0.0            # Punish slipping of paws
FAC_ARM_CONTACT = 0.01    # Punish crawling on arms and elbows
FAC_SMOOTH_1 = 1.0        # Punish jitter and vibrational movement, 1st order
FAC_SMOOTH_2 = 1.0        # Punish jitter and vibrational movement, 2nd order
FAC_CLEARANCE = 0.0       # Factor to enfore foot clearance to PAW_Z_TARGET
PAW_Z_TARGET = 0.005      # Target height (m) of paw during swing phase

BOUND_ANG = 110         # Joint maximum angle (deg)
STEP_ANGLE = 11           # Maximum angle (deg) delta per step
ANG_FACTOR = 0.1          # Improve angular velocity resolution before clip.

# Values for randomization, to improve sim to real transfer.
RANDOM_GYRO = 0           # Percent
RANDOM_JOINT_ANGS = 0      # Percent
RANDOM_MASS = 0           # Percent, currently inactive
RANDOM_FRICTION = 0       # Percent, currently inactive

LENGTH_RECENT_ANGLES = 3  # Buffer to read recent joint angles
LENGTH_HISTORY = 10 # Number of steps to state history

# Size of oberservation space is set up of: 
# [LENGTH_JOINT_HISTORY, quaternion, gyro]
# SIZE_OBSERVATION = LENGTH_JOINT_HISTORY * 8 + 6 + 3
SIZE_OBSERVATION = 3+3+8


class OpenCatGymEnv(gym.Env):
    """ Gymnasium environment (stable baselines 3) for OpenCat robots.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, render_mode=None):
        self.step_counter = 0
        self.step_counter_session = 0
        self.state_history = np.zeros(SIZE_OBSERVATION*LENGTH_HISTORY)
        self.angle_history = np.array([])
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

        high = np.ones(SIZE_OBSERVATION*LENGTH_HISTORY)
        self.observation_space = gym.spaces.Box(high = high, low = -high)

    def _get_obs(self, action):
        state_vel, state_angvel = map(np.asarray, p.getBaseVelocity(self.robot_id))
        state_vel = np.clip(state_vel, -1.0 , 1.0)
        state_angvel_clip = np.clip(state_angvel*ANG_FACTOR, -1.0, 1.0)
        self.state_robot = np.concatenate((state_vel, state_angvel_clip, action))
        self.state_history = np.append(self.state_robot, self.state_history)[:-SIZE_OBSERVATION]
        return self.state_history

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        last_position = p.getBasePositionAndOrientation(self.robot_id)[0][0]
        joint_angs = np.asarray(p.getJointStates(self.robot_id, self.joint_id),
                                                   dtype=object)[:,0]
        # shoulder_left, elbow_left, shoulder_right, elbow_right, hip_right, knee_right, hip_left, knee_left
        ds = np.deg2rad(STEP_ANGLE) # Maximum change of angle per step

        joint_angs += np.clip(action - joint_angs, -1.0, 1.0) * ds # Change per step including agent action
        
        min_ang = -self.bound_ang
        max_ang = self.bound_ang

        joint_angs = np.clip(joint_angs, min_ang, max_ang)

        # Transform angle to degree and perform rounding, because 
        # OpenCat robot have only integer values.
        joint_angsDeg = np.rad2deg(joint_angs.astype(np.float64))
        joint_angsDegRounded = joint_angsDeg.round()
        joint_angs = np.deg2rad(joint_angsDegRounded)

        # Simulate delay for data transfer. Delay has to be modeled to close 
        # "reality gap").
        p.stepSimulation()
        # Set new joint angles
        p.setJointMotorControlArray(self.robot_id, 
                                    self.joint_id, 
                                    p.POSITION_CONTROL, 
                                    joint_angs, 
                                    forces=np.ones(8)*0.2)
        p.stepSimulation() # Delay of data transfer

        # Normalize joint_angs
        joint_angs /= self.bound_ang

        self.recent_angles = np.append(self.recent_angles, joint_angs)
        self.recent_angles = np.delete(self.recent_angles, np.s_[0:8])

        joint_angs_prev = self.recent_angles[8:16]
        joint_angs_prev_prev = self.recent_angles[0:8]

        p.stepSimulation() # Emulated delay of data transfer via serial port
        current_position = p.getBasePositionAndOrientation(self.robot_id)[0][0]
        movement_forward = current_position - last_position
        # joints = np.clip(np.mean(1.0 - np.abs(action)), 1e-3, 1.0)**0.5
        # print("actions:", )
        forward = np.clip(movement_forward*300, 1e-3, 1.0)
        body_stability = 1.0 - np.clip(np.asarray(p.getBaseVelocity(self.robot_id)[1])*ANG_FACTOR, 0.0, 1.0)
        change_direction = np.sign(joint_angs-joint_angs_prev) == np.sign(joint_angs_prev-joint_angs_prev_prev)
        # reward = (forward*change_direction*body_stability*joints)**(1.0/4.0)
        reward = 0.0
        # Set state of the current state.
        terminated = False
        truncated = False
        info = {"forward": forward, "change_direction": change_direction, "body_stability": body_stability}
        # Stop criteria of current learning episode: 
        # Number of steps or robot fell.
        self.step_counter += 1
        if self.step_counter > EPISODE_LENGTH:
            self.step_counter_session += self.step_counter
            terminated = False
            truncated = True

        elif self.is_fallen(): # Robot fell
            self.step_counter_session += self.step_counter
            reward = 0
            terminated = True
            truncated = False

        self.observation = self._get_obs(action)

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
        self.joint_id = []
        #paramIds = []
        for j in range(p.getNumJoints(self.robot_id)):
            joint_type = p.getJointInfo(self.robot_id, j)[2]

            if (joint_type == p.JOINT_PRISMATIC 
                or joint_type == p.JOINT_REVOLUTE):
                self.joint_id.append(j)
                #paramIds.append(p.addUserDebugParameter(joint_name.decode("utf-8")))
                # Limiting motor dynamics. Although bittle's dynamics seem to 
                # be be quite high like up to 7 rad/s.
                p.changeDynamics(self.robot_id, j, maxJointVelocity = np.pi*10) 
        
        # Setting start position. This influences training.
        joint_angs = np.deg2rad(np.array([1, 0, 1, 0, 1, 0, 1, 0])*50) 

        i = 0
        for j in self.joint_id:
            p.resetJointState(self.robot_id,j, joint_angs[i])
            i = i+1

        # Normalize joint angles.
        joint_angs /= self.bound_ang

        # Initialize robot state history with reset position
        state_joints = np.asarray(
            p.getJointStates(self.robot_id, self.joint_id), dtype=object)[:,0]
        state_joints /= self.bound_ang 
        
        self.state_history = np.zeros(SIZE_OBSERVATION*LENGTH_HISTORY)
        self.recent_angles = np.tile(state_joints, LENGTH_RECENT_ANGLES)
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