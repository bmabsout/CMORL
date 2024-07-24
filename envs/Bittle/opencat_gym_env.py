import pathlib
import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data

def p_mean(values, p, axis=None, slack=1e-7):
    """Calculate mean of values with power p."""
    array = np.asarray(values) + slack
    return np.mean(np.abs(array)**p, axis=axis)**(1.0/p) - slack

class OpenCatGymEnv(gym.Env):
    """Gymnasium environment for OpenCat robots."""

    metadata = {'render.modes': ['human']}

    def __init__(self, render_mode=None, observe_joints=False):
        self.observe_joints = observe_joints
        self.step_counter = 0
        
        # Constants
        self.BOUND_ANG = 110  # Joint maximum angle (degrees)
        self.MAX_ANGLE_CHANGE = 11  # Maximum angle (degrees) delta per step
        self.ANG_FACTOR = 0.1  # Improve angular velocity resolution before clip
        self.LENGTH_RECENT_ANGLES = 3  # Buffer to read recent joint angles
        
        # Observation space size
        self.SIZE_OBSERVATION = 3 + 3 + 1 + 3  # Linear vel, Angular vel, Time, Gravity Direction
        self.TOTAL_OBSERVATION = self.SIZE_OBSERVATION + (self.LENGTH_RECENT_ANGLES * self.NUM_JOINTS if observe_joints else 0)

        # Set up PyBullet
        p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=-170, cameraPitch=-40, cameraTargetPosition=[0.4,0,0])
        
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        
        # Load robot URDF
        urdf_path = pathlib.Path(__file__).parent.resolve() / "models/bittle_esp32.urdf"
        self.robot_id = p.loadURDF(str(urdf_path), [0, 0, 0.08], p.getQuaternionFromEuler([0, 0, 0]), 
                                   flags=p.URDF_USE_SELF_COLLISION)
        # Initialize joints
        def is_relevant_joint(joint_id):
            joint_type = p.getJointInfo(self.robot_id, joint_id)[2]
            return joint_type in (p.JOINT_PRISMATIC, p.JOINT_REVOLUTE)

        self.joint_ids = list(filter(is_relevant_joint, range(p.getNumJoints(self.robot_id))))
        self.NUM_JOINTS = len(self.joint_ids)
        
        for j in self.joint_ids:
            p.changeDynamics(self.robot_id, j, maxJointVelocity=np.pi*10)

        # Define action and observation spaces
        action_high = np.ones(self.NUM_JOINTS)
        self.action_space = gym.spaces.Box(-action_high, action_high)
        obs_high = np.ones(self.TOTAL_OBSERVATION)
        self.observation_space = gym.spaces.Box(-obs_high, obs_high)

    def gravity_direction(self, id):
        """Calculate the direction of gravity in the box's local coordinate system."""
        _, orientation = p.getBasePositionAndOrientation(id)
        rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        world_gravity = np.array([0, 0, -1])
        local_gravity = np.dot(rotation_matrix.T, world_gravity)
        return local_gravity / np.linalg.norm(local_gravity)

    def _get_obs(self):
        """Get the current observation."""
        gravity_direction = self.gravity_direction(self.robot_id)
        state_vel, state_angvel = map(np.asarray, p.getBaseVelocity(self.robot_id))
        state_vel = np.clip(state_vel, -1.0, 1.0)
        state_angvel_clip = np.clip(np.rad2deg(state_angvel) * self.ANG_FACTOR, -1.0, 1.0)
        time_obs = np.fmod(self.step_counter / 100.0, 1.0)
        
        obs = np.concatenate((state_vel, state_angvel_clip, gravity_direction, [time_obs]))
        
        if self.observe_joints:
            obs = np.concatenate((obs, self.recent_angles.flatten()))
        
        return obs.astype(np.float32)

    def get_joint_angs(self):
        return np.rad2deg(np.asarray(p.getJointStates(self.robot_id, self.joint_ids), dtype=object)[:,0])
    
    def control_motors(self, action):
        joint_angs = self.get_joint_angs()
        # Change per step including agent action
        with_action = joint_angs + np.clip(action*self.bound_ang - joint_angs, -1.0, 1.0) * self.MAX_ANGLE_CHANGE
        # due to the low resolution of the real servos, we need to round the angles
        low_resolution_angs = np.round(with_action)
        new_joint_angs = np.clip(low_resolution_angs, -self.bound_ang, self.bound_ang)
        p.setJointMotorControlArray(
            self.robot_id, 
            self.joint_ids, 
            p.POSITION_CONTROL, 
            new_joint_angs, 
            forces=np.ones(self.NUM_JOINTS)*0.2
        )
        return new_joint_angs / self.BOUND_ANG

    def add_joints_angles_to_history(self, joint_angles):
        self.recent_angles = np.roll(self.recent_angles, -1, axis=0)
        self.recent_angles[-1] = joint_angles

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        self.step_counter += 1

        normalized_joint_angs = self.control_motors(action)

        # Update recent angles history
        self.add_joints_angles_to_history(normalized_joint_angs)

        # get the observation before stepping the simulation because of communication delay
        observation = self._get_obs()

        last_position = p.getBasePositionAndOrientation(self.robot_id)[0][0]
        for _ in range(5): # 5 steps so the simulation runs at 240/5 = 48Hz, we cannot go faster as PWM is 50Hz
            p.stepSimulation() # runs at 240Hz, this is needed as pybullet is tuned for 240Hz operation
        current_position = p.getBasePositionAndOrientation(self.robot_id)[0][0]
        
        # Calculate reward components
        movement_forward = np.clip((current_position - last_position) * 300, 0.0, 1.0)
        body_stability = 1.0 - np.clip(np.asarray(p.getBaseVelocity(self.robot_id)[1]) * 0.2, 0.0, 1.0)
        body_stability_scalar = p_mean(body_stability, p=-1.0)
        change_direction = np.sign(self.recent_angles[-1] - self.recent_angles[-2]) == np.sign(self.recent_angles[-2] - self.recent_angles[-3])
        change_direction_scalar = p_mean(change_direction, p=0.1)
        
        # Calculate final reward
        reward = p_mean([movement_forward, change_direction_scalar, body_stability_scalar], p=-4.0)
        
        # Check termination conditions
        terminated = self.is_fallen()
        truncated = False
        
        info = {"forward": movement_forward, "change_direction": change_direction_scalar, "body_stability": body_stability_scalar}
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.step_counter = 0
        # reset robot position and orientation
        p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0.08], p.getQuaternionFromEuler([0, 0, 0]))
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])
        # Set initial joint positions
        initial_angles = np.deg2rad([50, 0, 50, 0, 50, 0, 50, 0])
        for i, joint_id in enumerate(self.joint_ids):
            p.resetJointState(self.robot_id, joint_id, initial_angles[i])
        
        # Initialize recent angles history
        self.recent_angles = np.tile(initial_angles / self.BOUND_ANG, (self.LENGTH_RECENT_ANGLES, 1))
        
        observation = self._get_obs()
        info = {}
        
        return observation.astype(np.float32), info

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()

    def is_fallen(self):
        """Check if robot is fallen (pitch or roll > 1.3 rad)."""
        _, orient = p.getBasePositionAndOrientation(self.robot_id)
        orient = p.getEulerFromQuaternion(orient)
        return abs(orient[0]) > 1.3 or abs(orient[1]) > 1.3

# Register the environment
gym.register("Bittle-custom", OpenCatGymEnv)