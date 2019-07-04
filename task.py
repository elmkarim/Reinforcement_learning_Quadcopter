import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):

        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal: reach position (0,0,10)
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #Penalty on position
        penalty_pos = (abs(self.sim.pose[:3] - self.target_pos)).sum()
        
        #Penalty on angles varying from 0 (quadcopter should stay horizontal while hovering)
        penalty_angle = np.abs(np.sin(self.sim.pose[3:6])).sum()

        
        #Add penalty on angle velocity
        penalty_angle_v = np.abs(self.sim.angular_v).sum()

        
        #Penality on velocity
        penalty_v = np.abs(self.sim.v).sum()
        
        #Penalty only on z
        penalty_pos_z =  (abs(self.sim.pose[2:3] - self.target_pos[2:3])).sum()
       
        
#         penalty = 0.01 * (0.3*penalty_pos + 0.1*penalty_angle + 0.1*penalty_angle_v + 0.1*penalty_v)
#         penalty = 0.1 * penalty_pos_z
        
#         penalty = np.clip(penalty, 0, 1)


#         penalty = 0.01 * (0.1*penalty_pos + 0.3*penalty_angle + 0.3*penalty_angle_v)
#         reward = 1. - penalty 
        
#         reward = np.tanh(reward)

        reward = 1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos)
        reward = np.tanh(reward)).sum()
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state