import gym
from gym import spaces
import math
import numpy as np
import random
import pybullet as p
import time
import pybullet_data

class TurtleBot(gym.Env):
    def __init__(self, sim_active):
        super(TurtleBot, self).__init__()
        self.sim_status = sim_active
        if(self.sim_status==1):
            physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
        else:
            physicsClient = p.connect(p.DIRECT)#or p.DIRECT for non-graphical version
        
        self.MAX_EPISODE = 10000
        self.x_threshold = 10
        
        high = np.array([self.x_threshold, self.x_threshold],
                        dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([ 3, 3 ])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.steps_left = self.MAX_EPISODE
        self.state = [0, 0]
        self.x_target = [0.5, 0.5]
        self.start_simulation()
        
    def step(self, action):
        ## Send Commands to Actuator
        if(action[0]==0):
            speed_left = 1
        elif(action[0]==1):
            speed_left = 0
        else:
            speed_left = -1
        if(action[0]==0):
            speed_right = 1
        elif(action[0]==1):
            speed_right = 0
        else:
            speed_right = -1
        
        p.setJointMotorControl2(self.boxId,0,p.VELOCITY_CONTROL,targetVelocity=speed_left*20,force=1000)
        p.setJointMotorControl2(self.boxId,1,p.VELOCITY_CONTROL,targetVelocity=speed_right*20,force=1000)
        
        ## Update Simulations
        p.stepSimulation()
        time.sleep(1./240.)

        ## Read Sensors or Link Information
        (linkWorldPosition,
            linkWorldOrientation,
            localInertialFramePosition,
            localInertialFrameOrientation,
            worldLinkFramePosition,
            worldLinkFrameOrientation,
            worldLinkLinearVelocity,
            worldLinkAngularVelocity) = p.getLinkState(self.boxId,0, computeLinkVelocity=1, computeForwardKinematics=1)
        
        self.state = [linkWorldPosition[0], linkWorldPosition[1]]
        done = bool(
            self.steps_left<0
        )

        if not done:
            error = np.array(self.state) - np.array(self.x_target)
            reward = - np.linalg.norm(error)**2
        else:
            reward = -100
        if not done:
            self.steps_left = self.steps_left-1
        self.act = action
        self.cur_done = done
        return np.array([self.state]), reward, done, {}
        #return np.array([self.state]), reward, done

    def start_simulation(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

        ## Setup Physics
        p.setGravity(0,0,-10)

        ## Load Plane
        planeId = p.loadURDF("plane.urdf")

        ## Load Robot
        startPos = [self.state[0],self.state[1],0.0]
        startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.boxId = p.loadURDF("turtlebot/turtlebot.urdf",startPos, startOrientation)
        # boxId = p.loadURDF("biped/biped2d_pybullet.urdf",startPos, startOrientation)

    def reset(self):
        p.resetSimulation()
        self.start_simulation()
        self.state = [0,0]
        self.steps_left = self.MAX_EPISODE
        return np.array([self.state])

    def render(self, mode='human'):
        print(f'State {self.state}, action: {self.act}, done: {self.cur_done}')