#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)# 不显示DeprecationWarning
""" main script """
import config
import numpy as np
from pyrep.objects.shape import Shape
import gymnasium as gym
from pyrep.robots.mobiles.tracker import Tracker
from pyrep.robots.mobiles.target import Target
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.proximity_sensor import ProximitySensor
from os.path import dirname, join, abspath
import random
from copy import copy
import numpy as np
from PIL import Image
import time
import torch
from pyrep import PyRep
from tianshou.utils.net.common import Net
from gymnasium import spaces
import cv2
from Segment_net import get_segment_model
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
import matplotlib.pyplot as plt
from torchvision.utils import save_image
SCENE_FILE = join(dirname(abspath(__file__)),
                  'Safe_v0.ttt')

Initial_positions_target=[[-0.55,-3.2,0.048],
                          [-1.9,-2.8,0.048],
                          [-2.4,-2.3,0.048],
                          [-1.8,0.6,0.048],
                          [1.8,2.5,0.048],
                          [-4.3,0,0.048],
                          ]
Initial_positions_tracker=[[-0.55,-2.2,0.048],
                           [-1.9,-1.6,0.048],
                           [-2.4,-1.3,0.048],
                           [-1.8,1.6,0.048],
                           [1.8,3.5,0.048],
                           [-4.3,1,0.048],
                           ]



class Track(gym.Env):
    # metadata = {
    #     "name": "tictactoe_v3",
    #     "is_parallelizable": False,
    #     "render_fps": 1,
    # }
    metadata = {
        "name": "VAT_v1",
        "is_parallelizable": True,
    }
    def __init__(
        self,headless:bool = True,
    ):
        super().__init__()
               # ——————参数设置——————
        self.keep_track_counter = 0
        self.last_position = 0
        self.last_distance=1.0
        self.last_angle_diff=0
        self.baseline=5
        self.w1=0.4
        self.w2=0.6
        self.best_distance=0.7
        self.best_angle=0
        self.max_distance=3.0
        self.max_angle=30
        self.last_driving_angle=0
        self.headless=headless
        self.time=0

        self.count=0
        self.save_fig_step=0

        self.segment_model=get_segment_model(True)


        self.maxDetectDistance=1
        self.minDetectDistance=0.2
        self.stop_detection=0
        self.action_conduct_time=100
        self.target_action_flag=0
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
                        low=0, high=255, shape=(3, 250, 250), dtype=np.float32
                    )


    def seed(self, seed=None):
       np.random.seed(seed)

    def get_laser_data(self):
        laser_data=[]
        for i in range(16):
            data=self.usensors[i].read()
            if data != -1:
                if data < self.minDetectDistance:
                    data=self.minDetectDistance
                data=1-((data-self.minDetectDistance)/(self.maxDetectDistance-self.minDetectDistance))
            else:
                data=1
            laser_data.append(data)
        return laser_data

    def target_motion_control(self,action_conduct_time,target_action_flag):# 控制target运动
        target_action = {'forward': [5, 5], 'turn_left': [-1.2, 1.2], 'turn_right': [1.2, -1.2]}

        option = random.choices([0,1,2],weights=[0.2,0.4,0.4])
        #print("状态变化前：", action_conduct_time, target_action_flag,option)
        if option[0]== 0 and action_conduct_time <= 0:
            target_action_flag = 0
            action_conduct_time = 300
            #print("前车直行")
            # vrepInterface.move_target_wheels(target_action['forward'][0], target_action['forward'][1])
        elif option[0] == 1 and action_conduct_time <= 0:
            target_action_flag = 1
            action_conduct_time = 300
            #print("前车左转")
        # vrepInterface.move_target_wheels(target_action['turn_left'][0], target_action['turn_left'][1])
        elif option[0] == 2 and action_conduct_time <= 0:
            #print("前车右转")
            target_action_flag = 2
            action_conduct_time = 300
            # vrepInterface.move_target_wheels(target_action['turn_right'][0], target_action['turn_right'][1])

        if target_action_flag == 0:
            self.target.set_joint_target_velocities([target_action['forward'][0], target_action['forward'][1]])
            action_conduct_time -= 1  # 执行时间减1
        elif target_action_flag == 1:
            action_conduct_time -= 1  # 执行时间减1
            if action_conduct_time >= 250:
                self.target.set_joint_target_velocities([target_action['turn_left'][0], target_action['turn_left'][1]])
            else:
                self.target.set_joint_target_velocities([target_action['forward'][0], target_action['forward'][1]])
        elif target_action_flag == 2:
            action_conduct_time -= 1  # 执行时间减1
            if action_conduct_time >= 250:
                self.target.set_joint_target_velocities([target_action['turn_right'][0], target_action['turn_right'][1]])
            else:
                self.target.set_joint_target_velocities([target_action['forward'][0], target_action['forward'][1]])
        #print("状态变化后：", action_conduct_time, target_action_flag, option)
        return action_conduct_time,target_action_flag
    def show_segment(self,observation,pause_time):
        img_transform = Compose([
        # transforms.Resize(config['input_h'], config['input_w']),
            transforms.Normalize(),
        ])
        input=cv2.resize(observation,(256,256)) # input的像素值在0-1之间
        input = np.rot90(input, 3)
        plt.subplot(1,2,1)
        plt.imshow(input)
        # input = input.transpose(2, 0, 1)
        # input=np.array([input[2],input[1],input[0]])
        # input = input.transpose(1, 2, 0)
        input=input*255
        # if self.save_fig_step %5 ==0:
        #     cv2.imwrite('./outputs/get_{}.png'.format(self.count),input)
        augmented=img_transform(image=input)
        input=augmented['image']
        input = input.astype('float32')/255
        input = input.transpose(2, 0, 1)
        input=torch.from_numpy(input)
        input=input.cuda()
        input=input.reshape([1,3,256,256])
        self.segment_model=self.segment_model.cuda()
        output=self.segment_model(input)
        obs=output[0].detach().cpu()
        obs=obs.numpy()
        obs=np.transpose(obs,(1,2,0))
        plt.subplot(1,2,2)
        plt.imshow(obs,cmap='gray')
        plt.pause(pause_time)

    def get_state(self):
        self.save_fig_step+=1
        observation = self.kinect.capture_rgb()
        observation = np.rot90(observation, 3)  
        observation = np.rot90(observation, 3)  
        observation = np.rot90(observation, 3)  # 需要旋转270度才能变成标准的图像，与相机获得的图像匹配
        observation_return = observation.transpose(2, 0, 1)
        self.show_segment(observation,0.01)

        return observation_return


    def step(self, action):
        self.action_conduct_time,self.target_action_flag=self.target_motion_control(self.action_conduct_time,self.target_action_flag)
        action_index=config.valid_actions[action]
        # print("action:",action_index)
        x_vel = config.valid_actions_dict[action_index][0]
        z_vel = config.valid_actions_dict[action_index][1]
        wheel_dist = 0.14
        wheel_radius = 0.036
        left_vel = x_vel - z_vel * wheel_dist / 2
        right_vel = x_vel + z_vel * wheel_dist / 2
        final_left_vel = left_vel / wheel_radius
        final_right_vel = right_vel / wheel_radius
        #print("action:", [final_left_vel, final_right_vel])
        self.tracker.set_joint_target_velocities([final_left_vel / 2, final_right_vel / 2])
        self.action_conduct_time,self.target_action_flag=self.target_motion_control(self.action_conduct_time,self.target_action_flag)
        self.pr.step() # Step the physics simulation
        #time.sleep(0.1)
        next_state = self.get_state()
        #print(self.get_laser_data())
        # 获得奖励值及其他状态标志
        reward, done, truncated= self.get_reward()
        target_location=self.target.get_2d_pose()
        tracker_location=self.tracker.get_2d_pose()
        if target_location[0]>24 or target_location[1]>24 or tracker_location[0]>24 or tracker_location[1] >24:
            done=True
            print("——————————————————Robot is out of the map !—————————————————————————")

        return next_state, reward, done, truncated, {}
    def get_reward(self):
        done = False
        truncated=False
        collision_obstacles =  self.obstacles.check_collision(None) 
        collision_boder =  self.boder.check_collision(None) 
        collision= collision_obstacles or collision_boder
        # 前后车之间的距离，是否丢失前车，前后车相对角度
        distance, in_range_flag, angle,driving_angle = self.if_in_range()
        #print("前车与后车的距离为：",distance,"前车与后车的角度差为：",angle,"度")
        if collision == 1:  # 如果发生了碰撞
            done = True
            #print("发生了碰撞")
            reward = -100
            self.keep_track_counter = 0
            return reward, done, truncated
            # print("因碰撞得到的惩罚值是:", reward)
        if distance <0.4:
            #print("两车距离过近，发生碰撞")
            done = True
            reward = -100
            self.keep_track_counter = 0
            # print("因超出视野范围得到的惩罚值是：", reward)
            return reward, done, truncated
        if in_range_flag == 0:  # 如果前车超出了后车的视野范围
            #print("跟丢了")
            truncated = True
            reward = -100
            self.keep_track_counter = 0
            # print("因超出视野范围得到的惩罚值是：", reward)
            return reward, done, truncated
        else:
            #在视野范围内
            self.keep_track_counter+=1
            # reward =(self.last_angle_diff-angle)*(self.last_distance-distance)*(50/(np.abs(angle)*np.abs(distance)+1))-self.baseline
            # reward = 50 / (np.abs(angle) * np.abs(distance) + 1) - self.baseline #-4.18~45
            reward = 1-self.w1*np.abs(distance-self.best_distance)/self.max_distance - self.w2*np.abs(angle-self.best_angle)/self.max_angle #0~1

            #reward+= self.keep_track_counter*0.1
            reward += -0.4 if (self.last_distance-distance) <0 else 0.4#距离变大增加惩罚
            reward += -0.3 if (self.last_angle_diff - angle) <0 else 0.3#角度偏离增加惩罚
            reward += -0.3 if (self.last_driving_angle - driving_angle) < 0 else 0.3  # 角度偏离增加惩罚
            # if (self.last_distance-distance) <0:
            #     print("距离相较变大，上一时刻两机器人之间的距离为：",self.last_distance)
            # if (self.last_angle_diff - angle) <0:
            #     print("相对角度相较变大，上一时刻两机器人之间的相对角度为：",self.last_angle_diff)
            # if (self.last_driving_angle - driving_angle) <0:
            #     print("行驶方向差相较变大，上一时刻两机器人之间的行驶方向差为：",self.last_driving_angle)
            new_position = self.tracker.get_2d_pose()
            now_position=[new_position[0],new_position[1]]
            self.last_position = now_position
        self.last_distance = distance
        self.last_angle_diff = angle
        self.last_driving_angle = driving_angle
        return reward*20, done, truncated

    def check_relations(self,x=None,y=None):
        """ judge if the people in  the range of kinect on the robot"""
        tracker_location = self.tracker.get_2d_pose()
        target_location = self.target.get_2d_pose()
        if x != None and y!=None:
            target_location[0]=x
            target_location[1]=y
        relative_pos = [target_location[0] - tracker_location[0], target_location[1] - tracker_location[1]]
        distance = np.sqrt(relative_pos[0] * relative_pos[0] + relative_pos[1] * relative_pos[1])
        tracker_direction = tracker_location[2] / np.pi * 180 - 90  # 因为车辆自身坐标系与地图坐标系存在差异，所以需要减去90度修正
        if tracker_direction < 0:
            tracker_direction += 360
        target_direction = target_location[2] / np.pi * 180 - 90  # 因为车辆自身坐标系与地图坐标系存在差异，所以需要减去90度修正
        if target_direction < 0:
            target_direction += 360
        #print("tracker_direction:", tracker_direction)
        #print("target_direction:", target_direction)
        #  得到target相对于tracker的角度
        target_to_tracker_angle = np.arctan(relative_pos[1] / relative_pos[0])  # 得到的是弧度值

        if relative_pos[0] > 0 and relative_pos[1] >0:
            target_to_tracker_angle = target_to_tracker_angle / np.pi * 180
        elif relative_pos[0] < 0 and relative_pos[1] > 0:
            target_to_tracker_angle = 180 + target_to_tracker_angle / np.pi * 180
        elif relative_pos[0] < 0 and relative_pos[1] < 0:
            target_to_tracker_angle = 180 + target_to_tracker_angle / np.pi * 180
        elif relative_pos[0] >= 0 and relative_pos[1] < 0:
            target_to_tracker_angle = 360 + target_to_tracker_angle / np.pi * 180
        elif relative_pos[0] == 0 and relative_pos[1] >0:
            target_to_tracker_angle=90
        elif relative_pos[0] == 0 and relative_pos[1] <0:
            target_to_tracker_angle=-90
        elif relative_pos[0]>0 and relative_pos[1] == 0:
            target_to_tracker_angle=0
        elif relative_pos[0] <0 and relative_pos[1] == 0:
            target_to_tracker_angle=180

        #print("target_to_tracker_angle:", target_to_tracker_angle)
        #print("tracker_direction",tracker_direction)
        # target目标位置与tracker形成的夹角与tracker的行进方向之间的关系
        angles_between_target_tracker = min(np.abs(target_to_tracker_angle - tracker_direction),
                                            np.abs(np.abs(target_to_tracker_angle - tracker_direction) - 360))
        #print("angles_between_target_tracker:",angles_between_target_tracker)

        return distance, angles_between_target_tracker
    def if_in_range(self):
        """ judge if the people in  the range of kinect on the robot"""
        flag = 0
        tracker_location = self.tracker.get_2d_pose()
        target_location = self.target.get_2d_pose()
        relative_pos = [target_location[0] - tracker_location[0], target_location[1] - tracker_location[1]]
        distance = np.sqrt(relative_pos[0] * relative_pos[0] + relative_pos[1] * relative_pos[1])
        tracker_direction = tracker_location[2] / np.pi * 180 - 90  # 因为车辆自身坐标系与地图坐标系存在差异，所以需要减去90度修正
        if tracker_direction < 0:
            tracker_direction += 360
        target_direction = target_location[2] / np.pi * 180 - 90  # 因为车辆自身坐标系与地图坐标系存在差异，所以需要减去90度修正
        if target_direction < 0:
            target_direction += 360
        #print("tracker_direction:", tracker_direction)
        #print("target_direction:", target_direction)
        #  得到target相对于tracker的角度
        target_to_tracker_angle = np.arctan(relative_pos[1] / relative_pos[0])  # 得到的是弧度值

        if relative_pos[0] > 0 and relative_pos[1] >0:
            target_to_tracker_angle = target_to_tracker_angle / np.pi * 180
        elif relative_pos[0] < 0 and relative_pos[1] > 0:
            target_to_tracker_angle = 180 + target_to_tracker_angle / np.pi * 180
        elif relative_pos[0] < 0 and relative_pos[1] < 0:
            target_to_tracker_angle = 180 + target_to_tracker_angle / np.pi * 180
        elif relative_pos[0] >= 0 and relative_pos[1] < 0:
            target_to_tracker_angle = 360 + target_to_tracker_angle / np.pi * 180
        elif relative_pos[0] == 0 and relative_pos[1] >0:
            target_to_tracker_angle=90
        elif relative_pos[0] == 0 and relative_pos[1] <0:
            target_to_tracker_angle=-90
        elif relative_pos[0]>0 and relative_pos[1] == 0:
            target_to_tracker_angle=0
        elif relative_pos[0] <0 and relative_pos[1] == 0:
            target_to_tracker_angle=180

        #print("target_to_tracker_angle:", target_to_tracker_angle)
        # print("tracker_direction",tracker_direction)
        # target目标位置与tracker形成的夹角与tracker的行进方向之间的关系
        angles_between_target_tracker = min(np.abs(target_to_tracker_angle - tracker_direction),
                                            np.abs(np.abs(target_to_tracker_angle - tracker_direction) - 360))

        # 判断是否在视野角度内
        if np.abs(angles_between_target_tracker) <= 30 and distance >= 0.3 and distance <= 3.0:
            flag = 1

        driving_angle = target_direction - tracker_direction


        return distance, flag, angles_between_target_tracker, driving_angle

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

    def reset(self, seed=None, options=None):
        if self.time ==0:
        #——————启动Pyrep——————
            self.pr = PyRep()
            self.pr.launch(SCENE_FILE, self.headless)
            self.pr.start()
            self.pr.step() # Step the physics simulation
        #——————创建环境中的实体——————
            self.usensors=[]
            for i in range(1,17):
                self.usensors.append(ProximitySensor(f'Tracker_ultrasonicSensor{i}'))
            self.target = Target()
            self.tracker=Tracker()
            self.kinect = VisionSensor('kinect_rgb')
            self.obstacles=Shape("Obstacles")
            self.boder=Shape("boder")
            #self.obstacles=Shape("Obstacles")
            #self.boder=Shape("boder")

        #——————初始化——————
            self.tracker.set_control_loop_enabled(False)
            self.tracker.set_motor_locked_at_zero_velocity(True)
            self.target.set_control_loop_enabled(False)
            self.target.set_motor_locked_at_zero_velocity(True)
            self.initial_tracker_positions = self.tracker.get_2d_pose()
            self.initial_target_positions = self.target.get_2d_pose()
            self.seed()
        self.time+=1
        if self.time%20==0:
            self.pr.stop()
            self.pr.start()
        index=random.randint(0, len(Initial_positions_target)-1)
        self.tracker.set_2d_pose(Initial_positions_tracker[index])
        self.target.set_2d_pose(Initial_positions_target[index])
        self.target.set_motor_locked_at_zero_velocity(True)
        self.tracker.set_motor_locked_at_zero_velocity(True)
        self.action_conduct_time=100
        self.target_action_flag=0
        return self.get_state(),{}


    def close(self):
        # self.pr.shutdown()
        if self.time>0:
            self.pr.shutdown()

        


    def render(self):
        self.pr.launch(SCENE_FILE, False)
        self.pr.start()
#if __name__ == "__main__":
    #env = VAT_env(True)
    #parallel_api_test(env, num_cycles=3)
