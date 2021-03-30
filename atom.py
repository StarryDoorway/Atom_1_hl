"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import random
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import time
from CalculateCost import CalculateCost


class AtomEnv(gym.Env):

    """
    记得去改reward threshold

    Description:
        To find the structrue of low cost;
        3 atoms for now(num_atoms = 3);
        The tool to culculate the cost is provided by  College of Chemistry , Jilin University.

    Observation:
        Type: Discrete(3 * num_atoms)
        Num	Observation               Min             Max
        0	x of atom_1                0               1
        1	y of atom_1                0               1
        2	z of atom_1                0               1
        3	x of atom_2                0               1
        ...
        8   z of atom_3                0               1



    Actions:
        Type: Discrete(6 * num_atoms)
        Num	 Action
        0	 x of atom_1  +0.01
        1	 y of atom_1  +0.01
        2	 z of atom_1  +0.01
        3	 x of atom_2  +0.01
        ...
        17   z of atom_3  -0.01


        Note: action 0~2  belongs to atom_1;3~5 belongs to atom_2 ... They all +0.01.
              action 9~11 belongs to atom_1 ... 15~17 belongs to atom_3.They all -0.01


    Reward:
        Reward is -1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [0,1]

    Episode Termination:
        1.Episode length is greater than 1000.


    """



    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.num_atoms = 1
        self.num_coordinates = self.num_atoms * 3
        #满足条件的原子结构坐标保存在found_stru中
        self.found_stru = []
        self.done = False

        #for an atom , it has three dimentional coordinates(x,y,z) ,and each x,y,z can "+0.01" or "-0.01"
        high = np.ones(self.num_coordinates)
        low = np.zeros(self.num_coordinates)

        self.action_space = spaces.Discrete(2 * self.num_coordinates)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        done = False

        changedCoordinate = action % self.num_coordinates
        if action < self.num_coordinates:
            self.state[changedCoordinate] += 0.01
        else:
            self.state[changedCoordinate] -= 0.01


        if self.state[changedCoordinate] > 1:
            self.state[changedCoordinate] -= 1
        elif self.state[changedCoordinate] < 0:
            self.state[changedCoordinate] += 1

        coordinates_for_cost = self.state.reshape((self.num_atoms,3))
        cost = CalculateCost('R-3M', [20,20,20,90,90,120],coordinates_for_cost)
        reward = -1
        # print('cost: ',cost,'   state:',self.state, '   action: ',action)

        fo = open("cost_record.txt", "w")
        cost_record = 'cost: ' + str(cost) + '   state:' + str(self.state) + '   action: ' + str(action) + '\n'
        fo.write(cost_record)
        fo.close()
        #迭代5000步后结束，cost满足条件时，保存坐标在found_structure中
        if cost > 70:
            reward = -1
        # elif cost > 50:
        #     reward = 1
        # elif cost > 30:
        #     reward = 2
        elif cost < 70:
            print(self.state)
            reward = 1000
            done = True
            self.found_stru.append(self.state)
            fo = open("./found_structure/found_stru.txt", "a+")
            print(cost)
            goal = str(self.found_stru[-1])+'\t' +str(cost) + '\n'
            fo.write(goal)
            fo.close()



        # 在主流程中控制具体跑多少步,如果找到了n个结构以上就算合格了,具体再改


        return np.array(self.state), reward, done, {}

    def reset(self):
        #初始化state
        self.state = self.np_random.uniform(low=0, high=1, size=(self.num_coordinates ,))
        self.state = np.round(self.state, 2)

        return np.array(self.state)

    def render(self, mode='human'):
        # screen_width = 600
        # screen_height = 400
        #
        # world_width = self.x_threshold * 2
        # scale = screen_width/world_width
        # carty = 100  # TOP OF CART
        # polewidth = 10.0
        # polelen = scale * (2 * self.length)
        # cartwidth = 50.0
        # cartheight = 30.0
        #
        # if self.viewer is None:
        #     from gym.envs.classic_control import rendering
        #     self.viewer = rendering.Viewer(screen_width, screen_height)
        #     l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        #     axleoffset = cartheight / 4.0
        #     cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        #     self.carttrans = rendering.Transform()
        #     cart.add_attr(self.carttrans)
        #     self.viewer.add_geom(cart)
        #     l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        #     pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        #     pole.set_color(.8, .6, .4)
        #     self.poletrans = rendering.Transform(translation=(0, axleoffset))
        #     pole.add_attr(self.poletrans)
        #     pole.add_attr(self.carttrans)
        #     self.viewer.add_geom(pole)
        #     self.axle = rendering.make_circle(polewidth/2)
        #     self.axle.add_attr(self.poletrans)
        #     self.axle.add_attr(self.carttrans)
        #     self.axle.set_color(.5, .5, .8)
        #     self.viewer.add_geom(self.axle)
        #     self.track = rendering.Line((0, carty), (screen_width, carty))
        #     self.track.set_color(0, 0, 0)
        #     self.viewer.add_geom(self.track)
        #
        #     self._pole_geom = pole
        #
        # if self.state is None:
        #     return None
        #
        # # Edit the pole polygon vertex
        # pole = self._pole_geom
        # l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        # pole.v = [(l, b), (l, t), (r, t), (r, b)]
        #
        # x = self.state
        # cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        # self.carttrans.set_translation(cartx, carty)
        # self.poletrans.set_rotation(-x[2])
        #
        # return self.viewer.render(return_rgb_array=mode == 'rgb_array')
        return 0

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
