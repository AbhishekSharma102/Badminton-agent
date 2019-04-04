#!/usr/bin/env python3

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
#from PIL import Image
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from math import sqrt, log, atan, exp, sin, cos, tan, pi, ceil, floor
#import sys
#sys.path.append("../..") 
from badminton.court import RenderCourt
from gym import spaces
import random
from scipy.optimize import bisect

class BadmintonEnv(gym.Env):
    def __init__(self):
        #LOAD PYGAME WINDOW
        self.render_court = RenderCourt()
        pygame.init()
        display = (800,600)
        self.screen = pygame.display.set_mode(display, OPENGL)
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (display[0]/display[1]), 0.1, 10000.0)
        self.view_mat = self.render_court.IdentityMat44()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0, 0, -2000)
        glRotatef(-45, 1, 0, 0)
        glGetFloatv(GL_MODELVIEW_MATRIX, self.view_mat)
        glLoadIdentity()
        self.font = pygame.font.Font(None, 60)
        #parameters representing the game
        self.current_state = None
        #indicates whether a rally is over
        self.done = False
        #frame rate of the game
        self.frame_rate = 1/30
        # points scored by each player
        self.player_points={'p1':0, 'p2':0}
        #court rectangle corener point (Note: each pixel is assumed to be 1cm long)
        #everything is stored in meters
        #x: long axis y:short axis, rows and columns logic
        self.court_0 = {'x':0.1,'y':0.1}
        #court rectangle
        self.court_rect = {'x':13.4, 'y':6.1}
        #image frame with court marking
        self.base_frame = np.zeros((1400,720,3))
        for x in np.arange(0,1340):
            for y in np.arange(0,610):
                self.base_frame[int(self.court_0['x']*100)+x, int(self.court_0['y']*100)+y] = [0,255,0]
               
        self.base_frame[669:671,:,:] = [0, 0, 0]
        #image frame with court markings + current player positions + current shuttle position
        self.current_frame = np.zeros((1400,720,3))
        #indicate whose chance is it to play the shuttle
        self.chance = 'p1'
        #player current positions
        self.p_pos = {'p1':{'x':0.1, 'y':3.05}, 'p2':{'x':10.75, 'y':3.05}}
        #p_attr stores at what veocity and angle player wishes to play the shuttle
        self.p_attr = {'p1':{'v':0, 'alpha':0}, 'p2':{'v':0, 'alpha':0}}
        #player in the court are represented by cone
        #cone height in m 
        self.pcone_h = 3
        #cone radis in m
        self.pradius = 0.5
        """
        shuttle trajectory attributs
        vi: intial velocity of the shuttle
        thetai: initial launch angle of the shuttle
        (psi, cx, cy): indicate the line of direction of the shuttle, cx and cy are on the net, psi is along long axis
        (k0, z0): launch point of the shuttle relative to line of direction
        """
        self.shuttle_attr = {'vi':56, 'thetai':1.0472, 'cx':6.7, 
                             'cy':3.05, 'psi':1.5708, 'k0':-6.6, 'z0':0.5}
        #shuttle relative position line of direction
        self.shuttle_rel_pos = {'k':-6.6, 'z':0.5, 't':0}    
        self.shuttle_rel_pos_prev = {'k':-6.6, 'z':0.5, 't':-1}
        #avoid oscillations in trajectory
        #store intial value inside sin of y(t)
        
        #self.make_frame()
        #player speed ranges
        self.max_player_speed = 10
        self.min_player_speed = 0
        #taken from internet
        self.max_shuttle_speed = 50
        self.min_shuttle_speed = 0
        #angle range
        self.min_angle = -np.pi
        self.max_angle = np.pi
        self.min_action_space_attr = np.array([self.min_player_speed, self.min_angle, self.min_player_speed, self.min_angle, self.min_angle])
        self.max_action_space_attr = np.array([self.max_player_speed, self.max_angle, self.max_player_speed, self.max_angle, self.max_angle])
        self.action_space = spaces.Box(low=self.min_action_space_attr, high=self.max_action_space_attr)
        #assuming maximum height reached by shuttle as 20 meters
        #observation_space has first 3 indexes for player pos, shuttle pos, and then remaining are for shuttle pos at that particular time frame. (assuming no shot lasts for more than 5 secs)
        self.observation_space = spaces.Box(low=0, high=20, shape=(3*(3 + int(5/self.frame_rate)),), dtype=np.float32)
        self.shuttle_trajectory = None
        
        self.destination_point = [0, 0, 0]
            
    def new_rally(self):
        #players come back to center court 
        self.p_pos = {'p1':{'x':3.35, 'y':3.05}, 'p2':{'x':10.05, 'y':3.05}}
        #shuttle is launched by p1
        self.shuttle_rel_pos = {'k':-3.35, 'z':0.5, 't':0}
        self.shuttle_rel_pos_prev = {'k':-3.35, 'z':0.5, 't':-1}
        self.shuttle_trajectory = None
        self.shuttle_attr = {'vi':1, 'thetai':0.1, 'cx':6.7, 
                             'cy':3.05, 'psi':0, 'k0':-3.35, 'z0':0.5}
        self.chance = 'p1'
        #self.make_frame()
        
    def shuttle_trajectory_wrt_t(self, vi, thetai, t):
        #planar shuttle trajectory using quadratic air resistance model
        #d-t #z-t
        v_t = 6.8;
        g = 9.8;
        #print(vi, thetai)
        d = (pow(v_t,2)/g)*log((vi*cos(thetai)*g*t+pow(v_t,2))/pow(v_t,2));
        
        arg_init = (((v_t/(vi*cos(thetai)))*(exp(g*0/pow(v_t,2))-1))
                + atan(v_t/(vi*sin(thetai))))
        nearest_pi_multiple = ceil(arg_init/pi)*pi-0.009
        
        arg = (((v_t/(vi*cos(thetai)))*(exp(g*d/pow(v_t,2))-1))
                + atan(v_t/(vi*sin(thetai))))
        arg = min(arg, nearest_pi_multiple)
        
        z = (pow(v_t,2)/g)*log(abs(((sin(arg))/sin(atan(v_t/(vi*sin(thetai)))))));
        return d,z
    
    def shuttle_trajectory_wrt_d(self, vi, thetai, d):
        #planar shuttle trajectory using quadratic air resistance model
        #dz
        v_t = 6.8;
        g = 9.8;
        arg_init = (((v_t/(vi*cos(thetai)))*(exp(g*0/pow(v_t,2))-1))
                + atan(v_t/(vi*sin(thetai))))
        nearest_pi_multiple = ceil(arg_init/pi)*pi-0.0009
        
        arg = (((v_t/(vi*cos(thetai)))*(exp(g*d/pow(v_t,2))-1))
                + atan(v_t/(vi*sin(thetai))))
        arg = min(arg, nearest_pi_multiple)
        
        z = (pow(v_t,2)/g)*log(abs(((sin(arg))/sin(atan(v_t/(vi*sin(thetai)))))));
        return z
    
    def move_shuttle(self):
        #store shuttle previous position
        self.shuttle_rel_pos_prev = self.shuttle_rel_pos.copy()
        #increment current time using frame rate
        self.shuttle_rel_pos['t'] += self.frame_rate
        #find the distance(d) and height(z) traversed by the shuttle
        #print(self.shuttle_attr)
        vi, thetai, t = self.shuttle_attr['vi'], self.shuttle_attr['thetai'], self.shuttle_rel_pos['t']
        #print(vi, thetai, t)
        d, z = self.shuttle_trajectory_wrt_t(vi, thetai, t)
        #based on player chance compute the shuttle position on court 
        if(self.chance == 'p1'):
            self.shuttle_rel_pos['k'] = self.shuttle_attr['k0'] - d 
            self.shuttle_rel_pos['z'] = self.shuttle_attr['z0'] + z 
        else:
            self.shuttle_rel_pos['k'] = self.shuttle_attr['k0'] + d 
            self.shuttle_rel_pos['z'] = self.shuttle_attr['z0'] + z 

    def get_shuttle_pos(self):
        #find shittle absolute co-ordinated from conrdinates raltive to line of direction
        sx = (self.shuttle_rel_pos['k']*cos(self.shuttle_attr['psi']) + self.shuttle_attr['cx'])
        sy = (self.shuttle_rel_pos['k']*sin(self.shuttle_attr['psi']) + self.shuttle_attr['cy'])
        sz = self.shuttle_rel_pos['z']
        return sx,sy,sz
    
    def shuttle_within_range(self, player):
        #find if the shuttle is inside the respective player cone
        #find if the shuttle crossed the net
        sx, sy, sz = self.get_shuttle_pos()
        if(self.chance=='p1'):
            flag = (self.shuttle_rel_pos['k'] < 0)
        else:
            flag = (self.shuttle_rel_pos['k'] > 0)
            
        return (flag and ((pow((sx-self.p_pos[player]['x']),2)+pow((sy-self.p_pos[player]['y']),2)-
                pow(((self.pradius/self.pcone_h)*(self.pcone_h-sz)),2)) <= 0))
    
    def move_player(self, player, p_action):
        #compute player current position using r(t) = r(t-1) + v*t in x and y direction
        if(p_action['p_attr']['v']):
            if(player == 'p1'):
                self.p_pos[player]['x'] += p_action['p_attr']['v']*self.frame_rate*cos(p_action['p_attr']['alpha'])
                self.p_pos[player]['y'] += p_action['p_attr']['v']*self.frame_rate*sin(p_action['p_attr']['alpha'])
            else:
                self.p_pos[player]['x'] += -p_action['p_attr']['v']*self.frame_rate*cos(p_action['p_attr']['alpha'])
                self.p_pos[player]['y'] += -p_action['p_attr']['v']*self.frame_rate*sin(p_action['p_attr']['alpha'])
                
            #keep the player within bounds
            if(player=='p1'):
                self.p_pos[player]['x'] = max(0, self.p_pos[player]['x'])
                self.p_pos[player]['x'] = min((self.court_rect['x']/2), self.p_pos[player]['x'])
            else:
                self.p_pos[player]['x'] = max((self.court_rect['x']/2), self.p_pos[player]['x'])
                self.p_pos[player]['x'] = min(self.court_rect['x'], self.p_pos[player]['x'])
            self.p_pos[player]['y'] = max(0, self.p_pos[player]['y'])
            self.p_pos[player]['y'] = min(self.court_rect['y'], self.p_pos[player]['y'])
            return True
        return False
    
    def shuttle_hit(self, player, p_action):
        #change shuttle trajectory based on respective player hitting
        flag = 0
        #get the shuttle current position where the player will be making contact
        sx, sy, sz = self.get_shuttle_pos()
        if(player=='p1'):
            if(p_action['s_attr']['vi'] > 0):
                if(self.shuttle_within_range('p1')):
                    flag=1
                    z0 = sz
                    cy = sy + (6.7 - sx)*(tan(p_action['s_attr']['psi']))
                    k0 = -sqrt(pow((6.7 - sx),2) + pow((cy - sy),2))
                    self.shuttle_attr = {'vi':p_action['s_attr']['vi'], 'thetai':p_action['s_attr']['thetai'],
                                         'cx':6.7, 'cy':cy, 
                                         'psi':p_action['s_attr']['psi'], 'k0':k0, 'z0':z0}
                    
                    self.shuttle_rel_pos['k'] = k0
                    self.shuttle_rel_pos['z'] = z0
                    self.shuttle_rel_pos['t'] = 0
                    self.chance = 'p2' 
                    self.shuttle_trajectory = None
                
        else:
            if(p_action['s_attr']['vi'] > 0):
                if(self.shuttle_within_range('p2')):
                    flag=1
                    z0 = sz
                    cy = sy - (sx - 6.7)*(tan(p_action['s_attr']['psi']))
                    k0 = sqrt(pow((6.7 - sx),2) + pow((cy - sy),2))
                    self.shuttle_attr = {'vi':p_action['s_attr']['vi'], 'thetai':p_action['s_attr']['thetai'],
                                         'cx':6.7, 'cy':cy, 
                                         'psi':p_action['s_attr']['psi'], 'k0':k0, 'z0':z0}
                    
                    self.shuttle_rel_pos['k'] = k0
                    self.shuttle_rel_pos['z'] = z0
                    self.shuttle_rel_pos['t'] = 0
                    self.chance = 'p1'
                    self.shuttle_trajectory = None
        
    def outside_court_bounds(self):
        #check if the current shuttle position is outside court bounds
        sx, sy, sz = self.get_shuttle_pos()
        if(sx < 0 or sx > (self.court_rect['x'])):
            return True
        if(sy < 0 or sy > (self.court_rect['y'])):
            return True
        return False
    
    def score(self):
        
        #check if the shuttle has crossed the net
        if(self.shuttle_rel_pos['k']*self.shuttle_rel_pos_prev['k']<=0):
            d = abs(self.shuttle_attr['k0'])
            vi, thetai = self.shuttle_attr['vi'], self.shuttle_attr['thetai']
            z =  self.shuttle_trajectory_wrt_d(vi, thetai, d)
            if(z < 1.524):
                if(self.chance == 'p1'):
                    self.player_points['p1']+=1
                else:
                    self.player_points['p2']+=1
                self.new_rally()
                return
            
        #if the shuttle is on the ground
        #give point to player who played that shot if shuttle is within court bounds
        # and shuttle is in the opp.court
        #start a new rally
        sx, sy, sz = self.get_shuttle_pos()
        flag = self.outside_court_bounds()
        if(flag):
            if(self.chance == 'p1'):
                self.player_points['p1']+=1
            else:
                self.player_points['p2']+=1
            self.new_rally()
            return
        
        if(sz <= 0):
            if(self.chance == 'p1'):
                if(self.shuttle_rel_pos['k']>0):
                    self.player_points['p1']+=1
                else:
                    self.player_points['p2']+=1
            else:
                if(self.shuttle_rel_pos['k']<0):
                    self.player_points['p2']+=1
                else:
                    self.player_points['p1']+=1
            self.new_rally()
            
        
    def act(self, p1_action, p2_action):
        #take action for each player
        #player can either hit the shuttle or move
        if(self.chance == 'p1'):
            self.shuttle_hit(self.chance, p1_action)
            self.move_player('p1', p1_action)
            self.move_player('p2', p2_action)
        
        elif(self.chance == 'p2'):
            self.shuttle_hit(self.chance, p2_action)
            self.move_player('p2', p2_action) 
            self.move_player('p1', p1_action)
        
        else:
            pass

        #move shuttle based on action
        # either shuttle continues on previous trjectory
        # or changes trajectory
        self.move_shuttle()
        # allot point if shuttle on ground
        self.score()
        #self.make_frame()
    
        
    def make_frame(self):
        #add players and shuttle to the base frame 
        self.current_frame = self.base_frame.copy()
        '''
        p1
        '''
        x = int((self.p_pos['p1']['x'] + self.court_0['x'])*100)
        y = int((self.p_pos['p1']['y'] + self.court_0['y'])*100)
        self.current_frame[x-10:x+10, y-10:y+10] = [255,0,255]
        '''
        p2
        '''
        x = int((self.p_pos['p2']['x'] + self.court_0['x'])*100)
        y = int((self.p_pos['p2']['y'] + self.court_0['y'])*100)
        self.current_frame[x-10:x+10, y-10:y+10] = [255,0,255]
        
        '''
        shuttle
        '''
        sx, sy, sz = self.get_shuttle_pos()
        x = int((sx + self.court_0['x'])*100)
        y = int((sy + self.court_0['y'])*100)
        self.current_frame[x-5:x+5, y-5:y+5] = [100,0,100]
        
    def show_current_frame(self):
        #img = Image.fromarray(np.uint8(self.current_frame) , 'RGB')
        #img.show()
        pass
    
    '''def render(self, i):
        img = Image.fromarray(np.uint8(self.current_frame) , 'RGB')
        z = self.shuttle_rel_pos['z']
        #b = self.shuttle_within_range('p2')
        img.save(f'./Data/frames/{i}_{z}.jpg')
        #self.video.write(cv2.imread('./Data/current_frame.jpg'))
    '''
    def checkEpisodeOver(self):
        if self.player_points['p1'] >= 15 or self.player_points['p2'] >= 15:
            #print("Episode over")
            return True
        return False

    def getReward(self, points_before_action):
        return (self.player_points['p1'] - points_before_action['p1']) - (self.player_points['p2'] - points_before_action['p2'])
    
    def convertActionListToAction(self, action_list):
        action = {'p_attr': { 'v': action_list[0],
                          'alpha': action_list[1]
                        },
                  's_attr': { 'vi': action_list[2],
                          'thetai': action_list[3],
                          'psi': action_list[4]
                        }
                 }
        return action
    
    def closestInterceptablePoint(self):
        """
        returns x, y, z, t, points where it lands and time taken since it was hit.
        """
        stuttle_eqn = lambda t: 0.5 + self.shuttle_trajectory_wrt_t(self.shuttle_attr['vi'], 
                                                              self.shuttle_attr['thetai'], abs(t))[1]
        t_upper = 1
        while(stuttle_eqn(t_upper)>0):
            t_upper += 1
        t_before_landing = bisect(stuttle_eqn, 0, t_upper)
        frame_time = floor(t_before_landing/self.frame_rate)*self.frame_rate#+self.frame_rate
        
        while(stuttle_eqn(frame_time)<0 and frame_time>0):
            frame_time -= self.frame_rate
            
        d, z = self.shuttle_trajectory_wrt_t(self.shuttle_attr['vi'], self.shuttle_attr['thetai'], frame_time)
        
        #based on player chance compute the shuttle rel position on court 
        if(self.chance == 'p1'):
            k = self.shuttle_attr['k0'] - d 
            z = self.shuttle_attr['z0'] + z 
        else:
            k = self.shuttle_attr['k0'] + d 
            z = self.shuttle_attr['z0'] + z 
        #get shuttle abs position in xy
        x = (k*cos(self.shuttle_attr['psi']) + self.shuttle_attr['cx'])
        y = (k*sin(self.shuttle_attr['psi']) + self.shuttle_attr['cy'])
        return x,y,z,frame_time
    
    def getRandomVelocityTheta(self):
        velocity = random.randint(self.min_shuttle_speed+20, self.max_shuttle_speed-25)
        theta = random.uniform(np.pi/6, np.pi/4)
        return velocity, theta
    
    def getShuttleAttributesForP2(self):
        if self.shuttle_within_range('p2'):
            #return one of hard coded trajectories
            speed_of_return, thetai_val = self.getRandomVelocityTheta()
            psi = None
            if self.p_pos['p2']['y'] <= 203:
                psi = random.uniform(0, np.pi/6)
            elif self.p_pos['p2']['y'] >= 407:
                psi = random.uniform(-np.pi/6, 0)
            else:
                psi = random.uniform(-np.pi/6, np.pi/6)
            return {'vi': speed_of_return, 'thetai': thetai_val, 'psi': psi}
        else:
            return { 'vi': 0, 'thetai': 0, 'psi': 0 }
        
    def getShuttleAttributesForP1(self):
        if self.shuttle_within_range('p1'):
            #return one of hard coded trajectories
            speed_of_return, thetai_val = self.getRandomVelocityTheta()
            psi = None
            if self.p_pos['p1']['y'] <= 203:
                psi = random.uniform(0, np.pi/4)
            elif self.p_pos['p1']['y'] >= 407:
                psi = random.uniform(-np.pi/4, 0)
            else:
                psi = random.uniform(-np.pi/4, np.pi/4)
            return {'vi': speed_of_return, 'thetai': thetai_val, 'psi': psi}
        else:
            return { 'vi': 0, 'thetai': 0, 'psi': 0 }
    
    def getPlayerAttributesForP1(self):
        if self.chance == 'p2' or self.shuttle_within_range('p1'):
            return { 'v': 0, 'alpha': 0 }
        else:
            #get shortest path between curr pos and the landing point
            closest_x, closest_y, closest_z, closest_time = self.closestInterceptablePoint()
            self.destination_point = [closest_x, closest_y, closest_z]
            distance_to_reach = sqrt(pow(closest_x - self.p_pos['p1']['x'], 2) + pow(closest_y - self.p_pos['p1']['y'], 2) )
            player_attrs = {}
            try:
                player_attrs['v'] = min(distance_to_reach/(closest_time-self.shuttle_rel_pos['t']), self.max_player_speed)
            except:
                player_attrs['v'] = 0
            if(closest_x - self.p_pos['p1']['x'] > 0):
                player_attrs['alpha'] = atan((closest_y - self.p_pos['p1']['y']) / abs(closest_x - self.p_pos['p1']['x'])) #NEED TO THINK
            else:
                if(closest_y - self.p_pos['p1']['y'] > 0):
                    player_attrs['alpha'] = pi - atan((closest_y - self.p_pos['p1']['y']) / abs(closest_x - self.p_pos['p1']['x'])) 
                else:
                    player_attrs['alpha'] = - pi - atan((closest_y - self.p_pos['p1']['y']) / abs(closest_x - self.p_pos['p1']['x'])) 
            
            return player_attrs
    
    def getPlayerAttributesForP2(self):
        if self.chance == 'p1' or self.shuttle_within_range('p2'):
            return { 'v': 0, 'alpha': 0 }
        else:
            #get shortest path between curr pos and the landing point
            closest_x, closest_y, closest_z, closest_time = self.closestInterceptablePoint()
            self.destination_point = [closest_x, closest_y, closest_z]
            distance_to_reach = sqrt(pow(closest_x - self.p_pos['p2']['x'], 2) + pow(closest_y - self.p_pos['p2']['y'], 2) )
            player_attrs = {}
            try:
                player_attrs['v'] = min(distance_to_reach/(closest_time - self.shuttle_rel_pos['t']), self.max_player_speed)
            except:
                player_attrs['v'] = 0
            if(self.p_pos['p2']['x'] - closest_x > 0):
                player_attrs['alpha'] = atan((self.p_pos['p2']['y'] - closest_y) / abs(self.p_pos['p2']['x'] - closest_x)) #NEED TO THINK
            else:
                if(self.p_pos['p2']['y'] - closest_y > 0):
                    player_attrs['alpha'] = pi - atan((self.p_pos['p2']['y'] - closest_y) / abs(self.p_pos['p2']['x'] - closest_x)) 
                else:
                    player_attrs['alpha'] = - pi - atan((self.p_pos['p2']['y'] - closest_y) / abs(self.p_pos['p2']['x'] - closest_x)) 
            
            return player_attrs
    
    def getP1Action(self):
        action = {}
        action['p_attr'] = self.getPlayerAttributesForP1()
        action['s_attr'] = self.getShuttleAttributesForP1()
        return action
    
    def getP2Action(self):
        action = {}
        action['p_attr'] = self.getPlayerAttributesForP2()
        action['s_attr'] = self.getShuttleAttributesForP2()
        return action
    
    def getShuttleTrajectory(self):
        if self.shuttle_trajectory != None:
            #print("Old trajectory is being used")
            return
        #print("New trajectory is being computed")
        self.shuttle_trajectory = []
        for i in range(0, int(5/self.frame_rate)):
            d, z = self.shuttle_trajectory_wrt_t(self.shuttle_attr['vi'], self.shuttle_attr['thetai'], i*self.frame_rate)
            cx, cy, psi = self.shuttle_attr['cx'], self.shuttle_attr['cy'], self.shuttle_attr['psi']
            xval = cx - d*cos(psi)
            yval = cy - d*sin(psi)
            self.shuttle_trajectory.append([xval, yval, z])
    
    def setObservationState(self):
        obs = []
        obs.append([self.p_pos['p1']['x'], self.p_pos['p1']['y'], 0])
        obs.append([self.p_pos['p2']['x'], self.p_pos['p2']['y'], 0])
        obs.append(list(self.get_shuttle_pos()))
        self.getShuttleTrajectory()
        obs += (self.shuttle_trajectory)
        return np.ravel(np.array(obs))
        
    def drawScore(self, score):
        textsurface = self.myfont.render(score, False, (0, 0, 0))
        self.screen.blit(textsurface,(0,0))
      
    
    def step(self, action):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : int
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        points_before_action = self.player_points
        #Take action. Need to define action_state, as p1 action is action_state[action]
        #p1_action = self.convertActionListToAction(action)
        p1_action = self.getP1Action()
        p2_action = self.getP2Action()
        #print(self.chance)
        #print(p1_action)
        #print(p2_action)
        self.act(p1_action, p2_action)
        reward = self.getReward(points_before_action)
        
        obs = self.setObservationState()
        episode_over = self.checkEpisodeOver()
        #print(str(self.player_points['p1']) + ":" + str(self.player_points['p2']))
        return obs, reward, episode_over, {}
    
    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        Should be new game, no need for game point.
        Returns: observation (object): the initial observation of the
            space.
        """
        self.new_rally()
        self.player_points['p1'] = 0
        self.player_points['p2'] = 0
        self.shuttle_trajectory = None
        return self.setObservationState()
    
    def drawText(self, position, textString):     
        font = pygame.font.Font (None, 64)
        textSurface = font.render(textString, True, (255,255,255,255), (0,0,0,255))     
        textData = pygame.image.tostring(textSurface, "RGBA", True)     
        glRasterPos3d(*position)     
        glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)
    
    
    def render(self, mode='human'):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode is 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        text = str(self.player_points['p1']) + ":" + str(self.player_points['p2'])
        self.render_court.p1_coord = [self.p_pos['p1']['y']*100 - 305, self.p_pos['p1']['x']*100 - 670]
        self.render_court.p2_coord = [self.p_pos['p2']['y']*100 - 305, self.p_pos['p2']['x']*100 - 670]
        self.render_court.destination_coord = [self.destination_point[1]*100 - 305, self.destination_point[0]*100 - 670, self.destination_point[2]*100]
        sx, sy, sz = self.get_shuttle_pos()
        self.render_court.shuttle_coord = [sy*100 - 305, sx*100 - 670, sz*100]
        glPushMatrix()
        glLoadIdentity()
        glTranslatef(0, 0, 0)
        glRotatef(0, 0, 0, 0)
        glMultMatrixf(self.view_mat)
        glGetFloatv(GL_MODELVIEW_MATRIX, self.view_mat)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        self.render_court.boundary()
        self.render_court.net()
        self.render_court.shuttle()
        self.render_court.playerOne()
        self.render_court.playerTwo()
        self.render_court.printDestination()
        self.drawText((500, 300, 10), text)
        glPopMatrix()
        pygame.time.wait(100)
        pygame.display.flip()
        #pass
    
    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        logger.warn("Could not seed environment %s", self)
        return
    
    def close(self):
        pygame.quit()
