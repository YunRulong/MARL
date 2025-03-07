import numpy as np
import math as m
import copy
import tool
from typing import List
from model import *

class custom_action_space:
    def missile_get(mod=0):
        res=[]
        if mod==0:
            for i in range(-6,7):
                for j in range(-6,7):
                    res.append([i*10,j*10,0,0])
        elif mod==1:
            for i in range(30):
                t=0.1*i+3
                res.append([t,t,0,0])
            '''for i in range(1,5):
                for j in range(1,5):
                    res.append([i*2,j*2,0,0])'''
        return res
    
class CustomAgent:
    def __init__(self,no:int,position:float[3],angle:float[3],speed:float,missile_mode:int) -> None:
        self.no=no
        self.position=copy.deepcopy(position)#x,y,z
        self.angle=copy.deepcopy(angle)      #roll,pitch,yaw
        self.speed=speed
        self.target=None
        self.launched=False
        self.missile_train_mode = missile_mode
        self.state=None
        self.next_state=None
        self.act_index=None
        self.act=None
        self.reward=0
    @property
    def x(self):
        return self.position[0]
    @property
    def y(self):
        return self.position[1]
    @property
    def z(self):
        return self.position[2]
    @x.setter
    def x(self,value):
        self.position[0]=value
    @y.setter
    def y(self,value):
        self.position[1]=value
    @z.setter
    def z(self,value):
        self.position[2]=value
    @property
    def roll(self):#存储角度，输出弧度
        return tool.torad(self.angle[0])    
    @property
    def pitch(self):
        return tool.torad(self.angle[1])
    @property
    def yaw(self):
        return tool.torad(self.angle[2])
    @roll.setter
    def roll(self,value):#输入弧度，存储角度
        self.angle[0]=tool.todeg(value)
    @pitch.setter
    def pitch(self,value):
        self.angle[1]=tool.todeg(value)
    @yaw.setter
    def yaw(self,value):
        self.angle[2]=tool.todeg(value)
       
    def __repr__(self) -> str:
        return str(self.__dict__)
    
    def pnorm(self,opt:list):
        res=[]
        for i in range(3):
            res.append(tool.norm(self.position[i],opt.boundary[i][0],opt.boundary[i][1]))
        return np.array(res)
    def anorm(self):
        res=[]
        for i in range(3):
            res.append(tool.norm(self.angle[i],-180,180))
        return np.array(res)
    def apnorm(self):
        return np.append(self.pnorm(),self.anorm())
    def vapnorm(self):
        return np.append(self.apnorm(),np.array([tool.norm(self.speed,0,2000)]))
    def action_space(self)->np.ndarray:
        raise ValueError("err->virtual fun!")
    def step(self):
        raise ValueError("err->virtual fun!")
    def distence(self,agent):
        return tool.distence(self.position,agent.position)
    def hit_boundary(self,boundary):
        for i in range(3):
            if self.position[i]<=boundary[i][0] or self.position[i]>=boundary[i][1]:
                return True
        return False
    def get_reward(self,boundary):
        reward=0
        
        r=boundary[0][1]-boundary[0][0]
        for i in range(3):
            min_gap=999999999
            for boun in boundary[i]:
                gap=abs(self.position[i]-boun)
                if gap<=min_gap:
                    min_gap=gap
            reward=(min_gap-r/2)/(r/2)
        return reward
    def aplnorm(self):
        lanch_norm=np.array([1.0]) if self.launched else np.array([0.0])
        return np.append(lanch_norm,self.apnorm())
    def action_space(self):
        return custom_action_space.missile_get(self.missile_train_mode)
    
    def step(self,ny,nz,roll,g,dt):
        ny,nz,_,_=tool.common_guidance(self,self.target,ny,nz,g)
        self.pitch += g / self.speed * (nz - m.cos(self.pitch)) * dt
        self.yaw -= g / (self.speed * m.cos(self.pitch)) * ny * dt
        dx = self.speed * m.cos(self.pitch) * m.cos(self.yaw)
        dy = - self.speed * m.cos(self.pitch) * m.sin(self.yaw)
        dz = self.speed * m.sin(self.pitch) 
        self.x += dx*dt
        self.y += dy*dt
        self.z += dz*dt
    def step2(self,ny,nz,mlist:list,g,dt,drag_coefficient=0.001):
        mx1 = self.x
        my1 = self.y
        mz1 = self.z
        vm = self.speed

        mx0, my0, mz0 = mlist

        speed_yaw0 = m.atan2(my1 - my0, mx1 - mx0)
        speed_pitch0 = m.atan2(m.sqrt((mx1 - mx0) * (mx1 - mx0) + (my1 - my0) * (my1 - my0)), mz1 - mz0)
        speed_yaw1 = ny * g * dt / vm + speed_yaw0
        speed_pitch1 = nz * g * dt / vm + speed_pitch0
        self.yaw = -speed_yaw1
        self.pitch = (m.pi / 2 - speed_pitch1)
        # 更新导弹速度
        vx = vm * m.cos(self.pitch) * m.cos(self.yaw)
        vy = vm * m.cos(self.pitch) * m.sin(-self.yaw)
        vz = vm * m.sin(self.pitch)
        vz -= g * dt

        self.speed = m.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        self.speed -= drag_coefficient * self.speed * dt

        mx2 = mx1 + dt * vm * m.sin(speed_pitch1) * m.cos(speed_yaw1)
        my2 = my1 + dt * vm * m.sin(speed_pitch1) * m.sin(speed_yaw1)
        mz2 = mz1 + dt * vm * m.cos(speed_pitch1)

        self.x = mx2
        self.y = my2
        self.z = mz2

        return [mx1, my1, mz1]
    def is_hit(self,range):
        return self.distence(self.target)<=range
    def cal_state(self):
        state=self.aplnorm()
        state=np.append(state,self.target.apnorm())
        self.state=state
    def get_reward(self,boundary):
        reward=super().get_reward(boundary)*2#0.5
        r=boundary[0][1]-boundary[0][0]
        distence=self.distence(self.target)
        reward+=(r/2-distence)/(r/2)*4#0.5
        '''if distence<5000:
            reward+=((5000-distence)/50000)
        if distence>5000:
            reward-=((distence-5000)/30000)
        if distence<1000:
            reward+=(1000-distence)/3000
        if distence<=Opt.explode_range:
            reward=3'''
        return reward
    def explode(self):#临时：触碰边界时变成未发射状态
        self.launched=False
    def update_state(self, state):
        self.state = state


    
if __name__=="__main__":
    p=CustomAgent([1,2,3],[30,2,3],100,0,[],None)

    print(m.sin(p.pitch))
    p.pitch+=m.pi/6*5
    print(p.angle[0])
    
    print(len(custom_action_space.missile_get(1)))
