import numpy as np
import math as m
import copy
from typing import Tuple
class tool:
    def norm(input,min,max):
        return (input-min)/(max-min)
    def deg_fix(input):
        while(input>180 or input<=-180):
            if(input>180):
                input-=360
            elif(input<=-180):
                input+=360
        return input
    def todeg(input):
        return tool.deg_fix(np.rad2deg(input))
    def torad(input):
        return np.deg2rad(tool.deg_fix(input))
    def distence(list1, list2):
        sum_squares = 0
        for i in range(len(list1)):
            diff = list1[i] - list2[i]
            sum_squares += diff ** 2    
        return sum_squares ** 0.5
    
class Guidance:#比例导引
    def __init__(self,args) -> None:
        self.missile=None#所属的导弹
        self.g=args.g
        k1=args.k1
        k2=args.k2
    def sample(self,state):
        missile=self.missile;plane=self.missile.target
        dpitch=m.atan((plane.z-missile.z)/((plane.x-missile.x)**2+(plane.y-missile.y)**2)**0.5)
        nz=self.k1*((dpitch-missile.pitch)/m.pi/2)*missile.speed/self.g+m.cos(missile.pitch)
        dy=(plane.y-missile.y);dx=plane.x-missile.x
        dyaw=m.atan(abs(dy)/(abs(dx)+1e-8))
        if dx<0 and dy<0:
            dyaw+=-m.pi
        elif dx<0 and dy>0:
            dyaw=m.pi-dyaw
        elif dx>0 and dy<0:
            dyaw=-dyaw
        dyaw=-dyaw
        ddyaw=missile.yaw-dyaw
        if ddyaw > m.pi:
            ddyaw = ddyaw-2 * m.pi
        elif ddyaw < -m.pi:
            ddyaw = 2 * m.pi + ddyaw
        ny=self.k2*((ddyaw)/m.pi/2)*missile.speed/self.g
        return ny,nz,0,0
    def sample_old(self,state):#正常比例导引，但是不准可能存在错误，暂时弃用
        missile=self.missile;target=self.missile.target
        dx = missile.speed * m.cos(missile.pitch) * m.cos(missile.yaw)
        dy = - missile.speed * m.cos(missile.pitch) * m.sin(missile.yaw)
        dz = missile.speed * m.sin(missile.pitch)
        dx_t = target.speed * m.cos(target.pitch) * m.cos(target.yaw)
        dy_t = - target.speed * m.cos(target.pitch) * m.sin(target.yaw)
        dz_t = target.speed * m.sin(target.pitch)
        dist = missile.distence(target)
        dR = ((missile.z - target.z) * (dz - dz_t) + (missile.y - target.y) * (dy - dy_t) + 
              (missile.x - target.x) * (dx - dx_t)) / dist
        dtheta_L = ((dz_t - dz) * m.sqrt((target.x - missile.x) ** 2 + (target.y - missile.y) ** 2) - 
            (target.z - missile.z) * ((target.x - missile.x) * (dx_t - dx) + (target.y - missile.y) * (dy_t - dy)) / 
            m.sqrt((target.x - missile.x) ** 2 + (target.y - missile.y) ** 2)) / (dist**2)
        nz = self.k1 * abs(dR) * dtheta_L / self.g+np.cos(missile.pitch)
        dfea_L = ((dy_t - dy) * (target.x - missile.x) - (target.y - missile.y) * (dx_t - dx)) / (
                (target.x - missile.x) ** 2 + (target.y - missile.y) ** 2)
        ny = self.k2 * abs(dR) * dfea_L / self.g

        return ny,nz,0,0

class MissileActionSpace:
    def missile_get(mod=0):
        res=[]
        if mod==0:
            for i in range(-4,6):
                for j in range(-4,6):
                    res.append([i*10,j*10,0,0])
        elif mod==1:
            for i in range(30):
                t=0.1*i+3
                res.append([t,t,0,0])
            '''for i in range(1,5):
                for j in range(1,5):
                    res.append([i*2,j*2,0,0])'''
        return res
class Object:
    def __init__(self,no:int,position:Tuple[float, float, float],angle:Tuple[float, float, float],speed:float) -> None:
        self.no=no
        self.position=copy.deepcopy(position)#x,y,z
        self.angle=copy.deepcopy(angle)      #roll,pitch,yaw
        self.speed=speed
        self.target=None
        self.launched=False
        self.state=None
        self.next_state=None
        self.act_index=None
        self.act=None
        self.donw=False
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

class Missile(Object):
    def __init__(self,no:int,position:Tuple[float, float, float],angle:Tuple[float, float, float],speed:float,missile_mode:int) -> None:
        super().__init__(no,position,angle,speed)
        self.missile_train_mode = missile_mode
        self.big_reward=False    
    def pnorm(self,boundary:list):
        res=[]#boundary格式类似[[0,20000],[0,20000],[0,20000]]表示x,y,z的范围
        for i in range(3):
            res.append(tool.norm(self.position[i],boundary[i][0],boundary[i][1]))
        return np.array(res)
    def anorm(self):
        res=[]
        for i in range(3):
            res.append(tool.norm(self.angle[i],-180,180))
        return np.array(res)
    def apnorm(self,boundary:list):
        return np.append(self.pnorm(boundary),self.anorm())
    def vapnorm(self,boundary:list):
        return np.append(self.apnorm(boundary),np.array([tool.norm(self.speed,0,2000)]))
    def distence(self,agent):
        return tool.distence(self.position,agent.position)
    def hit_boundary(self,boundary):
        for i in range(3):
            if self.position[i]<=boundary[i][0] or self.position[i]>=boundary[i][1]:
                return True
        return False    
    def aplnorm(self):
        lanch_norm=np.array([1.0]) if self.launched else np.array([0.0])
        return np.append(lanch_norm,self.apnorm())
    def action_space(self):
        return MissileActionSpace.missile_get(self.missile_train_mode)
    def get_reward(self,boundary):
        reward = 0        
        r=boundary[0][1]-boundary[0][0]
        distence=self.distence(self.target)
        reward+=(r/2-distence)/(r/2)
        if self.done and (not self.big_reward):
            reward +=10
            self.big_reward=True
        return reward
    def common_guidance(self,plane,k1,k2,g):
        dpitch=m.atan((plane.z-self.z)/((plane.x-self.x)**2+(plane.y-self.y)**2)**0.5)
        nz=k1*((dpitch-self.pitch)/m.pi/2)*self.speed/g+m.cos(self.pitch)
        dy=(plane.y-self.y);dx=plane.x-self.x
        dyaw=m.atan(abs(dy)/(abs(dx)+1e-8))
        if dx<0 and dy<0:
            dyaw+=-m.pi
        elif dx<0 and dy>0:
            dyaw=m.pi-dyaw
        elif dx>0 and dy<0:
            dyaw=-dyaw
        dyaw=-dyaw
        ddyaw=self.yaw-dyaw
        if ddyaw > m.pi:
            ddyaw = ddyaw-2 * m.pi
        elif ddyaw < -m.pi:
            ddyaw = 2 * m.pi + ddyaw
        ny=k2*((ddyaw)/m.pi/2)*self.speed/g
        return ny,nz,0,0
    
    def step(self,k1,k2,roll,g,dt):#step
        ny,nz,_,_=self.common_guidance(self,self.target,k1,k2,g)
        self.pitch += g / self.speed * (nz - m.cos(self.pitch)) * dt
        self.yaw -= g / (self.speed * m.cos(self.pitch)) * ny * dt
        dx = self.speed * m.cos(self.pitch) * m.cos(self.yaw)
        dy = - self.speed * m.cos(self.pitch) * m.sin(self.yaw)
        dz = self.speed * m.sin(self.pitch) 
        self.x += dx*dt
        self.y += dy*dt
        self.z += dz*dt

    def step2(self,ny,nz,mlist:list,g,dt,drag_coefficient=0.001):#旧step
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
    def explode(self):#临时：触碰边界时变成未发射状态
        self.launched=False
    
if __name__=="__main__":
    p=Missile(1,[1,2,3],[30,2,3],100,0)
    print(p.angle)
    p.pitch+=m.pi/6*5
    print(p.position)
    print(p.angle)
    print(p.speed)
    print(len(MissileActionSpace.missile_get(0)))