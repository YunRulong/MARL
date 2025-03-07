import os,csv,glob,shutil
import numpy as np
import matplotlib.pyplot as plt
import math as m
import main
def norm(input,min,max):
    return (input-min)/(max-min)
def demorm(input,min,max):
    return input*(max-min)+min

def deg_fix(input):
    while(input>180 or input<=-180):
        if(input>180):
            input-=360
        elif(input<=-180):
            input+=360
    return input

def todeg(input):
    return deg_fix(np.rad2deg(input))
def torad(input):
    return np.deg2rad(deg_fix(input))

def distence(list1, list2):
    sum_squares = 0
    for i in range(len(list1)):
        diff = list1[i] - list2[i]
        sum_squares += diff ** 2
    
    return sum_squares ** 0.5

def insert_in_dict(dict,k,v):
    if k not in dict:
        dict[k]=[v]
    else:
        dict[k].append(v)

def read_csv(file_path):
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        res=[]
        head_flag=True
        for row in csv_reader:
            if head_flag:
                head_flag=False
                continue
            line=[]
            for item in row:
                line.append(float(item))
            res.append(line) 
    return res
def write_csv(save_dir,data):
    fname=save_dir
    folder_path = os.path.dirname(fname)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    data.insert(0,['x','y','z','roll','pitch','yaw'])
    with open(fname, 'w',newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def write_acmi(target_name,source_dir,time_unit,save_dir,explode_time=10):#爆炸持续10单位时间
    target_name=save_dir+"/acmi/"+target_name+'.acmi'
    folder_path = os.path.dirname(target_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    data_dict={}
    csv_files = glob.glob(os.path.join(source_dir, '*.csv'))
    plane_counts=[0,0]
    missile_counts=[0,0]
    content="FileType=text/acmi/tacview\nFileVersion=2.1\n0,ReferenceTime=2023-12-09T00:00:00Z\n"
    for file_name in csv_files:
        fname = os.path.basename(file_name)
        fname, _ = os.path.splitext(fname)
        type,belong,start_time= fname.split('_')
        belong=int(belong);start_time=float(start_time)
        colors=['Blue','Red']
        if type=='plane':
            obj_name='F16'
            plane_counts[belong]+=1
            obj_no='a'+str(np.sum(plane_counts))
            color=colors[belong]
        else: 
            obj_name='AIM-9L'
            missile_counts[belong]+=1
            obj_no='b'+str(np.sum(missile_counts))
            color=colors[belong]
        apdata=read_csv(file_name)
        for i in range(len(apdata)):
            apdata[i][0]/=111319.5;apdata[i][1]/=111319.5;apdata[i][5]+=90
        for i,line in enumerate(apdata):
            line_string=obj_no+','+'T='+'|'.join(str(elem) for elem in line)+',Name='+obj_name+',Color='+color+'\n'
            time=str(start_time+i*time_unit)
            insert_in_dict(data_dict,time,line_string)
        end_time=start_time+len(apdata)*time_unit
        insert_in_dict(data_dict,str(end_time),'-'+obj_no+'\n')
        if type=='plane':
            line_string=obj_no+'F,'+'T='+'|'.join(str(elem) for elem in apdata[-1])+',Type=Misc+Explosion'+',Color='+color[belong]+',Radius=300\n'
            for i in range(explode_time):
                time=str(end_time+i*time_unit)
                insert_in_dict(data_dict,time,line_string)
    for t,line_strings in data_dict.items():
        content+='#'+t+'\n'
        for line_string in line_strings:
            content+=line_string
    with open(target_name, 'w') as file:
        file.write(content)
def ini_info(opt):
    with open(opt.save_dir+'/ini_info.txt', 'w', newline='', encoding='utf-8') as file:
        file.write(str(opt.__dict__))
def sample_list(lst, num_samples): 
    res=[] 
    if len(lst) < num_samples:  
        step=1 
    else:
        step = len(lst) / num_samples  # 计算步长，可能不是整数  
    for i in range(num_samples):
        idx=int(round(i * step))
        if idx<len(lst):
            res.append(lst[idx])
    return res
def common_guidance(missile,plane,k1,k2,g):
    dpitch=m.atan((plane.z-missile.z)/((plane.x-missile.x)**2+(plane.y-missile.y)**2)**0.5)
    nz=k1*((dpitch-missile.pitch)/m.pi/2)*missile.speed/g+m.cos(missile.pitch)
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
    ny=k2*((ddyaw)/m.pi/2)*missile.speed/g
    return ny,nz,0,0

class Guidance:#比例导引
    learnable=False
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
        #ny=dyaw
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
