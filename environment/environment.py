from typing import List
from missile import *
import random
import numpy as np
class Env:
    #总初始化
    def __init__(self,args) -> None:
        self.boundary = args.boundary
        self.position_range = args.position_range#位置随即范围
        self.angle_range = args.angle_range#角度随即范围
        self.blue_start_position = args.blue_start_position#蓝方导弹位置
        self.blue_start_angle = args.blue_start_angle#蓝方导弹角度
        self.red_start_position = args.red_start_position#红方导弹位置
        self.red_start_angle = args.red_start_anglee#红方导弹角度        
        self.red_high_value_pisotion = args.red_high_value_pisotion#红方高价值目标
        self.blue_missile_count = args.blue_missile_count#蓝方导弹数量
        self.red_missile_count = args.red_missile_count#红方导弹数量        

        self.time_unit=1/args.frame
        self.g=args.g
        self.k1=args.k1
        self.k2=args.k2
        self.explode_range=args.explode_range
        self.boundary=args.boundary
        self.reset()
    #每局开始的初始化
    def reset(self):
        self.time_step=0 
        self.done = False
        self.high_value_count=len(self.red_high_value_pisotion)
        self.red_missile_live_count=self.red_missile_count
        self.blue_missile_live_count=self.blue_missile_list
        self.blue_missile_list:Missile = []#蓝方导弹列表
        self.red_missile_list:Missile = []#红方导弹列表
        self.red_high_value_list:Object=[]#高价值目标列表
        self.fake_target=Object(0,[self.boundary[0][1],self.boundary[1][1],self.boundary[2][1]],[0,0,0])#没有目标后填充目标
        #蓝方导弹初始化
        for i in range(self.blue_missile_count):            
            position=[]
            angle=[]
            for j in range(3):
                position.append(random.uniform(self.blue_start_position[i][j]-self.position_range[j],self.blue_start_position[i][j]+self.position_range[j]))
                angle.append(random.uniform(self.blue_start_angle[i][j]-self.angle_range[j],self.blue_start_angle[i][j]+self.angle_range[j]))
            self.blue_missile_list.append(Missile(i,position,angle,2000,0))
        #红方导弹初始化              
        for i in range(self.red_missile_count):            
            position=[]
            angle=[]
            for j in range(3):
                position.append(random.uniform(self.red_start_position[i][j]-self.position_range[j],self.red_start_position[i][j]+self.position_range[j]))
                angle.append(random.uniform(self.red_start_angle[i][j]-self.angle_range[j],self.red_start_angle[i][j]+self.angle_range[j]))     
            self.red_missile_list.append(Missile(i,position,angle,2000,0))
        #红方高价值目标
        for i in range(len(self.red_high_value_pisotion)):
            position=[]
            angle=[0,0,0]
            for j in range(3):
                position.append(random.uniform(self.red_high_value_pisotion[i][j]-self.position_range[j],self.red_high_value_pisotion[i][j]+self.position_range[j]))
            self.red_high_value_list.append(Object(i,position,angle,0))
        #初始化发射状态
        for i,missile in enumerate(self.blue_missile_list):
            missile.target = self.red_high_value_list[i%len(self.red_high_value_pisotion)]
            missile.launched = True
        for i,missile in enumerate(self.red_missile_list):
            missile.target = self.blue_missile_list[i%len(self.blue_missile_count)]
            missile.launched = True
        self.done=False
        self.red_data=[[] for _ in range(self.red_missile_count)]
        self.blue_data=[[] for _ in range(self.blue_missile_count)]    
        #返回观测状态
        obs = self.get_obs()
        state = self.get_state()
        return obs, state   
    #返回全局状态
    '''
    state一维向量:[红方0状态,红方1状态,...,红方n状态,蓝方1状态,...,蓝方n状态,存活标识]
    状态格式:阵营，自身阵营的独热编码,x,y,z,roll,pitch,yaw,speed,live标致
    存活标识:红1存活,红2存活,...,红n存活,蓝1存活,...,蓝n存活
    '''  
    def get_state(self):
        # 为盟友添加独热编码 ID
        missiles_with_id = []
        live_mask = [0]*(self.blue_missile_count+self.red_missile_count)
        for idx, missile in enumerate(self.red_missile_list):
            faction_id = [0]
            one_hot_id = [0] * self.red_missile_count            
            one_hot_id[idx] = 1
            if missile.done < 1:
                live_mask[idx] = 1
                missiles_with_id.extend([faction_id+one_hot_id + [missile.position+missile.angle+missile.speed+(1-missile.done)]])
            else:
                live_mask[idx] = 0
                missiles_with_id.extend([faction_id+one_hot_id + [missile.position+missile.angle+missile.speed+(1-missile.done)]])
        for idx, missile in enumerate(self.blue_missile_list):
            faction_id = [1]
            one_hot_id = [0] * self.blue_missile_count
            one_hot_id[idx] = 1
            if missile.done < 1:
                live_mask[self.red_missile_count+idx] = 1
                missiles_with_id.extend([faction_id+one_hot_id + [missile.position+missile.angle+missile.speed+(1-missile.done)]])
            else:
                live_mask[self.red_missile_count+idx] = 0
                missiles_with_id.extend([faction_id+one_hot_id + [missile.position+missile.angle+missile.speed+(1-missile.done)]])
        # 拼接状态
        state=[item for missile in missiles_with_id for item in missile]+live_mask
        return state.astype(np.float32)
    
    '''  
    总obs:  [
            [导弹1obs],
            [导弹2obs],
            [导弹3obs]
            ]
    导弹obs:[自己的数据+队友数据+敌方数据]

    自己的obs:[x,y,z,roll,pitch,yaw,speed]全部归一化
    队友的obs:[相对x,相对y,相对z,roll,pitch,yaw,speed]全部归一化
    敌人的obs:[相对x,相对y,相对z,roll,pitch,yaw,speed]全部归一化
    '''
    #返回所有智能体的局部观测
    def get_obs(self):
        state = []
        for i in range(self.red_missile_count):
            state.append([self.get_obs_self(i)+self.get_obs_ally(i)+self.get_obs_enemy()])
        return state
    #返回自身观测(x,y,z,roll,pitch,yaw,speed)
    def get_obs_self(self,i:int):
        obs_self = [self.red_missile_list[i].position+self.red_missile_list[i].angle+self.red_missile_list[i].speed]
        #obs_self = self.blue_missile_list[i].vapnorm(self.boundary)
        return obs_self 
    #返回对队友的观测
    def get_obs_ally(self, i: int):
        obs_ally = []
        for j, ally in enumerate(self.red_missile_list):
            if j != i:  # 排除当前导弹
                # 将 position 和 angle 拼接成一个7元组
                ally_info = np.concatenate([ally.position+ally.angle+ally.speed])
                # ally_info = np.concatenate(ally.vapnorm(self.boundary))
                obs_ally.append(ally_info)        
        # 将所有队友的7元组拼接成一个一维向量
        if obs_ally:  # 确保有队友信息
            obs_ally = np.hstack(obs_ally)
        else:
            obs_ally = np.array([])  # 如果没有队友，返回空数组        
        return obs_ally
    #返回对敌人的观测
    def get_obs_enemy(self):
        obs_enemy = []
        for enemy in self.blue_missile_list:
            # 将 position 和 angle 拼接成一个六元组
            enemy_info = np.concatenate([enemy.position+enemy.angle+enemy.speed])
            #enemy_info = np.concatenate(enemy.vapnorm(self.boundary))
            obs_enemy.append(enemy_info)        
        # 将所有导弹的六元组拼接成一个一维向量
        obs_enemy = np.hstack(obs_enemy)
        return obs_enemy
    #得到可用动作
    def get_avail_agent_actions(self,agent_id:int):
        agent = self.red_missile_list[agent_id]
        avail_action=[1]*(self.blue_missile_count+1)
        for i,missile in enumerate(self.blue_missile_list):
            if missile.done:
                avail_action[i]=0
        return avail_action
    #返回环境信息    
    def get_env_info(self):  
        '''
        get_env_info返回env_info包含        
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }
        env_info["agent_features"] = self.ally_state_attr_names
        env_info["enemy_features"] = self.enemy_state_attr_names
        '''      
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }        
        env_info["agent_features"] = self.ally_state_attr_names
        env_info["enemy_features"] = self.enemy_state_attr_names
        return env_info
    def implement_policy(self,missile:Missile,action:int):
        if action < self.blue_missile_count:
            missile.target=self.blue_missile_count[action]
        else:
            missile.target=self.fake_target
    #计算agent.reward, env.done
    def step(self,actions):        
        moving_missile_list=[]
        step_reward=0
        info = {"battle_won": False}
        #执行策略  
        for i, missile in enumerate(self.red_missile_list):
            self.implement_policy(missile,actions[i])
        #导弹运动迭代
        for missile in self.red_missile_list+self.blue_missile_list:
            if missile.done < 1:
                moving_missile_list.append(missile)            
        for missile in moving_missile_list:
            missile.step(self.k1,self.k2,0,self.g,self.time_unit)        
        #计算导弹奖励
        for missile in self.red_missile_list:
            missile.reward=missile.get_reward(self.boundary)
            step_reward+=missile.reward
        #存储数据（未完成）
        time=self.time_step*self.time_unit
        current_blue_missile_info = []
        for bm in self.blue_missile_list:
            current_blue_missile_info.append(bm.position+bm.angle)
        self.data[0].append(current_blue_missile_info)
        current_red_missile_info = []
        for rm in self.red_missile_list:
            current_red_missile_info.append(rm.position+rm.angle)
        self.data[1].append(current_red_missile_info) 
        self.time_step+=1       
        #判断爆炸
        for missile in self.red_missile_list:
            if missile.is_hit(self.explode_range):
                missile.done=True
                missile.target.done=True
        for missile in self.blue_missile_list:
            if missile.is_hit(self.explode_range):
                missile.done=True
                missile.target.done=True
        #判断撞墙
        for missile in moving_missile_list:
            if missile.hit_boundary(self.boundary):
                missile.done=True
                missile.big_reward=True
        #判断结束        
        if self.high_value_count == 0 or self.red_missile_live_count==0 or self.blue_missile_live_count==0:
            self.done=True        
        #判断胜利
        if self.done:
            count = 0        
            for missile in self.blue_missile_list:
                if missile.done:
                    count +=1
            if count/len(self.blue_missile_count) >= 0.6:
                info["battle_won"] = True
        return step_reward, self.done, info
 
import argparse    
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.boundary = [[-10000,10000],[-10000,10000],[0,30000]]
    args.blue_missile_position = [[-5000,0,10000],[-4000,0,10000]]#蓝方导弹位置
    args.red_missile_position = [[5000,0,10000],[4000,0,10000]]#红方导弹位置
    args.red_high_value_pisotion = [[5000,0,0],[4000,0,0]]#红方高价值目标
    args.frame = 10
    args.g = 9.8
    args.explode_range = 10
    env=Env(args)
    print(env.__dict__)
    
    