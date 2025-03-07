from environment.environment import *
from model import *
# 启用环境进行训练，env.done==True或超过单次最大步数则结束该次训练，返回奖励值
def train_one_episode(opt, net_list,rpm_list):
    env=Env(opt,net_list,rpm_list)
    env.cal_state(next=False)#将计算的state存入state
    step = 0
    all_agent_list=env.get_all_agent_list()
    total_reward_list=[0]*len(all_agent_list)
    loss_list=[[] for _ in range(len(all_agent_list))]
    

    while not env.done and step<opt.max_step:#已有state->act->next_state,reward,done
        agent_list=env.get_agent_list()#带神经网络的对象
        obj_list=env.get_obj_list()#战场运动的对象
        step += 1
        for obj in obj_list:
            obj.act=obj.sample()
        env.step(step)#reward,done
        env.cal_state()#next_state
        for agent in agent_list:
            agent.rpm.add((agent.state, agent.act_index, agent.reward, agent.next_state, env.done))
        for i,agent in enumerate(all_agent_list):
            loss=1e3 if len(loss_list[i])==0 else loss_list[i][-1]
            if agent in agent_list:#只有在运动的学习，不在的取上一个loss和reward
                if (agent.rpm.size() > opt.memory_warmup_size) and (step % opt.learn_freq == 0):
                    experiences = agent.rpm.sample(opt.batch_size)
                    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*experiences)# s,a,r,s',done
                    loss = agent.net.learn(batch_state, batch_action, batch_reward, batch_next_state, batch_done)# 智能体更新价值网络
            total_reward_list[i] += agent.reward
            loss_list[i].append(loss)
        for agent in agent_list:
            agent.state = agent.next_state
    loss_list = [np.mean(sublist) for sublist in loss_list]
    total_reward_list=[reward/step for reward in  total_reward_list]
    net_list=env.get_net_list()
    rpm_list=env.get_rpm_list()
    
    return env,net_list,rpm_list,total_reward_list,loss_list,step


def train(opt):
    plane_net_list=[[],[]]
    missile_net_list=[[],[]]
    for i in range(2):
        if opt.plane_agent:
            plane_net_list[i].append(DQNModel(opt.plane_state_size,opt.plane_action_size,opt.gamma,opt.learning_rate))
        else:
            plane_net=DQNModel(opt.plane_state_size,opt.plane_action_size,opt.gamma,opt.learning_rate)
            plane_net.load('40000/plane'+str(i)+'0.pth')
            plane_net_list[i].append(plane_net)
        if opt.missile_agent:
            missile_net=DQNModel(opt.missile_state_size,opt.missile_action_size,opt.gamma,opt.learning_rate)
            missile_net_list[i].append(missile_net)
        else:
            missile_net_list[i].append(Guidance(opt.g))
            
    net_list=[plane_net_list,missile_net_list];rpm_list=None

    loss_all=[]
    reward_all=[]
    for episode in range(1,opt.max_episode+1):
        env,net_list,rpm_list,reward_list,loss_list,step=train_one_episode(opt,net_list,rpm_list)
        reward_all.append(reward_list);loss_all.append(loss_list)
        reward_list=["{:.4f}".format(item) for item in reward_list]
        loss_list=["{:.4f}".format(item) for item in loss_list]
        info='Episode:'+str(episode)+'|total_step:'+str(step)
        info+='|reward:'+str(reward_list)+'|loss:'+str(loss_list)
        print(info)
        if episode%opt.save_freq==0 :#and step>50
            env.write_csv()
            tool.write_acmi('demo'+str(episode).zfill(5),opt.save_dir+'/csv',env.time_unit)
            env.save(episode)
    tool.draw(reward_all)
if __name__=="__main__":
    #opt=args
    #train(opt)
    pass