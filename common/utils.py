import inspect
import functools
import torch


def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def td_lambda_target(batch, max_episode_len, q_targets, args):
    # batch.shep = (episode_num, max_episode_len， n_agents，n_actions)
    # q_targets.shape = (episode_num, max_episode_len， n_agents)
    episode_num = batch['o'].shape[0]
    mask = (1 - batch["padded"].float()).repeat(1, 1, args.n_agents)
    terminated = (1 - batch["terminated"].float()).repeat(1, 1, args.n_agents)
    r = batch['r'].repeat((1, 1, args.n_agents))
    # --------------------------------------------------n_step_return---------------------------------------------------
    '''
    1. 每条经验都有若干个n_step_return，所以给一个最大的max_episode_len维度用来装n_step_return
    最后一维,第n个数代表 n+1 step。
    2. 因为batch中各个episode的长度不一样，所以需要用mask将多出的n-step return置为0，
    否则的话会影响后面的lambda return。第t条经验的lambda return是和它后面的所有n-step return有关的，
    如果没有置0，在计算td-error后再置0是来不及的
    3. terminated用来将超出当前episode长度的q_targets和r置为0
    '''
    n_step_return = torch.zeros((episode_num, max_episode_len, args.n_agents, max_episode_len))
    for transition_idx in range(max_episode_len - 1, -1, -1):
        # 最后计算1 step return
        n_step_return[:, transition_idx, :, 0] = (r[:, transition_idx] + args.gamma * q_targets[:, transition_idx] * terminated[:, transition_idx]) * mask[:, transition_idx]        # 经验transition_idx上的obs有max_episode_len - transition_idx个return, 分别计算每种step return
        # 同时要注意n step return对应的index为n-1
        for n in range(1, max_episode_len - transition_idx):
            # t时刻的n step return =r + gamma * (t + 1 时刻的 n-1 step return)
            # n=1除外, 1 step return =r + gamma * (t + 1 时刻的 Q)
            n_step_return[:, transition_idx, :, n] = (r[:, transition_idx] + args.gamma * n_step_return[:, transition_idx + 1, :, n - 1]) * mask[:, transition_idx]
    # --------------------------------------------------n_step_return---------------------------------------------------

    # --------------------------------------------------lambda return---------------------------------------------------
    '''
    lambda_return.shape = (episode_num, max_episode_len，n_agents)
    '''
    lambda_return = torch.zeros((episode_num, max_episode_len, args.n_agents))
    for transition_idx in range(max_episode_len):
        returns = torch.zeros((episode_num, args.n_agents))
        for n in range(1, max_episode_len - transition_idx):
            returns += pow(args.td_lambda, n - 1) * n_step_return[:, transition_idx, :, n - 1]
        lambda_return[:, transition_idx] = (1 - args.td_lambda) * returns + \
                                           pow(args.td_lambda, max_episode_len - transition_idx - 1) * \
                                           n_step_return[:, transition_idx, :, max_episode_len - transition_idx - 1]
    # --------------------------------------------------lambda return---------------------------------------------------
    return lambda_return

class Guidance:#比例导引
    learnable=False
    
    def __init__(self,g,k1,k2) -> None:
        self.missile=None#所属的导弹
        self.g=g
        self.k1=k1
        self.k2=k2
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