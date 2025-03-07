import argparse
import json
"""
Here are the param for the training

"""
def parse_nested_list(s: str) -> list:
    """将字符串解析为嵌套列表"""
    try:
        return json.loads(s.replace(" ", ""))  # 移除空格避免格式错误
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError("Invalid boundary format! Must be a JSON array.")

def get_common_args():
    parser = argparse.ArgumentParser()
    '''
    # the environment setting default
    parser.add_argument('--difficulty', type=str, default='7', help='the difficulty of the game')
    parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')
    parser.add_argument('--map', type=str, default='5m_vs_6m', help='the map of the game')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default='', help='absolute path to save the replay')
    '''
    #new environment setting
    parser.add_argument('--replay_dir', type=str, default='./replay', help='absolute path to save the replay')
    parser.add_argument('--map', type=str, default='custom_design', help='the map of the game')
    parser.add_argument('--boundary',type=parse_nested_list, default=[[-10000,10000],[-10000,10000],[0,30000]],help='三维边界,格式为JSON数组,例如：[[x_min,x_max],[y_min,y_max],[z_min,z_max]]')
    parser.add_argument('--blue_start_position', type=parse_nested_list, default=[[-6000,0,10000],[-5000,0,10000],[-4000,0,10000],[-3000,0,10000]],help='蓝方起始位置格式为JSON数组,例如：[[x1,y1,z1],[x2,y2,z2]]')
    parser.add_argument('--red_start_position', type=parse_nested_list, default=[[8500,0,5000],[7500,0,6000],[6500,0,5000],[5500,0,6000],[4500,0,5000],[3500,0,6000]],help='红方起始位置格式为JSON数组,例如：[[x1,y1,z1],[x2,y2,z2]]')
    parser.add_argument('--red_high_value_pisotion', type=parse_nested_list, default=[[7000,0,0],[6000,0,0],[5000,0,0],[4000,0,0]],help='红方高价值目标格式为JSON数组,例如：[[x1,y1,z1],[x2,y2,z2]]')
    parser.add_argument('--blue_start_angle', type=parse_nested_list, default=[[0,0,0],[0,0,0],[0,0,0],[0,0,0]],help='蓝方起始角度roll,pitch,yaw格式为JSON数组,例如：[[roll1,pitch1,yaw1],[roll2,pitch2,yaw2]]')
    parser.add_argument('--red_start_angle', type=parse_nested_list, default=[[0,0,180],[0,0,180],[0,0,180],[0,0,180],[0,0,180],[0,0,180]],help='红方起始角度roll,pitch,yaw格式为JSON数组,例如：[[roll1,pitch1,yaw1],[roll2,pitch2,yaw2]]')
    parser.add_argument('--blue_missile_count', type=int, default=4,help='蓝方导弹数量')
    parser.add_argument('--red_missile_count', type=int, default=6,help='红方导弹数量')
    parser.add_argument('--position_range', type=float, nargs=3,default=[1000,1000,1000],help='位置随机范围')
    parser.add_argument('--angle_range', type=float, nargs=3,default=[0,0,0],help='角度随机范围')
    parser.add_argument('--blue_missile_mode', type=int, default=1,help='蓝方导弹模式比例制导')
    parser.add_argument('--red_missile_mode', type=int, default=0,help='红方导弹模式博弈模型')
    parser.add_argument('--g', type=float, default=9.8,help='环境重力加速度')
    parser.add_argument('--frame', type=int, default=10,help='每秒的帧数量')
    parser.add_argument('--custom_agent', type=bool, default=1,help='是否使用自定义的智能体默认使用')
    parser.add_argument('--episode_limit', type=int, default=200,help='最长时间,单位0.1s')
    #missile setting
    parser.add_argument('--missile_speed', type=float, default=2000,help='导弹初始速度')
    parser.add_argument('--explode_range', type=int, default=100,help='导弹爆炸范围')
    parser.add_argument('--k1', type=float, default=3.6, help='比例制导系数')
    parser.add_argument('--k2', type=float, default=3.6, help='比例制导系数')
    #new environment setting结束
    # The alternative algorithms are vdn, coma, central_v, qmix, qtran_base,
    # qtran_alt, reinforce, coma+commnet, central_v+commnet, reinforce+commnet，
    # coma+g2anet, central_v+g2anet, reinforce+g2anet, maven
    parser.add_argument('--alg', type=str, default='qmix', help='the algorithm to train the agent')
    parser.add_argument('--n_steps', type=int, default=2000000, help='total time steps')
    parser.add_argument('--n_episodes', type=int, default=1, help='the number of episodes before once training')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
    parser.add_argument('--evaluate_cycle', type=int, default=1000, help='how often to evaluate the model')
    parser.add_argument('--evaluate_epoch', type=int, default=32, help='number of the epoch to evaluate the agent')
    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--evaluate', type=bool, default=False, help='whether to evaluate the model')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use the GPU')
    args = parser.parse_args()
    return args


# arguments of coma
def get_coma_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of vnd、 qmix、 qtran
def get_mixer_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    args.lr = 5e-4

    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.05
    anneal_steps = 50000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the train steps in one epoch
    args.train_steps = 1

    # experience replay
    args.batch_size = 32
    args.buffer_size = int(5e4)

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # QTRAN lambda
    args.lambda_opt = 1
    args.lambda_nopt = 1

    # prevent gradient explosion
    args.grad_norm_clip = 10

    # MAVEN
    args.noise_dim = 16
    args.lambda_mi = 0.001
    args.lambda_ql = 1
    args.entropy_coefficient = 0.001

    return args


# arguments of central_v
def get_centralv_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of central_v
def get_reinforce_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # how often to save the model
    args.save_cycle = 5000

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of coma+commnet
def get_commnet_args(args):
    if args.map == '3m':
        args.k = 2
    else:
        args.k = 3
    return args


def get_g2anet_args(args):
    args.attention_dim = 32
    args.hard = True
    return args

