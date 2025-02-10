import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from reward_function_base import BaseRewardFunction
from utils.utils import wrap_PI, get_AO_TA_R, orientation_reward, range_reward
import pdb

class PursueReward(BaseRewardFunction):
    """
    Measure the difference between the current posture and the target posture
    """
    def __init__(self, config):
        super().__init__(config)
        self.target_dist = getattr(self.config, 'target_dist', 3)

    def get_reward(self, task, env):
        """
        Args:
            task: task instance
            env: environment instance

        Returns:
            (tensor): reward
        """
        # posture reward
        ego_agents = torch.arange(env.num_envs, device=env.device) * env.num_agents
        enm_agents = ego_agents + 1
        ego_pos = env.s[ego_agents, :3]
        enm_pos = env.s[enm_agents, :3]
        ego_vel = env.es[ego_agents, :3]
        enm_vel = env.es[enm_agents, :3]
        ego_AO, ego_TA, distance = get_AO_TA_R(ego_pos, enm_pos, ego_vel, enm_vel)
        ego_orientation_reward = orientation_reward(ego_AO, ego_TA)
        enm_orientation_reward = orientation_reward(torch.pi - ego_TA, torch.pi - ego_AO)
        ego_range_reward = range_reward(self.target_dist, distance * 0.3048 / 1000)
        enm_range_reward = ego_range_reward
        ego_reward = ego_orientation_reward * ego_range_reward
        enm_reward = enm_orientation_reward * enm_range_reward
        reward = torch.hstack((ego_reward.reshape(-1, 1), enm_reward.reshape(-1, 1)))
        reward = 0.01 * reward.reshape(-1)
        contain_inf = (True in torch.isinf(reward))
        if contain_inf:
            pdb.set_trace()
        return reward
