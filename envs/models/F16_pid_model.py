import os
import sys
import torch
from torchdiffeq import odeint_adjoint as odeint
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from F16_model import F16Model
from F16.F16_dynamics import F16Dynamics
from algorithms.pid.controller import Controller
from utils.utils import wrap_PI


class F16PidModel(F16Model):
    def __init__(self, config, n, device, random_seed):
        super().__init__(config, n, device, random_seed)
        self.controller = Controller(dt=self.dt, n=self.n, device=device)
        self.max_heading = getattr(self.config, 'max_heading', 0.5)
        self.min_heading = getattr(self.config, 'min_heading', -0.5)
        self.max_npos = getattr(self.config, 'max_npos', 5000)
        self.min_npos = getattr(self.config, 'min_npos', -5000)
        self.max_epos = getattr(self.config, 'max_epos', 5000)
        self.min_epos = getattr(self.config, 'min_epos', -5000)
        self.init_T = getattr(self.config, 'init_T', 2000)

    def reset(self, env):
        done = env.is_done.bool()
        bad_done = env.bad_done.bool()
        exceed_time_limit = env.exceed_time_limit.bool()
        reset = (done | bad_done) | exceed_time_limit
        reset = torch.reshape(reset, (env.num_envs, env.num_agents, 1))
        reset = torch.any(reset.squeeze(axis=-1), axis=-1)
        reset_env = torch.nonzero(reset).squeeze(axis=-1)
        agents = torch.arange(self.n, device=self.device)
        agents = agents.reshape(env.num_envs, env.num_agents)
        reset_agents = agents[reset_env, :]
        reset_agents = reset_agents.reshape(-1)
        self.s[reset_agents, :] = torch.zeros_like(self.s[reset_agents, :])  # state
        self.u[reset_agents, :] = torch.zeros_like(self.u[reset_agents, :])
        self.s[reset_agents, 0] = torch.rand_like(self.s[reset_agents, 0]) * (self.max_npos - self.min_npos) + self.min_npos
        self.s[reset_agents, 1] = torch.rand_like(self.s[reset_agents, 1]) * (self.max_epos - self.min_epos) + self.min_epos
        self.s[reset_agents, 2] = torch.rand_like(self.s[reset_agents, 2]) * (self.max_altitude - self.min_altitude) + self.min_altitude
        self.s[reset_agents, 5] = torch.rand_like(self.s[reset_agents, 5]) * (self.max_heading - self.min_heading) + self.min_heading
        self.s[reset_agents, 6] = torch.rand_like(self.s[reset_agents, 6]) * (self.max_vt - self.min_vt) + self.min_vt
        self.u[reset_agents, 0] = self.init_T
        self.recent_s[reset_agents] = self.s[reset_agents]
        self.recent_u[reset_agents] = self.u[reset_agents]
    
    def update(self, action, env):
        action = torch.clamp(action, -1, 1)
        self.controller.roll_dem = 0.9 * self.controller.roll_dem + 0.1 * action[:, 1].reshape(-1, 1) * 4 * torch.pi / 9
        self.controller.pitch_dem = 0.9 * self.controller.pitch_dem + 0.1 * action[:, 2].reshape(-1, 1) * torch.pi / 12
        # self.controller.roll_dem = wrap_PI(self.s[:, 3].reshape(-1, 1) + action[:, 1].reshape(-1, 1) * torch.pi / 18)
        # self.controller.pitch_dem = wrap_PI(self.s[:, 4].reshape(-1, 1) + action[:, 2].reshape(-1, 1) * torch.pi / 18)
        self.controller.yaw_dem = wrap_PI(self.s[:, 5].reshape(-1, 1) + action[:, 3].reshape(-1, 1) * torch.pi / 60)
        self.controller.stabilize(env)
        T = 0.9 * self.u[:, 0].reshape(-1, 1) + 0.1 * action[:, 0].reshape(-1, 1) * 0.225 * 76300 / 0.3048
        el = -self.controller.el
        ail = -self.controller.ail
        rud = -self.controller.rud
        lef = torch.zeros((self.n, 1), device=self.device)
        self.recent_u = self.u
        self.u = torch.hstack((T, el))
        self.u = torch.hstack((self.u, ail))
        self.u = torch.hstack((self.u, rud))
        self.u = torch.hstack((self.u, lef))
        self.recent_s = self.s
        self.s = odeint(self.dynamics,
                        torch.hstack((self.s, self.u)),
                        torch.tensor([0., self.dt], device=self.device),
                        method=self.solver)[1, :, :self.num_states]