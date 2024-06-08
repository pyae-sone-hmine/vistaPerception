from typing import Dict, Any
from collections import deque
import numpy as np
import cvxopt

from vista.entities.agents.Car import Car
from vista.core.World import World
from vista.utils import transform
from vista.entities.agents.Dynamics import tireangle2curvature


def get_controller(config):
    return globals()[config['type']](config)


class BaseController:
    def __init__(self, config: Dict[str, Any], **kwargs):
        self._config = config

    def __call__(self, agent: Car, **kwargs):
        raise NotImplementedError

    @property
    def config(self) -> Dict[str, Any]:
        return self._config


class PurePursuit(BaseController):
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config)
        self._ts = deque(maxlen=2)

    def __call__(self, agent: Car):
        lookahead_dist = self.config['lookahead_dist']
        Kp = self.config['Kp']

        if len(self._ts) < 2:
            dt = 1 / 30.
        else:
            dt = self._ts[1] - self._ts[0]
            self._ts.append(agent.timestamp)

        speed = agent.human_speed

        road = agent.road
        ego_pose = agent.ego_dynamics.numpy()[:3]
        road_in_ego = np.array([ # TODO: vectorize this: slow if road buffer size too large
            transform.compute_relative_latlongyaw(_v, ego_pose)
            for _v in road
        ])

        dist = np.linalg.norm(road_in_ego[:,:2], axis=1)
        dist[road_in_ego[:,1] < 0] = 9999. # drop road in the back
        tgt_idx = np.argmin(np.abs(dist - lookahead_dist))
        dx, dy, dyaw = road_in_ego[tgt_idx]

        arc_len = speed * dt
        curvature = (Kp * np.arctan2(-dx, dy) * dt) / arc_len
        curvature_bound = [
            tireangle2curvature(_v, agent.wheel_base)
            for _v in agent.ego_dynamics.steering_bound]
        curvature = np.clip(curvature, *curvature_bound)

        return curvature, speed


class PID(BaseController):
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config)
        self._ts = deque(maxlen=2)

        self._lat_pid = simple_pid.PID(config['lateral']['Kp'],
                                       config['lateral']['Ki'],
                                       config['lateral']['Kd'],
                                       setpoint=0.)
        if 'longitudinal' in config.keys():
            self._long_pid = simple_pid.PID(config['longitudinal']['Kp'],
                                            config['longitudinal']['Ki'],
                                            config['longitudinal']['Kd'],
                                            setpoint=0.)

    def __call__(self, agent: Car):
        # Lateral PID (with lookahead)
        lookahead_dist = self.config['lateral']['lookahead_dist']

        if len(self._ts) < 2:
            dt = 1 / 30.
        else:
            dt = self._ts[1] - self._ts[0]
            self._ts.append(agent.timestamp)

        road = agent.road
        ego_pose = agent.ego_dynamics.numpy()[:3]
        road_in_ego = np.array([ # TODO: vectorize this: slow if road buffer size too large
            transform.compute_relative_latlongyaw(_v[:3], ego_pose)
            for _v in road
        ])

        dist = np.linalg.norm(road_in_ego[:,:2], axis=1)
        dist[road_in_ego[:,1] < 0] = 9999. # drop road in the back
        tgt_idx = np.argmin(np.abs(dist - lookahead_dist))
        lat_dx, lat_dy, lat_dyaw = road_in_ego[tgt_idx]

        heading_err_weight = self.config['lateral'].get('heading_err_weight', 0.0)
        heading_err_tol = self.config['lateral'].get('heading_err_tol', 0.0)
        heading_err = 0. if abs(agent.relative_state.yaw) <= heading_err_tol \
            else agent.relative_state.yaw
        error = lat_dx + heading_err_weight * heading_err
        curvature_bound = [
            tireangle2curvature(_v, agent.wheel_base)
            for _v in agent.ego_dynamics.steering_bound]
        self._lat_pid.output_limits = tuple(curvature_bound)
        self._lat_pid.sample_time = dt
        curvature = self._lat_pid(error)

        # Longtitudinal PID
        if 'longitudinal' in self.config.keys():
            self._long_pid.output_limits = agent.ego_dynamics.speed_bound
            long_dy = road_in_ego[1, 1]
            speed = self._long_pid(-long_dy)
        else:
            speed = agent.human_speed

        return curvature, speed



class NonLinearStateFeedback(BaseController):
    """ Controller used in "A sampling-based partial motion planning 
        framework for system-compliant navigation along a reference path" """
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config)
        self._ts = deque(maxlen=2)

    def __call__(self, agent: Car):
        if len(self._ts) < 2:
            dt = 1 / 30.
        else:
            dt = self._ts[1] - self._ts[0]
            self._ts.append(agent.timestamp)

        lookahead_dist = self.config['lookahead_dist']

        road = agent.road
        ego_pose = agent.ego_dynamics.numpy()[:3]
        road_in_ego = np.array([ # TODO: vectorize this: slow if road buffer size too large
            transform.compute_relative_latlongyaw(_v, ego_pose)
            for _v in road
        ])

        dist = np.linalg.norm(road_in_ego[:,:2], axis=1)
        dist[road_in_ego[:,1] < 0] = 9999. # drop road in the back
        tgt_idx = np.argmin(np.abs(dist - lookahead_dist))
        dx, dy, dyaw = road_in_ego[tgt_idx]

        delta_theta = dyaw
        delta_d = -dx # NOTE: need this probably due to some coordinate issue

        d_idx = int(agent._road.maxlen / 2. - road.shape[0] + tgt_idx)
        lookahead_ts = agent.trace.get_master_timestamp(agent.segment_index,
            agent.frame_index + d_idx)
        v_ref = agent.trace.f_speed(lookahead_ts)
        c_ref = agent.trace.f_curvature(lookahead_ts)

        w_f = v_ref * c_ref
        w = w_f \
            + self.config['K1'] * v_ref * np.sin(delta_theta) / (1e-8 + delta_theta) * delta_d \
            - self.config['K2'] * delta_theta

        speed = agent.human_speed
        curvature = w / speed

        curvature_bound = [
            tireangle2curvature(_v, agent.wheel_base)
            for _v in agent.ego_dynamics.steering_bound]
        curvature = np.clip(curvature, *curvature_bound)

        return curvature, speed


class CBF(BaseController):
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config)
    
    def __call__(self, agent: Car, world: World):
        ## fetch states
        dyn   = agent.ego_dynamics
        s     = dyn.s
        d     = dyn.d
        mu    = dyn.mu
        v     = dyn.speed
        delta = dyn.steering
        kappa = dyn.kappa
        
        if len(world.agents) > 1:
            obstacle = world.agents[1]
            assert obstacle.id != agent.id
            
            obs_dyn   = obstacle.ego_dynamics
            obs_s     = obs_dyn.s
            obs_d     = obs_dyn.d
            obs_mu    = obs_dyn.mu
            obs_v     = obs_dyn.speed
            obs_delta = obs_dyn.steering
            obs_kappa = obs_dyn.kappa
            
            has_obstacle = True
        else:
            has_obstacle = False
        
        ## obstacle avoidance cbf
        lrf, lr = 0.5, 2.78 # lr/(lr+lf)
        beta = np.arctan(lrf*np.tan(delta))
        cos_mu_beta = np.cos(mu + beta)
        sin_mu_beta = np.sin(mu + beta)
        mu_dot = v/lr*np.sin(beta) - kappa*v*cos_mu_beta/(1 - d*kappa)
        
        Q = np.eye(2)
        q = np.array([0., 0.]) # no reference control

        if has_obstacle:
            ds = s - obs_s
            obs_d = obs_d + np.sign(obs_d)*5
            dd = d - obs_d
            
            barrier = ds**2 + dd**2 - 7.9**2  # radius of the obstacle-covering disk is 7.9 < 8m (mpc), avoiding the set boundary
            barrier_dot = 2*ds*v*cos_mu_beta/(1 - d*kappa) + 2*dd*v*sin_mu_beta
            Lf2b = 2*(v*cos_mu_beta/(1 - d*kappa))**2 + 2*(v*sin_mu_beta)**2 - 2*ds*v*sin_mu_beta*mu_dot/(1 - d*kappa) + 2*ds*kappa*v**2*sin_mu_beta*cos_mu_beta/(1 - d*kappa)**2 + 2*dd*v*cos_mu_beta*mu_dot
            LgLfbu1 = 2*ds*cos_mu_beta/(1 - d*kappa) + 2*dd*sin_mu_beta
            LgLfbu2 = (-2*ds*v*sin_mu_beta/(1 - d*kappa) + 2*dd*v*cos_mu_beta)*lrf/np.cos(delta)**2/(1 + (lrf*np.tan(delta))**2)
            
            p = np.array(self.config.p_oa) # WHAT IS THIS FIXME
            G0 = np.array([-LgLfbu1, -LgLfbu2]).reshape(1, -1)
            h0 = Lf2b + (p[0] + p[1])*barrier_dot + p[0]*p[1]*barrier
            
            # print(barrier, barrier_dot+p[0]*barrier, v, d, obs_d, s, obs_s)
            
            # if (barrier <= 0) or (barrier_dot+p[0]*barrier <= 0):
            #     print("barrier", barrier, barrier_dot+p[0]*barrier) # DEV
        else:
            barrier = 0.
            G0 = np.array([0., 0.]).reshape(1, -1)
            h0 = 0.
        barrier_oa = barrier
        
        ## lane following cbf
        if self.config.use_lane_following_cbf: # PROBABLY SET THIS TO TRUE FIXME
            lf_cbf_threshold = self.config.lf_cbf_threshold # PLAY AROUND WITH THIS VALUE FIXME

            barrier = lf_cbf_threshold - d
            barrier_dot = -v*sin_mu_beta
            Lf2b = -v*cos_mu_beta*mu_dot
            LgLfbu1 = -sin_mu_beta
            LgLfbu2 = -v*cos_mu_beta*lrf/np.cos(delta)**2/(1 + (lrf*np.tan(delta))**2)
            p = np.array(self.config.p_lf)
            G1 = np.array([-LgLfbu1, -LgLfbu2]).reshape(1, -1)
            h1 = Lf2b + (p[0] + p[1])*barrier_dot + p[0]*p[1]*barrier

            barrier = d + lf_cbf_threshold
            barrier_dot = v*sin_mu_beta
            Lf2b = v*cos_mu_beta*mu_dot
            LgLfbu1 = sin_mu_beta
            LgLfbu2 = v*cos_mu_beta*lrf/np.cos(delta)**2/(1 + (lrf*np.tan(delta))**2)
            p = np.array(self.config.p_lf) # THIS ONE TOO FIXME
            G2 = np.array([-LgLfbu1, -LgLfbu2]).reshape(1, -1)
            h2 = Lf2b + (p[0] + p[1])*barrier_dot + p[0]*p[1]*barrier
            
            G = np.concatenate([G0,G1,G2], axis=0)
            h = np.array([h0,h1,h2])
        else:
            barrier = 0.
            G = G0
            h = np.array([h0])
        barrier_lf = barrier
            
        ## lane following CLF
        if self.config.use_lane_following_clf: # SET THIS TO NONE OR FALSE FIXME
            k1, k2, alpha, eps = self.config.lf_clf_params
            V_d = (d + k1*v*sin_mu_beta)**2 + alpha*(mu + k2*v/lr*np.sin(beta))**2
            LfV = 2*(d + k1*v*sin_mu_beta)*(v*sin_mu_beta + k1*v*cos_mu_beta*mu_dot) \
                    + 2*alpha*(mu + k2*v/lr*np.sin(beta))*mu_dot
            LgVu1 = 2*(d + k1*v*sin_mu_beta)*k1*sin_mu_beta \
                    + 2*alpha*(mu + k2*v/lr*np.sin(beta))*k2/lr*np.sin(beta)
            LgVu2 = (2*(d + k1*v*sin_mu_beta)*k1*v*cos_mu_beta \
                    + 2*alpha*(mu + k2*v/lr*np.sin(beta))*k2*v/lr*np.cos(beta))*lrf/np.cos(delta)**2/(1 + (lrf*np.tan(delta))**2)
            LgVud = -np.ones_like(LgVu2)
            
            G_clf = np.array([LgVu1, LgVu2, LgVud])
            h_clf = -LfV - eps*V_d
            
            # append slack variable
            G_aug = np.concatenate([G, np.zeros((G.shape[0],1))], axis=1)
            Q = np.eye(3)
            q = np.array([0., 0., 0.]) # no reference control
            
            G = np.concatenate([G_aug, G_clf[None,:]], axis=0)
            h = np.concatenate([h, np.array([h_clf])], axis=0)
    
        # print(barrier_oa, barrier_lf)
        
        # desired speed clf
        if self.config.use_desired_speed_clf:
            # eps = 10
            # vd = 8
            # LfV = 0
            eps, vd, LfV = self.config.ds_clf_params
            V = (v - vd)**2
            LgVu1 = 2*(v - vd)
            LgVu2 = np.zeros_like(LgVu1)
            LgVudv = -np.ones_like(LgVu1)
            
            G_clf_v = np.array([LgVu1,LgVu2,LgVu2, LgVudv])
            h_clf_v = -LfV - eps*V
            
            G_aug = np.concatenate([G, np.zeros((G.shape[0],1))], axis=1)
            Q = np.eye(4)
            Q[3] = 100*Q[3]
            q = np.array([0., 0., 0., 0.])
            
            G = np.concatenate([G_aug, G_clf_v[None, :]], axis = 0)
            h = np.concatenate([h, np.array([h_clf_v])], axis = 0)

        ## solve QP
        mat_Q = cvxopt.matrix(Q)
        mat_q = cvxopt.matrix(q)
        mat_G = cvxopt.matrix(G)
        mat_h = cvxopt.matrix(h)

        cvxopt.solvers.options['show_progress'] = False

        sol = cvxopt.solvers.qp(mat_Q, mat_q, mat_G, mat_h)
        tire_velocity = sol['x'][1]
        acceleration = sol['x'][0]
        
        return tire_velocity, acceleration