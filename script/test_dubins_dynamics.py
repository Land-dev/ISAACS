# --------------------------------------------------------
# Test script for Dubins dynamics
# --------------------------------------------------------

import numpy as np
import torch
from simulators.dynamics.dubins6D import Dubins6D
from simulators import DubinsPursuitEvasionEnv
from omegaconf import OmegaConf

def test_dubins_dynamics():
    """Test the Dubins dynamics implementation."""
    print("Testing Dubins6D dynamics...")
    
    # Create a simple config
    class Config:
        evader_velocity = 0.6
        pursuer_velocity = 0.6
        omega_e_max = 2.0
        omega_p_max = 2.0
        goalR = 0.25
        state_max = 2.0
        dt = 0.1
    
    cfg = Config()
    
    # Create action space
    action_space = {
        'ctrl': np.array([[-2.0, 2.0]]),  # evader turn rate
        'dstb': np.array([[-2.0, 2.0]])   # pursuer turn rate
    }
    
    # Create dynamics
    dynamics = Dubins6D(cfg, action_space)
    print(f"Dynamics created with state dimension: {dynamics.dim_x}")
    
    # Test state integration
    state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, np.pi])  # [x_e, y_e, theta_e, x_p, y_p, theta_p]
    control = np.array([0.3])  # evader turn rate
    disturbance = np.array([-0.3])  # pursuer turn rate
    
    print(f"Initial state: {state}")
    print(f"Control: {control}")
    print(f"Disturbance: {disturbance}")
    
    # Integrate forward
    state_next, ctrl_clip, dstb_clip = dynamics.integrate_forward(state, control, adversary=disturbance)
    print(f"Next state: {state_next}")
    print(f"Clipped control: {ctrl_clip}")
    print(f"Clipped disturbance: {dstb_clip}")
    
    # Test collision detection
    collision_dist = dynamics.get_collision_distance(state_next)
    is_collision = dynamics.is_collision(state_next)
    print(f"Collision distance: {collision_dist}")
    print(f"Is collision: {is_collision}")
    
    # Test bounds checking
    is_within_bounds = dynamics.is_within_bounds(state_next)
    print(f"Is within bounds: {is_within_bounds}")
    
    print("Dynamics test completed successfully!")

def test_dubins_environment():
    """Test the Dubins environment."""
    print("\nTesting DubinsPursuitEvasionEnv...")
    
    # Create configs
    class EnvConfig:
        seed = 0
        timeout = 300
        end_criterion = "failure"
        g_x_fail = 0.1
        obs_type = "perfect"
    
    class AgentConfig:
        agent_id = "ego"
        dyn = "Dubins6D"
        footprint = "Box"
        state_box_limit = [-2.0, 2.0, -2.0, 2.0]
        class action_range:
            ctrl = [[-2.0, 2.0]]
            dstb = [[-2.0, 2.0]]
        
        evader_velocity = 0.6
        pursuer_velocity = 0.6
        omega_e_max = 2.0
        omega_p_max = 2.0
        goalR = 0.25
        state_max = 2.0
        dt = 0.1
    
    class CostConfig:
        cost_type = "Lagrange"
        goalR = 0.25
        state_max = 2.0
        set_mode = "avoid"
        q1_collision = 10.0
        q2_bounds = 1.0
        w_control = 0.01
        w_disturbance = 0.01
    
    # Create environment
    env = DubinsPursuitEvasionEnv(EnvConfig(), AgentConfig(), CostConfig())
    print("Environment created successfully!")
    
    # Test reset
    obs = env.reset()
    print(f"Initial observation: {obs}")
    
    # Test step
    action = {
        'ctrl': np.array([0.5]),
        'dstb': np.array([-0.3])
    }
    
    obs_next, reward, done, info = env.step(action)
    print(f"Next observation: {obs_next}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Info: {info}")
    
    # Test constraints
    constraints = env.get_constraints(env.state, action, obs_next)
    print(f"Constraints: {constraints}")
    
    # Test cost
    cost = env.get_cost(env.state, action, obs_next, constraints)
    print(f"Cost: {cost}")
    
    print("Environment test completed successfully!")

if __name__ == "__main__":
    test_dubins_dynamics()
    test_dubins_environment()
    print("\nAll tests passed!") 