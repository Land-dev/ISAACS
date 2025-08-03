# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import sys
from types import SimpleNamespace
from shutil import copyfile
import numpy as np
import wandb
import argparse
from omegaconf import OmegaConf

from simulators import DubinsPursuitEvasionEnv, save_obj, PrintLogger
from agent.isaacs import ISAACS

def main(config_file):
    # Load config
    cfg = OmegaConf.load(config_file)
    cfg.train.device = cfg.solver.device
    
    # Setup output directory
    os.makedirs(cfg.solver.out_folder, exist_ok=True)
    copyfile(config_file, os.path.join(cfg.solver.out_folder, 'config.yaml'))
    log_path = os.path.join(cfg.solver.out_folder, 'log.txt')
    if os.path.exists(log_path):
        os.remove(log_path)
    sys.stdout = PrintLogger(log_path)
    sys.stderr = PrintLogger(log_path)

    if cfg.solver.use_wandb:
        wandb.init(
            entity='saslab', project=cfg.solver.project_name,
            name=cfg.solver.name
        )
        tmp_cfg = {
            'environment': OmegaConf.to_container(cfg.environment),
            'solver': OmegaConf.to_container(cfg.solver),
            'arch': OmegaConf.to_container(cfg.arch),
            'train': OmegaConf.to_container(cfg.train)
        }
        wandb.config.update(tmp_cfg)

    # Construct environment
    print("\n== Environment information ==")
    env = DubinsPursuitEvasionEnv(cfg.environment, cfg.agent, cfg.cost)
    env.step_keep_constraints = False
    env.report()

    # Construct solver
    print("\n== Solver information ==")
    solver = ISAACS(cfg.solver, cfg.train, cfg.arch, cfg.environment)
    policy = solver.policy
    env.agent.init_policy(
        policy_type="NNCS", cfg=SimpleNamespace(device=policy.device),
        actor=policy.ctrl.net
    )
    n_params_ctrl = sum(
        p.numel() for p in policy.ctrl.net.parameters() if p.requires_grad
    )
    print(f'\nTotal parameters in ctrl: {n_params_ctrl}')
    print(f"We want to use: {cfg.train.device}, and Agent uses: {policy.device}")
    print("Critic is using cuda: ", next(policy.critic.net.parameters()).is_cuda)

    # Training
    print("\n== Learning starts ==")
    train_record, train_progress, violation_record, episode_record, pq_top_k = (
        solver.learn(env)
    )
    
    # Save results
    train_dict = {
        'train_record': train_record,
        'train_progress': train_progress,
        'violation_record': violation_record,
        'episode_record': episode_record,
        'pq_top_k': list(pq_top_k.queue)
    }
    save_obj(train_dict, os.path.join(cfg.solver.out_folder, 'train'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf", "--config_file", help="config file path", type=str,
        default=os.path.join("config", "dubins_isaacs.yaml")
    )
    args = parser.parse_args()
    main(args.config_file) 