"""Training script for State-of-the-art RL Agents on QEC Gym environments."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

# Add the parent directory to the path so we can import surface_code_in_stem
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surface_code_in_stem.rl_control.gym_env import QECGymEnv, QECContinuousControlEnv
from surface_code_in_stem.rl_control.sota_agents import PPOAgent, ContinuousSACAgent
from surface_code_in_stem.rl_control.replay_buffer import ReplayBuffer, Experience

def train_ppo_decoder(
    distance: int = 3,
    rounds: int = 3,
    physical_error_rate: float = 0.005,
    episodes: int = 1000,
    batch_size: int = 64
):
    print(f"--- Training Transformer-PPO Decoder ---")
    print(f"Distance: {distance}, Rounds: {rounds}, p: {physical_error_rate}")
    
    env = QECGymEnv(
        distance=distance, 
        rounds=rounds, 
        physical_error_rate=physical_error_rate,
        use_mwpm_baseline=True
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = len(env.action_space.nvec)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )
    
    # Storage for PPO trajectory
    states, actions, log_probs, rewards, values = [], [], [], [], []
    
    success_history = []
    mwpm_success_history = []
    history: List[dict[str, float]] = []
    
    for episode in range(episodes):
        state, info = env.reset()
        
        # Select action
        action, log_prob, value = agent.select_action(state)
        
        # Execute action
        next_state, reward, terminated, truncated, env_info = env.step(action)
        
        # Store transition
        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value)
        
        success_history.append(1 if env_info["is_correct"] else 0)
        mwpm_success_history.append(1 if info.get("mwpm_correct", False) else 0)
        history.append(
            {
                "episode": float(episode + 1),
                "reward": float(reward),
                "rl_success": float(success_history[-1]),
                "mwpm_success": float(mwpm_success_history[-1]),
            }
        )
        
        # PPO Update
        if (episode + 1) % batch_size == 0:
            # Compute advantages using simple TD(0) or GAE
            # Since episodes are length 1, return is just the reward
            returns_t = torch.FloatTensor(rewards)
            values_t = torch.FloatTensor(values)
            advantages_t = returns_t - values_t
            
            states_t = torch.FloatTensor(np.array(states))
            actions_t = torch.FloatTensor(np.array(actions))
            log_probs_t = torch.FloatTensor(log_probs)
            
            loss_dict = agent.update(states_t, actions_t, log_probs_t, returns_t, advantages_t)
            
            # Print stats
            recent_success = np.mean(success_history[-batch_size:])
            recent_mwpm = np.mean(mwpm_success_history[-batch_size:])
            
            print(f"Ep {episode+1}/{episodes} | RL Success: {recent_success:.3f} | MWPM Success: {recent_mwpm:.3f} | "
                  f"Policy Loss: {loss_dict['policy_loss']:.3f} | Value Loss: {loss_dict['value_loss']:.3f}")
            
            # Clear storage
            states, actions, log_probs, rewards, values = [], [], [], [], []
            
    print("Training complete.\n")
    return history


def train_sac_calibration(
    distance: int = 3,
    rounds: int = 3,
    base_error_rate: float = 0.005,
    episodes: int = 100,
    batch_size: int = 64,
    use_diffusion: bool = False
):
    print(f"--- Training Continuous SAC Calibration ---")
    print(f"Distance: {distance}, Rounds: {rounds}, p_base: {base_error_rate}")
    
    env = QECContinuousControlEnv(
        distance=distance,
        rounds=rounds,
        base_error_rate=base_error_rate,
        parameter_dim=4,
        batch_shots=256
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    agent = ContinuousSACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_space=env.action_space,
        use_diffusion=use_diffusion,
        device=device
    )
    
    replay_buffer = ReplayBuffer(capacity=10000, prioritized=False)
    history: List[dict[str, float]] = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Sample action (exploration)
            if len(replay_buffer) > batch_size:
                action = agent.select_action(state, evaluate=False)
            else:
                action = env.action_space.sample()
                
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Store in buffer
            exp = Experience(
                state=torch.FloatTensor(state),
                action=torch.FloatTensor(action),
                reward=float(reward),
                next_state=torch.FloatTensor(next_state),
                done=bool(done)
            )
            replay_buffer.push(exp)
            
            state = next_state
            
            # Update SAC
            if len(replay_buffer) > batch_size:
                # Sample batch
                s_batch, a_batch, r_batch, next_s_batch, d_batch, _, _ = replay_buffer.sample(batch_size)
                
                # Mask: 1 if not done, 0 if done
                mask_batch = 1.0 - d_batch.float()
                
                loss_dict = agent.update_parameters(
                    s_batch, a_batch, r_batch, next_s_batch, mask_batch
                )
                
        print(f"Ep {episode+1}/{episodes} | Total Reward (Negative Logical Error): {episode_reward:.4f} | "
              f"Final p_L: {info.get('logical_error_rate', 0):.4f}")
        history.append(
            {
                "episode": float(episode + 1),
                "reward": float(episode_reward),
                "logical_error_rate": float(info.get("logical_error_rate", 0.0)),
                "effective_p": float(info.get("effective_p", 0.0)),
            }
        )
              
    print("Training complete.\n")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["ppo", "sac", "all"], default="all")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--use-diffusion", action="store_true", help="Use the flow-matching diffusion policy for SAC.")
    parser.add_argument("--output-dir", type=str, default="artifacts/rl_training", help="Directory for saved training histories.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode in ["ppo", "all"]:
        ppo_history = train_ppo_decoder(episodes=args.episodes)
        with (output_dir / "ppo_history.json").open("w", encoding="utf-8") as f:
            json.dump(ppo_history, f, indent=2)
        
    if args.mode in ["sac", "all"]:
        # Use fewer episodes for SAC since each episode is max_steps=50 shots=256
        sac_history = train_sac_calibration(episodes=min(50, args.episodes // 10), use_diffusion=args.use_diffusion)
        with (output_dir / "sac_history.json").open("w", encoding="utf-8") as f:
            json.dump(sac_history, f, indent=2)
