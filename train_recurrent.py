
import os
import time
import multiprocessing

import numpy as np
import torch
import gymnasium as gym

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from tqdm import tqdm

from sb3_contrib import RecurrentPPO
from domino_gym import DominoEnv


class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.start_time = None
        self.pbar = None
        self.last_steps = 0

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.pbar = tqdm(total=self.total_timesteps, desc="Training RecurrentPPO (High Entropy)", unit="steps")

    def _on_step(self) -> bool: return True

    def _on_rollout_end(self) -> None:
        now = time.time()
        steps = self.num_timesteps
        delta_steps = steps - self.last_steps
        if delta_steps > 0 and self.pbar is not None: self.pbar.update(delta_steps)
        self.last_steps = steps
        
        logs = self.logger.name_to_value
        ep_rew = logs.get("rollout/ep_rew_mean", None)
        ep_len = logs.get("rollout/ep_len_mean", None)
        clip_frac = logs.get("train/clip_fraction", None)
        
        postfix = {}
        if ep_rew is not None: postfix["rew"] = f"{ep_rew:.3f}"
        if ep_len is not None: postfix["len"] = f"{ep_len:.1f}"
        if clip_frac is not None: postfix["clip"] = f"{clip_frac:.3f}" 
        if self.pbar is not None: self.pbar.set_postfix(postfix)

    def _on_training_end(self) -> None:
        if self.pbar is not None: self.pbar.close()


def make_env_fn(index: int, seed: int = 0):
    def _init():
        from domino_gym import DominoEnv
        env = DominoEnv(num_players=2)
        env.seed(seed + index)
        return env
    return _init


def main():
   
    TOTAL_TIMESTEPS = 5_000_000
    N_ENVS = 16             
    N_STEPS = 128           
    BATCH_SIZE = 1024       
    N_EPOCHS = 5            
    LEARNING_RATE = 3e-4   
    SEED = 42

    set_random_seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    LOG_DIR = "./logs_recurrent"
    os.makedirs(LOG_DIR, exist_ok=True)

    env_fns = [make_env_fn(i, seed=SEED) for i in range(N_ENVS)]
    venv = SubprocVecEnv(env_fns, start_method='spawn') 
    venv = VecMonitor(venv, LOG_DIR)

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=venv,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,       
        ent_coef=0.1,          
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device="cuda",
        policy_kwargs={
            "net_arch": [{"pi": [128, 128], "vf": [128, 128]}], #
            "lstm_hidden_size": 128, 
            "n_lstm_layers": 1,
            "enable_critic_lstm": True
        }
    )

    cb = ProgressCallback(total_timesteps=TOTAL_TIMESTEPS)
    
    print("--- CONFIGURACIÓN AGRESIVA DE ENTRENAMIENTO ---")
    print("Objetivo: Romper el equilibrio 50% Win Rate vs Random")
    print(f"Entropía: {model.ent_coef:.2f} (Alta exploración)")
    print(f"Learning Rate: {model.learning_rate}")
    print("-------------------------------------------------")

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb, progress_bar=False)

    model_path = os.path.join(LOG_DIR, "domino_recurrent_ppo_final.zip")
    model.save(model_path)
    print(f"Modelo guardado en: {model_path}")
    venv.close()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()