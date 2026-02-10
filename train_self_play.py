
import os
import time
import multiprocessing
import shutil
import random as rng
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from tqdm import tqdm

from sb3_contrib import RecurrentPPO
from domino_gym import DominoEnv


TOTAL_CYCLES = 10          
STEPS_PER_CYCLE = 750_000   
EVAL_GAMES = 500           
N_ENVS = 30                
RANDOM_RATIO = 0.2           


BEST_DIR = Path("./best_models")
GEN_DIR = Path("./generations")
BEST_DIR.mkdir(parents=True, exist_ok=True)
GEN_DIR.mkdir(parents=True, exist_ok=True)


CURRENT_BEST = "./logs_recurrent/domino_recurrent_ppo_final.zip"


class CurriculumOpponent:
    """Oponente Mixto: 80% Maestro, 20% Random."""
    def __init__(self, model, random_ratio=0.2):
        self.model = model
        self.random_ratio = random_ratio
    
    def predict(self, obs, legal_moves):
        if rng.random() < self.random_ratio:
           
            valid_idxs = np.where(legal_moves == 1)[0]
            if len(valid_idxs) > 0:
                return rng.choice(valid_idxs)
            return 0
        else:
           
            action, _ = self.model.predict(obs, deterministic=True)
            return action

def make_env_fn(index: int, seed: int = 0, opponent_policy=None):
    def _init():
        from domino_gym import DominoEnv
        env = DominoEnv(num_players=2, opponent_policy=opponent_policy)
      
        env.seed(seed + index)
        return env
    return _init


class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.start_time = None
        self.pbar = None
        self.last_steps = 0

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Self-Play (Mixed)", unit="steps")

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
        fps = logs.get("time/fps", None)
        
       
        postfix = {}
        if ep_rew is not None: postfix["rew"] = f"{ep_rew:.3f}"
        if ep_len is not None: postfix["len"] = f"{ep_len:.1f}"
        if fps is not None: postfix["fps"] = int(fps)
        
        if self.pbar is not None: self.pbar.set_postfix(postfix)

    def _on_training_end(self) -> None:
        if self.pbar is not None: self.pbar.close()


def evaluate_match(student_model, teacher_model_path, n_games=200):
    print(f"\n--- ARENA PURA: Student vs Teacher ({n_games} partidas) ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = RecurrentPPO.load(teacher_model_path, device=device)
    
    class SimpleOpponent:
        def __init__(self, model): self.model = model
        def predict(self, obs, legal_moves):
            action, _ = self.model.predict(obs, deterministic=True)
            return action

    opponent_wrapped = SimpleOpponent(teacher)
    
    env_fns = [make_env_fn(i, seed=999, opponent_policy=opponent_wrapped.predict) for i in range(8)] 
    venv = SubprocVecEnv(env_fns, start_method='spawn')
    venv = VecMonitor(venv)

    obs = venv.reset()
    states = None
    episode_starts = np.ones((8,), dtype=bool)
    
    wins = 0
    games = 0
    
    while games < n_games:
        actions, states = student_model.predict(obs, state=states, episode_start=episode_starts, deterministic=True)
        obs, rewards, dones, infos = venv.step(actions)
        episode_starts = dones
        
        for i, info in enumerate(infos):
            if dones[i]:
                result = info.get("result", None)
                if result == "win": wins += 1
                games += 1
                if games >= n_games: break
    
    venv.close()
    win_rate = wins / n_games
    print(f"RESULTADO ARENA: Student ganó el {win_rate*100:.1f}% de las partidas.")
    return win_rate


def main():
    set_random_seed(42)
    device = "cuda"
    
    global CURRENT_BEST
    
    if not os.path.exists(CURRENT_BEST):
        print(f"Error: No se encuentra {CURRENT_BEST}")
        return

    print(f"\n{'='*60}")
    print(f"INICIANDO ENTRENAMIENTO CURRICULAR ROBUSTO")
    print(f"Hardware: 30 Entornos || N_GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Semillas: ALEATORIAS por ciclo (Para evitar memorización)")
    print(f"Estrategia: {int((1-RANDOM_RATIO)*100)}% Self-Play | {int(RANDOM_RATIO*100)}% Random")
    print(f"{'='*60}\n")

    for cycle in range(1, TOTAL_CYCLES + 1):
        
        cycle_seed = rng.randint(1, 100000)
        
        print(f"\n{'='*50}")
        print(f"CICLO {cycle} / {TOTAL_CYCLES}")
        print(f"Oponente actual: {os.path.basename(CURRENT_BEST)}")
        print(f"Semilla del ciclo: {cycle_seed} (Aleatoria)")
        print(f"{'='*50}")

        
        teacher_model = RecurrentPPO.load(CURRENT_BEST, device=device)
        opponent_wrapper = CurriculumOpponent(teacher_model, random_ratio=RANDOM_RATIO)

       
        env_fns = [make_env_fn(i, seed=cycle_seed, opponent_policy=opponent_wrapper.predict) for i in range(N_ENVS)]
        
        
        log_dir = f"./logs_self_play_cycle_{cycle}"
        
        venv = SubprocVecEnv(env_fns, start_method='spawn')
        venv = VecMonitor(venv, log_dir)

        
        student_model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=venv,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=N_ENVS * 128, 
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05, 
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0, 
            device=device,
            policy_kwargs={
                "net_arch": [{"pi": [128, 128], "vf": [128, 128]}], 
                "lstm_hidden_size": 128,
                "n_lstm_layers": 1,
                "enable_critic_lstm": True
            }
        )

      
        cb = ProgressCallback(total_timesteps=STEPS_PER_CYCLE)
        student_model.learn(total_timesteps=STEPS_PER_CYCLE, callback=cb, progress_bar=False)
        
        venv.close() 

       
        try:
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
                print(f"Limpieza: Carpeta {log_dir} eliminada.")
        except Exception as e:
            print(f"Aviso: No se pudo limpiar logs ({e})")

     
        candidate_path = GEN_DIR / f"candidate_cycle_{cycle}.zip"
        student_model.save(str(candidate_path))
        print(f"Candidato guardado en: {candidate_path}")

    
        fresh_student = RecurrentPPO.load(str(candidate_path), device=device)
        win_rate = evaluate_match(fresh_student, CURRENT_BEST, n_games=EVAL_GAMES)

      
        if win_rate > 0.55: 
            print(f"--- PROMOCIÓN: El nuevo modelo es SUPERIOR ({win_rate*100:.1f}% win rate) ---")
            
            best_copy_path = BEST_DIR / f"domino_gen_{cycle}.zip"
            shutil.copy(str(candidate_path), str(best_copy_path))
            
         
            CURRENT_BEST = str(candidate_path)
        else:
            print(f"--- DESCARTE: El nuevo modelo no superó al maestro ({win_rate*100:.1f}%) ---")
            print("Continuaremos entrenando contra la misma generación anterior.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()