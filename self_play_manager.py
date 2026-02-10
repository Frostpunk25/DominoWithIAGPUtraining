
import os
import shutil
import subprocess
from pathlib import Path
import time

CHECKPOINT_DIR = Path("./checkpoints")
BEST_DIR = Path("./best_models")
LOG_DIR = Path("./logs_recurrent")

for p in [CHECKPOINT_DIR, BEST_DIR, LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

def run_training(timesteps=5_000_000):
    print("\n--- Iniciando Entrenamiento ---")
    cmd = ["python", "train_recurrent.py"]
    subprocess.run(cmd, check=True)

def run_evaluation(model_path, n_games=500):
    print(f"\n--- Evaluando {model_path.name} ---")
    cmd = [
        "python", "evaluate_parallel.py", 
        "--model", str(model_path), 
        "--n-games", str(n_games)
    ]
    subprocess.run(cmd)

def save_checkpoint(model_source, name):
    dest = CHECKPOINT_DIR / name
    shutil.copy(str(model_source), str(dest))
    print(f"Checkpoint guardado en: {dest}")

if __name__ == "__main__":
    main_flow = True
    
    while main_flow:
        print("=== MENÚ SELF-PLAY ===")
        print("1. Entrenar (5M pasos)")
        print("2. Evaluar Último Modelo")
        print("3. Salir")
        
        opt = input("Opción: ")
        
        if opt == "1":
            run_training()
       
            fresh_model = LOG_DIR / "domino_recurrent_ppo_final.zip"
            if fresh_model.exists():
                ts = int(time.time())
                save_checkpoint(fresh_model, f"model_{ts}.zip")
        elif opt == "2":
            ckpts = sorted(CHECKPOINT_DIR.glob("*.zip"))
            if not ckpts:
                print("No hay checkpoints.")
                continue
            latest = ckpts[-1]
            run_evaluation(latest, n_games=200)
        else:
            break