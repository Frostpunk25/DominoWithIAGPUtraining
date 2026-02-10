
import torch
import numpy as np
import os

from domino_gym import ALL_PIECES, PIECE_TO_IDX
from domino_engine import DominoGame

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None


class DominoAI:
    
    
    def __init__(self, model_path="./best_models/domino_gen_10.zip", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.lstm_states = None
        self.episode_starts = None
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
            
        if RecurrentPPO is None:
            raise ImportError("sb3_contrib no está instalado")
            
        print(f"Cargando modelo desde: {model_path}")
        self.model = RecurrentPPO.load(model_path, device=self.device)
        self.model.set_training_mode(False)
        print(f"✓ Modelo cargado en {self.device}")
        
        self.reset_states()
    
    def reset_states(self):
      
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)
    
    def _build_obs(self, game: DominoGame, player: int = 1) -> np.ndarray:
       
        obs = np.zeros(169, dtype=np.float32)
        
      
        hand = game.hands[player]
        for p in hand:
            sorted_p = tuple(sorted(p))
            if sorted_p in PIECE_TO_IDX:
                obs[PIECE_TO_IDX[sorted_p]] = 1.0
        
      
        played_set = set()
        for move in game.mesa:
            ficha = move[1]
            sorted_p = tuple(sorted(ficha))
            played_set.add(sorted_p)
        
        for p in played_set:
            if p in PIECE_TO_IDX:
                obs[55 + PIECE_TO_IDX[p]] = 1.0
        
       
        if game.center_tile is None:
            obs[110] = -1.0
            obs[111] = -1.0
        else:
            obs[110] = game.extremos[0] / 9.0
            obs[111] = game.extremos[1] / 9.0
        
      
        other = 1 - player
        obs[112] = len(game.hands[other]) / 9.0
        
       
        obs[113] = len(hand) / 9.0
        
      
        legal_mask = np.zeros(55, dtype=np.float32)
        valid_moves = game.get_valid_moves(player)
        
        for move in valid_moves:
            ficha = move[0]
            sorted_p = tuple(sorted(ficha))
            if sorted_p in PIECE_TO_IDX:
                legal_mask[PIECE_TO_IDX[sorted_p]] = 1.0
        
        obs[114:169] = legal_mask
        
        return obs
    
    def predict(self, game: DominoGame, player: int = 1, deterministic: bool = True):
      
        valid_moves = game.get_valid_moves(player)
        if not valid_moves:
            return None
        
       
        obs = self._build_obs(game, player)
        obs_tensor = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        
       
        with torch.no_grad():
            action, self.lstm_states = self.model.predict(
                obs_tensor,
                state=self.lstm_states,
                episode_start=self.episode_starts,
                deterministic=deterministic
            )
        
        self.episode_starts = np.zeros((1,), dtype=bool)
        
       
        action_idx = int(action[0]) if isinstance(action, (list, np.ndarray)) else int(action)
        
        if action_idx < len(ALL_PIECES):
            selected_piece = ALL_PIECES[action_idx]
            
          
            for move in valid_moves:
                ficha, lado = move
                if tuple(sorted(ficha)) == selected_piece:
                    return move
            
           
            return valid_moves[0]
        
        return valid_moves[0]