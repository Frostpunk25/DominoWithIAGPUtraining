
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Callable, Tuple


MAX_PIP = 9
ALL_PIECES = []
for a in range(MAX_PIP + 1):
    for b in range(a, MAX_PIP + 1):
        ALL_PIECES.append((a, b))

PIECE_TO_IDX = {p: i for i, p in enumerate(ALL_PIECES)}
N_PIECES = len(ALL_PIECES)


OBS_SIZE = (N_PIECES * 2) + 4 + N_PIECES

def sorted_piece(t):
    return tuple(sorted(t))

class DominoEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_players: int = 2,
        seed: Optional[int] = None,
        opponent_policy: Optional[Callable] = None,
        reward_config: dict = None,
    ):
        super().__init__()
        self.num_players = num_players
        self.rng = random.Random(seed)
        self.np_random = np.random.RandomState(seed)

        self.action_space = spaces.Discrete(N_PIECES)

   
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1.0,
            shape=(OBS_SIZE,),
            dtype=np.float32,
        )

        self.deck = []
        self.hands = [[] for _ in range(self.num_players)]
        self.table = []
        self.played_pieces_set = set() 
        self.current_player = 0
        self.consecutive_passes = 0 

        self.opponent_policy = opponent_policy

        self.reward_config = reward_config or {
            "win": 1.0,
            "lose": -1.0,
            "draw": 0.0,   
           
            "invalid": 0.0, 
            "step_penalty": -0.01, 
        }

        self.max_hand_size = 9 
        self._seed = seed

    def seed(self, seed=None):
        self._seed = seed
        self.rng.seed(seed)
        self.np_random.seed(seed)

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.seed(seed)

        self.deck = ALL_PIECES.copy()
        self.rng.shuffle(self.deck)

        self.hands = [[] for _ in range(self.num_players)]
        for p in range(self.num_players):
            for _ in range(self.max_hand_size):
                if self.deck:
                    self.hands[p].append(self.deck.pop())

        self.table = []
        self.played_pieces_set = set()
        self.consecutive_passes = 0
        self.current_player = self.rng.randint(0, self.num_players - 1)

        obs = self._build_obs()
        return obs, {}

    def _build_obs(self) -> np.ndarray:
        obs = np.zeros(OBS_SIZE, dtype=np.float32)

      
        for p in self.hands[0]:
            obs[PIECE_TO_IDX[sorted_piece(p)]] = 1.0

       
        for p in self.played_pieces_set:
            obs[N_PIECES + PIECE_TO_IDX[sorted_piece(p)]] = 1.0

       
        left, right = -1, -1
        if self.table:
            left = self.table[0][0]
            right = self.table[-1][1]

        obs[N_PIECES * 2] = left / MAX_PIP if left >= 0 else -1.0
        obs[N_PIECES * 2 + 1] = right / MAX_PIP if right >= 0 else -1.0

       
        obs[N_PIECES * 2 + 2] = len(self.hands[1]) / self.max_hand_size if self.num_players > 1 else 0.0

       
        obs[N_PIECES * 2 + 3] = len(self.hands[0]) / self.max_hand_size

       
        legal_mask = self._get_legal_indices(0) 
        obs[N_PIECES * 2 + 4 : N_PIECES * 2 + 4 + N_PIECES] = legal_mask

        return obs

   
    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        action = int(action)
        
       
        legal_moves = self._get_legal_indices(0)
        valid_indices = np.where(legal_moves == 1)[0]
        
       
        if len(valid_indices) > 0:
            if action >= N_PIECES or legal_moves[action] == 0:
               
                action = np.random.choice(valid_indices)
                info["action_clipped"] = True 
        else:
            
            pass

       
        if len(valid_indices) > 0:
            piece_tuple = ALL_PIECES[action]
            
            real_piece = None
            for p in self.hands[0]:
                if sorted_piece(p) == sorted_piece(piece_tuple):
                    real_piece = p
                    break
            
            if real_piece:
                self._play_piece(real_piece)
                self.hands[0].remove(real_piece)
                self.played_pieces_set.add(sorted_piece(real_piece))
                self.consecutive_passes = 0 
                reward += self.reward_config["step_penalty"]
        else:
            
            self.consecutive_passes += 1

      
        if not self.hands[0]:
            reward += self.reward_config["win"]
            terminated = True
            info["result"] = "win"
            info["winner"] = 0
            return self._build_obs(), reward, terminated, truncated, info

        
        opponent_won = False
        for p_idx in range(1, self.num_players):
            opp_legal_moves = self._get_legal_indices(p_idx)
            opp_valid_indices = np.where(opp_legal_moves == 1)[0]
            
            if len(opp_valid_indices) > 0:
                self.consecutive_passes = 0
                
                
                action_opp = self.np_random.choice(opp_valid_indices)
                
                piece_tuple_opp = ALL_PIECES[int(action_opp)]
                real_piece_opp = None
                for p in self.hands[p_idx]:
                    if sorted_piece(p) == sorted_piece(piece_tuple_opp):
                        real_piece_opp = p
                        break
                
                if real_piece_opp:
                    self._play_piece(real_piece_opp)
                    self.hands[p_idx].remove(real_piece_opp)
                    self.played_pieces_set.add(sorted_piece(real_piece_opp))
                
                if not self.hands[p_idx]:
                    reward += self.reward_config["lose"]
                    terminated = True
                    info["result"] = "lose"
                    info["winner"] = p_idx
                    return self._build_obs(), reward, terminated, truncated, info
            else:
                self.consecutive_passes += 1

        
        if self.consecutive_passes >= self.num_players:
            terminated = True
            my_points = sum(a+b for a,b in self.hands[0])
            opp_points = sum(a+b for a,b in self.hands[1]) if self.num_players > 1 else 99999
            
            info["blocked"] = True
            if my_points < opp_points:
                reward += self.reward_config["win"]
                info["result"] = "win"
                info["winner"] = 0
            elif my_points > opp_points:
                reward += self.reward_config["lose"]
                info["result"] = "lose"
                info["winner"] = 1
            else:
                reward += self.reward_config["draw"]
                info["result"] = "draw"
            return self._build_obs(), reward, terminated, truncated, info

        return self._build_obs(), reward, terminated, truncated, info

    def _play_piece(self, piece: Tuple[int, int]):
        if not self.table:
            self.table.append(piece)
            return

        left_val = self.table[0][0]
        right_val = self.table[-1][1]
        a, b = piece

        if b == left_val:
            self.table.insert(0, (a, b))
        elif a == left_val:
            self.table.insert(0, (b, a))
        elif a == right_val:
            self.table.append((a, b))
        elif b == right_val:
            self.table.append((b, a))

    def _get_legal_indices(self, p_idx: int) -> np.ndarray:
        mask = np.zeros(N_PIECES, dtype=np.int8)

        if not self.table:
            for ph in self.hands[p_idx]:
                mask[PIECE_TO_IDX[sorted_piece(ph)]] = 1
            return mask

        left = self.table[0][0]
        right = self.table[-1][1]

        for ph in self.hands[p_idx]:
            a, b = ph
            if a == left or b == left or a == right or b == right:
                mask[PIECE_TO_IDX[sorted_piece(ph)]] = 1

        return mask
    
    def _build_obs_for_player(self, p_idx: int) -> np.ndarray:
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        
        for p in self.hands[p_idx]:
            obs[PIECE_TO_IDX[sorted_piece(p)]] = 1.0
        
        for p in self.played_pieces_set:
            obs[N_PIECES + PIECE_TO_IDX[sorted_piece(p)]] = 1.0
            
        left, right = -1, -1
        if self.table:
            left = self.table[0][0]
            right = self.table[-1][1]
        obs[N_PIECES * 2] = left / MAX_PIP if left >= 0 else -1.0
        obs[N_PIECES * 2 + 1] = right / MAX_PIP if right >= 0 else -1.0
        
        other_idx = 1 if p_idx == 0 else 0
        obs[N_PIECES * 2 + 2] = len(self.hands[other_idx]) / self.max_hand_size
        obs[N_PIECES * 2 + 3] = len(self.hands[p_idx]) / self.max_hand_size
        
        mask = self._get_legal_indices(p_idx)
        obs[N_PIECES * 2 + 4 : N_PIECES * 2 + 4 + N_PIECES] = mask
        
        return obs

    def render(self):
        print("-" * 20)
        print(f"Mesa: [{self.table[0] if self.table else ''} ... {self.table[-1] if self.table else ''}]")
        print(f"Mano Agente: {self.hands[0]}")
        print("-" * 20)

    def close(self):
        pass