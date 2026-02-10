
import random
from collections import deque

class DominoGame:
    def __init__(self, num_players=2, teams_mode=False, seed=None, tiles_per_player: int = 10):
       
        self.num_players = int(num_players)
        if self.num_players not in (2, 4):
            raise ValueError("num_players debe ser 2 o 4")
        self.teams_mode = bool(teams_mode)
        self._seed = seed
        self.tiles_per_player = int(tiles_per_player)

       
        self.all_pieces = [(i, j) for i in range(10) for j in range(i, 10)]

     
        self.reset()

  
    def reset(self):
     
        if self._seed is not None:
            random.seed(self._seed)

     
        self.piezas = list(self.all_pieces)
        random.shuffle(self.piezas)

      
        tpp = self.tiles_per_player
        self.hands = {}
        for p in range(self.num_players):
            start = p * tpp
            end = start + tpp
           
            self.hands[p] = list(self.piezas[start:end])

     
        self.mesa = []
        self.center_tile = None
        self.extremos = [-1, -1] 

       
        self.current_player = random.randint(0, self.num_players - 1)
        self.play_sequence = []
        self.pass_count = 0
        self.winner = -1
        self.game_over = False

      
        self.history_left = []
        self.history_right = []

        return {}

  

    def get_valid_moves(self, player):
       
        if player not in self.hands:
            return []

        hand = list(self.hands[player])
        if self.center_tile is None:
          
            return [(f, 'L') for f in hand]

        valid = []
        l_val, r_val = self.extremos
        for f in hand:
            v1, v2 = f
            if v1 == l_val or v2 == l_val:
                valid.append((f, 'L'))
            if v1 == r_val or v2 == r_val:
                valid.append((f, 'R'))
        return valid

    def _place_tile(self, player, ficha, lado):
      
        if ficha not in self.hands[player]:
            return False
      
        self.hands[player].remove(ficha)
      
        self.mesa.append((player, ficha, lado))

        if self.center_tile is None:
            self.center_tile = ficha
            self.extremos = [ficha[0], ficha[1]]
        
            self.history_right.append({'ficha': ficha, 'conector': ficha[0], 'nuevo_extremo': ficha[1]})
        else:
            if lado == 'L':
                target = self.extremos[0]
                v1, v2 = ficha
                nuevo = v2 if v1 == target else v1
                self.extremos[0] = nuevo
                self.history_left.append({'ficha': ficha, 'conector': target, 'nuevo_extremo': nuevo})
            else:
                target = self.extremos[1]
                v1, v2 = ficha
                nuevo = v2 if v1 == target else v1
                self.extremos[1] = nuevo
                self.history_right.append({'ficha': ficha, 'conector': target, 'nuevo_extremo': nuevo})

        self.play_sequence.append((player, ficha, lado))
        return True

    def _calculate_winner_by_points(self):
        sums = {p: sum(f[0] + f[1] for f in h) for p, h in self.hands.items()}
      
        return min(sums, key=sums.get)

    
    def step(self, action):
       
        player = self.current_player

        if action is None:
            self.pass_count += 1
        else:
            ficha, lado = action
           
            if ficha not in self.hands[player]:
                
                return -1.0, True
           
            if self.center_tile is not None:
                if lado == 'L':
                    if not (ficha[0] == self.extremos[0] or ficha[1] == self.extremos[0]):
                        return -1.0, True
                else:
                    if not (ficha[0] == self.extremos[1] or ficha[1] == self.extremos[1]):
                        return -1.0, True
            ok = self._place_tile(player, ficha, lado)
            if not ok:
                return -1.0, True
           
            self.pass_count = 0

       
        if len(self.hands[player]) == 0:
            self.winner = player
            self.game_over = True
            return 1.0, True

      
        if self.pass_count >= self.num_players:
            self.game_over = True
            self.winner = self._calculate_winner_by_points()
            return 0.0, True

       
        self.current_player = (self.current_player + 1) % self.num_players
        return 0.0, False

  
    def get_state_for_player(self, player, history_len=20):
       
        return {
            'hand': list(self.hands.get(player, [])),
            'extremos': list(self.extremos),
            'history_left': list(self.history_left[-history_len:]),
            'history_right': list(self.history_right[-history_len:]),
            'current_player': self.current_player,
            'num_players': self.num_players,
            'center_tile': self.center_tile
        }
