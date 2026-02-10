import pygame
import sys
import random
import numpy as np
import torch


try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    print("ERROR: sb3_contrib no instalado. pip install sb3-contrib")
    sys.exit(1)

from domino_engine import DominoGame
from domino_gym import ALL_PIECES, PIECE_TO_IDX


SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
BG_COLOR = (30, 90, 30)
TILE_COLOR = (245, 245, 235)
DOT_COLOR = (15, 15, 15)
HIGHLIGHT = (255, 215, 0)

TILE_W = 38
TILE_H = 76
GAP = 2
MARGIN_TOP = 120
MARGIN_BOTTOM = 150
MARGIN_LEFT = 100
MARGIN_RIGHT = 100


MODEL_PATH = "./best_models/domino_gen_10.zip"


class DominoGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Dominó Pro IA - 1vs1 (Gen 10)")
        self.font = pygame.font.SysFont("Segoe UI", 18, bold=True)
        self.big_font = pygame.font.SysFont("Segoe UI", 40, bold=True)
        self.clock = pygame.time.Clock()
        
       
        self.device = "cpu" 
        
        print(f"Dispositivo: {self.device}")
        print(f"Cargando modelo: {MODEL_PATH}")
        
        try:
            self.model = RecurrentPPO.load(MODEL_PATH, device=self.device)
            print("✓ Modelo cargado en CPU exitosamente")
        except Exception as e:
            print(f"ERROR al cargar modelo: {e}")
            sys.exit(1)
        
       
        self.state = "MENU"
        self.game = None
        self.tile_rects = []
        self.selected_tile_idx = None
        self.last_click_time = 0
        
        
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)

    def draw_pips(self, surface, x, y, number, size, vertical):
        
        cx, cy = x + size//2, y + size//2
        offset = size // 4
        r = 3
        
        pips_map = [
            (0,0), (-1,-1), (1,1), (1,-1), (-1,1), 
            (-1,0), (1,0), (0,-1), (0,1)
        ]
        
        active = []
        if number == 1: active = [0]
        elif number == 2: active = [1,2]
        elif number == 3: active = [0,1,2]
        elif number == 4: active = [1,2,3,4]
        elif number == 5: active = [0,1,2,3,4]
        elif number == 6: active = [1,2,3,4,5,6]
        elif number == 7: active = [0,1,2,3,4,5,6]
        elif number == 8: active = [1,2,3,4,5,6,7,8]
        elif number == 9: active = [0,1,2,3,4,5,6,7,8]

        for i in active:
            px, py = pips_map[i]
            pygame.draw.circle(surface, DOT_COLOR, (int(cx + px*offset), int(cy + py*offset)), r)

    def draw_tile_graphic(self, x, y, v1, v2, vertical=True, selected=False):
        w, h = (TILE_W, TILE_H) if vertical else (TILE_H, TILE_W)
        rect = pygame.Rect(x, y, w, h)
        
        pygame.draw.rect(self.screen, (20,20,20), (x+2, y+2, w, h), border_radius=4)
        pygame.draw.rect(self.screen, TILE_COLOR, rect, border_radius=4)
        
        color_borde = HIGHLIGHT if selected else (80,80,80)
        pygame.draw.rect(self.screen, color_borde, rect, 2 if selected else 1, border_radius=4)
        
        half = TILE_W
        
        if vertical:
            pygame.draw.line(self.screen, (150,150,150), (x+4, y+h//2), (x+w-4, y+h//2), 1)
            self.draw_pips(self.screen, x, y, v1, half, True)
            self.draw_pips(self.screen, x, y+h//2, v2, half, True)
        else:
            pygame.draw.line(self.screen, (150,150,150), (x+w//2, y+4), (x+w//2, y+h-4), 1)
            self.draw_pips(self.screen, x, y, v1, half, False)
            self.draw_pips(self.screen, x+w//2, y, v2, half, False)
        
        return rect

    def calculate_snake_layout(self, history, start_x, start_y, start_direction):
        layout = []
        curr_x, curr_y = start_x, start_y
        direction = start_direction 
        vertical_dir = -1 if start_direction == 1 else 1
        step_long = TILE_H + GAP
        step_short = TILE_W + GAP
        
        for move in history:
            ficha = move['ficha']
            conector = move['conector']
            nuevo = move['nuevo_extremo']
            is_double = (ficha[0] == ficha[1])
            
            draw_x, draw_y = curr_x, curr_y
            draw_vertical = False
            val_left, val_right = 0, 0
            offset = 0
            
            if is_double:
                draw_vertical = True
                draw_y = curr_y - (TILE_H - TILE_W)//2
                if direction == -1: 
                    draw_x = curr_x - TILE_W 
                val_left, val_right = ficha[0], ficha[1]
                offset = step_short
            else:
                draw_vertical = False
                if direction == -1: 
                    draw_x = curr_x - TILE_H
                if direction == 1: 
                    val_left, val_right = conector, nuevo
                else: 
                    val_left, val_right = nuevo, conector
                offset = step_long

            layout.append({
                'x': draw_x, 'y': draw_y, 
                'v1': val_left, 'v2': val_right, 
                'vert': draw_vertical
            })
            curr_x += (offset * direction)
            
            limit_right = SCREEN_WIDTH - MARGIN_RIGHT
            limit_left = MARGIN_LEFT
            next_x = curr_x + (step_long * direction)
            hit_right = (direction == 1) and (next_x > limit_right)
            hit_left = (direction == -1) and (next_x < limit_left)
            
            if hit_right or hit_left:
                direction *= -1
                if direction == 1: 
                    curr_x = limit_left
                else: 
                    curr_x = limit_right
                curr_y += (step_long * vertical_dir)
                vertical_dir *= -1
                
        return layout

    def draw_board(self):
        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        
        if self.game.center_tile is None: 
            return

        c_ficha = self.game.center_tile
        c_vert = (c_ficha[0] == c_ficha[1])
        start_x = cx - (TILE_W//2 if c_vert else TILE_H//2)
        start_y = cy - (TILE_W//2) 
        self.draw_tile_graphic(start_x, start_y, c_ficha[0], c_ficha[1], c_vert)
        
        off_x = (TILE_W if c_vert else TILE_H) + GAP
        r_x = start_x + off_x
        r_y = start_y + (TILE_H//2 if c_vert else TILE_W//2) - TILE_W//2
        
       
        history_right_to_draw = self.game.history_right[1:] if self.game.history_right else []
        
        layout_r = self.calculate_snake_layout(history_right_to_draw, r_x, r_y, 1)
        for item in layout_r: 
            self.draw_tile_graphic(item['x'], item['y'], item['v1'], item['v2'], item['vert'])
            
        l_x = start_x - GAP
        l_y = start_y + (TILE_H//2 if c_vert else TILE_W//2) - TILE_W//2
        layout_l = self.calculate_snake_layout(self.game.history_left, l_x, l_y, -1)
        for item in layout_l: 
            self.draw_tile_graphic(item['x'], item['y'], item['v1'], item['v2'], item['vert'])

    def draw_hands(self):
        self.tile_rects = [] 
        hand = self.game.hands[0]
        total_w = len(hand) * (TILE_W + 5)
        start_x = (SCREEN_WIDTH - total_w) // 2
        y = SCREEN_HEIGHT - TILE_H - 20
        
        for i, f in enumerate(hand):
            sel = (i == self.selected_tile_idx)
            offset = -15 if sel else 0
           
            rect = self.draw_tile_graphic(
                start_x + i*(TILE_W+5), y + offset, 
                f[0], f[1], True, sel
            )
            self.tile_rects.append((rect, i, f))
            
       
        n_opp = len(self.game.hands[1])
        h = n_opp * 25
        sy = (SCREEN_HEIGHT - h) // 2
        sx = SCREEN_WIDTH - 50
        for k in range(n_opp): 
            pygame.draw.rect(
                self.screen, (80,60,40), 
                (sx, sy+k*25, 40, 20), 
                border_radius=3
            )

    def _build_obs_for_model(self):
        obs = np.zeros(169, dtype=np.float32)
        
       
        ai_hand = self.game.hands[1] 
        for p in ai_hand:
            sorted_p = tuple(sorted(p))
            if sorted_p in PIECE_TO_IDX:
                obs[PIECE_TO_IDX[sorted_p]] = 1.0

       
        played_set = set()
        for move in self.game.mesa:
            ficha = move[1]
            sorted_p = tuple(sorted(ficha))
            played_set.add(sorted_p)
        
        for p in played_set:
            if p in PIECE_TO_IDX:
                obs[55 + PIECE_TO_IDX[p]] = 1.0

       
        if self.game.center_tile is None:
            obs[110] = -1.0
            obs[111] = -1.0
        else:
            left_val = self.game.extremos[0]
            right_val = self.game.extremos[1]
            obs[110] = left_val / 9.0
            obs[111] = right_val / 9.0

       
        human_hand_len = len(self.game.hands[0])
        obs[112] = human_hand_len / 9.0

       
        obs[113] = len(ai_hand) / 9.0

       
        legal_mask = np.zeros(55, dtype=np.float32)
        valid_moves = self.game.get_valid_moves(1)
        
        if not valid_moves:
            pass
        else:
            for move in valid_moves:
                ficha = move[0]
                sorted_p = tuple(sorted(ficha))
                if sorted_p in PIECE_TO_IDX:
                    legal_mask[PIECE_TO_IDX[sorted_p]] = 1.0
        
        obs[114:169] = legal_mask
        return obs

    def get_ai_move(self):
        obs = self._build_obs_for_model()
        obs_batch = obs[None, :]
        valid_moves = self.game.get_valid_moves(1)
        
        if not valid_moves:
            return None
        
        with torch.no_grad():
            action, self.lstm_states = self.model.predict(
                obs_batch, state=self.lstm_states,
                episode_start=self.episode_starts, deterministic=True
            )
        
        self.episode_starts = np.zeros((1,), dtype=bool)
        
        if torch.is_tensor(action):
            action_idx = int(action.item())
        else:
            action_idx = int(action[0])

        if action_idx < len(ALL_PIECES):
            selected_piece = ALL_PIECES[action_idx]
            for move in valid_moves:
                ficha, lado = move
                if tuple(sorted(ficha)) == selected_piece:
                    return move
            return valid_moves[0]
        
        return valid_moves[0] if valid_moves else None

    def draw_menu(self):
        self.screen.fill((20, 20, 25))
        
        t = self.big_font.render("DOMINÓ PRO IA", True, HIGHLIGHT)
        self.screen.blit(t, (SCREEN_WIDTH//2 - t.get_width()//2, 80))
        
        sub = self.font.render("Modo: 1 vs 1 (Gen 10 - RecurrentPPO)", True, (200, 200, 200))
        self.screen.blit(sub, (SCREEN_WIDTH//2 - sub.get_width()//2, 140))
        
        mx, my = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()[0]
        current_time = pygame.time.get_ticks()
        input_blocked = (current_time - self.last_click_time < 300)

        rect = pygame.Rect(SCREEN_WIDTH//2 - 150, 300, 300, 80)
        hover = rect.collidepoint(mx, my)
        col = (50, 120, 50) if hover else (40, 40, 50)
        
        pygame.draw.rect(self.screen, col, rect, border_radius=12)
        pygame.draw.rect(self.screen, HIGHLIGHT if hover else (200,200,200), rect, 3, border_radius=12)
        
        surf = self.big_font.render("JUGAR 1 vs 1", True, (255,255,255))
        self.screen.blit(surf, (
            rect.centerx - surf.get_width()//2, 
            rect.centery - surf.get_height()//2
        ))
        
        instr = self.font.render("Haz clic para iniciar partida contra la IA", True, (150, 150, 150))
        self.screen.blit(instr, (SCREEN_WIDTH//2 - instr.get_width()//2, 420))
        
        if click and rect.collidepoint(mx, my) and not input_blocked:
            random.seed() 
            self.game = DominoGame(num_players=2, tiles_per_player=10)
            self.state = "PLAY"
            self.selected_tile_idx = None
            self.last_click_time = current_time
            self.lstm_states = None
            self.episode_starts = np.ones((1,), dtype=bool)
            pygame.time.delay(100)

    def run(self):
        while True:
            if self.state == "MENU":
                self.draw_menu()
                for e in pygame.event.get():
                    if e.type == pygame.QUIT: 
                        pygame.quit()
                        sys.exit()
                pygame.display.flip()
                
            elif self.state == "PLAY":
                self.screen.fill(BG_COLOR)
                self.draw_board()
                self.draw_hands()
                
                if self.game.game_over:
                    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                    overlay.fill((0,0,0, 180))
                    self.screen.blit(overlay, (0,0))
                    
                    if self.game.winner == 0:
                        win_txt = "¡VICTORIA!"
                        col = (0, 255, 0)
                    elif self.game.winner == 1:
                        win_txt = "GANÓ LA IA"
                        col = (255, 100, 100)
                    else:
                        win_txt = "EMPATE"
                        col = (255, 255, 0)
                        
                    surf = self.big_font.render(win_txt, True, col)
                    self.screen.blit(surf, (
                        SCREEN_WIDTH//2 - surf.get_width()//2, 
                        SCREEN_HEIGHT//2 - 60
                    ))
                    
                    sub = self.font.render("Click para volver al Menú", True, (255,255,255))
                    self.screen.blit(sub, (
                        SCREEN_WIDTH//2 - sub.get_width()//2, 
                        SCREEN_HEIGHT//2 + 20
                    ))
                  
                if not self.game.game_over:
                    turn_txt = f"Turno: {'TÚ' if self.game.current_player==0 else 'IA'}"
                    self.screen.blit(
                        self.font.render(turn_txt, True, (255,255,255)), 
                        (20, SCREEN_HEIGHT-100)
                    )
                    
                    info_txt = f"Tu mano: {len(self.game.hands[0])} | IA: {len(self.game.hands[1])}"
                    self.screen.blit(
                        self.font.render(info_txt, True, (200,200,200)), 
                        (20, 20)
                    )
                
                if not self.game.game_over:
                    turn = self.game.current_player
                    
                    if turn == 0:  
                        valid_moves = self.game.get_valid_moves(0)
                        if not valid_moves:
                            self.screen.blit(
                                self.big_font.render("¡PASO!", True, (255,0,0)), 
                                (SCREEN_WIDTH//2-50, SCREEN_HEIGHT-200)
                            )
                            pygame.display.flip()
                            pygame.time.delay(800)
                            self.game.step(None)
                            
                    else:  
                        pygame.display.flip()
                        pygame.time.delay(800)
                        
                        ai_move = self.get_ai_move()
                        
                        if not ai_move:
                            self.screen.blit(
                                self.big_font.render("IA PASA", True, (200, 200, 255)), 
                                (SCREEN_WIDTH//2 - 80, SCREEN_HEIGHT - 200)
                            )
                            pygame.display.flip()
                            pygame.time.delay(800)
                            self.game.step(None)
                        else:
                            self.game.step(ai_move)

                for e in pygame.event.get():
                    if e.type == pygame.QUIT: 
                        pygame.quit()
                        sys.exit()
                    
                    if e.type == pygame.MOUSEBUTTONDOWN:
                        if self.game.game_over:
                            self.last_click_time = pygame.time.get_ticks()
                            self.state = "MENU" 
                            self.game = None
                            self.lstm_states = None
                        
                        elif self.game.current_player == 0:
                            mx, my = pygame.mouse.get_pos()

                            for rect, idx, ficha in self.tile_rects:
                                if rect.collidepoint(mx, my):
                                    valid = self.game.get_valid_moves(0)
                                    possible_moves = [m for m in valid if m[0] == ficha]
                                    
                                    if not possible_moves:
                                        self.selected_tile_idx = None
                                    elif len(possible_moves) == 1:
                                        self.selected_tile_idx = idx 
                                        self.game.step(possible_moves[0])
                                        self.selected_tile_idx = None 
                                    else:
                                       
                                        self.selected_tile_idx = idx
                                          
                                        rel_y = my - rect.y
                                        is_top_half = rel_y < (rect.height / 2)
                                        
                                       
                                        clicked_val = ficha[0] if is_top_half else ficha[1]
                                        
                                        move_to_play = None
                                        
                                       
                                        for move in possible_moves:
                                            side = move[1] # 'L' o 'R'
                                          
                                            target_val = self.game.extremos[0] if side == 'L' else self.game.extremos[1]
                                            
                                           
                                            if clicked_val == target_val:
                                                move_to_play = move
                                                break
                                        
                                      
                                        if not move_to_play:
                                            move_to_play = possible_moves[0]
                                            
                                        self.game.step(move_to_play)
                                        self.selected_tile_idx = None
                                    
                                    break

                pygame.display.flip()
                self.clock.tick(30)


if __name__ == "__main__":
    DominoGUI().run()