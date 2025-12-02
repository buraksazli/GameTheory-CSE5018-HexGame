import pygame
import math
import copy
import time
import heapq
import random
from collections import deque

# --- CONFIGURATION ---
BOARD_SIZE = 9
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
HEX_RADIUS = 25
BG_COLOR = (30, 30, 30)
EMPTY_COLOR = (200, 200, 200)
P1_COLOR = (255, 50, 50)  # Red (Top-Bottom)
P2_COLOR = (50, 50, 255)  # Blue (Left-Right)
TEXT_COLOR = (255, 255, 255)
BORDER_COLOR = (50, 50, 50)
RED_BORDER = (220, 20, 20)
BLUE_BORDER = (20, 20, 255)
BUTTON_COLOR = (70, 70, 70)
BUTTON_HOVER = (100, 100, 100)
BUTTON_SELECTED = (50, 150, 50)


# --- LOGIC CLASS ---
class HexBoard:
    def __init__(self, size):
        self.size = size
        self.board = [[0 for _ in range(size)] for _ in range(size)]
        self.turn = 1
        self.winner = None

    def is_valid_move(self, row, col):
        return 0 <= row < self.size and 0 <= col < self.size and self.board[row][col] == 0

    def place_stone(self, row, col):
        if self.is_valid_move(row, col) and self.winner is None:
            self.board[row][col] = self.turn
            if self.check_win(self.turn):
                self.winner = self.turn
            else:
                self.turn = 3 - self.turn
            return True
        return False

    def get_neighbors(self, r, c):
        directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                neighbors.append((nr, nc))
        return neighbors

    def check_win(self, player):
        queue = deque()
        visited = set()

        if player == 1:  # Top to Bottom
            for c in range(self.size):
                if self.board[0][c] == player:
                    queue.append((0, c))
                    visited.add((0, c))
            target_row = self.size - 1
        else:  # Left to Right
            for r in range(self.size):
                if self.board[r][0] == player:
                    queue.append((r, 0))
                    visited.add((r, 0))
            target_col = self.size - 1

        while queue:
            r, c = queue.popleft()
            if player == 1 and r == target_row: return True
            if player == 2 and c == target_col: return True

            for nr, nc in self.get_neighbors(r, c):
                if self.board[nr][nc] == player and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return False


# --- AI AGENT CLASSES ---
class Agent:
    def get_move(self, board, player_color):
        raise NotImplementedError


class HumanAgent(Agent):
    def get_move(self, board, player_color):
        return None  # Handled by UI


class RandomAgent(Agent):
    def get_move(self, board, player_color):
        empty_cells = []
        for r in range(board.size):
            for c in range(board.size):
                if board.board[r][c] == 0:
                    empty_cells.append((r, c))
        if not empty_cells:
            return None
        return random.choice(empty_cells)


class DijkstraAgent(Agent):
    def get_shortest_path_length(self, board, player):
        size = board.size
        pq = []
        visited = set()

        if player == 1:
            for c in range(size):
                state = board.board[0][c]
                if state == player:
                    cost = 0
                elif state == 0:
                    cost = 1
                else:
                    continue
                heapq.heappush(pq, (cost, 0, c))
        else:
            for r in range(size):
                state = board.board[r][0]
                if state == player:
                    cost = 0
                elif state == 0:
                    cost = 1
                else:
                    continue
                heapq.heappush(pq, (cost, r, 0))

        while pq:
            cost, r, c = heapq.heappop(pq)
            if (r, c) in visited: continue
            visited.add((r, c))

            if player == 1 and r == size - 1: return cost
            if player == 2 and c == size - 1: return cost

            for nr, nc in board.get_neighbors(r, c):
                if (nr, nc) in visited: continue
                state = board.board[nr][nc]
                weight = float('inf')
                if state == player:
                    weight = 0
                elif state == 0:
                    weight = 1
                if weight != float('inf'):
                    heapq.heappush(pq, (cost + weight, nr, nc))

        return float('inf')

    def get_move(self, board, player):
        best_move = None
        best_score = -float('inf')
        opponent = 3 - player

        empty_cells = []
        for r in range(board.size):
            for c in range(board.size):
                if board.board[r][c] == 0:
                    empty_cells.append((r, c))

        center = board.size // 2
        if board.board[center][center] == 0:
            return (center, center)

        for r, c in empty_cells:
            board.board[r][c] = player
            my_path = self.get_shortest_path_length(board, player)
            op_path = self.get_shortest_path_length(board, opponent)
            board.board[r][c] = 0

            if my_path == float('inf'):
                score = -float('inf')
            else:
                score = op_path - my_path

            if score > best_score:
                best_score = score
                best_move = (r, c)

        return best_move


class MCTSAgent(Agent):
    def __init__(self, iterations=1000):
        self.iterations = iterations

    def get_move(self, board, player_color):
        root = MCTSNode()
        root.untried_moves = self.get_legal_moves(board)

        for _ in range(self.iterations):
            node = root
            simulation_board = copy.deepcopy(board)

            while node.untried_moves == [] and node.children:
                node = node.uct_select_child()
                simulation_board.place_stone(node.move[0], node.move[1])

            if node.untried_moves:
                m = node.untried_moves.pop()
                simulation_board.place_stone(m[0], m[1])
                new_node = MCTSNode(parent=node, move=m)
                new_node.untried_moves = self.get_legal_moves(simulation_board)
                node.children.append(new_node)
                node = new_node

            temp_turn = simulation_board.turn
            while simulation_board.winner is None:
                moves = self.get_legal_moves(simulation_board)
                if not moves: break
                rm = random.choice(moves)
                simulation_board.place_stone(rm[0], rm[1])

            won = (simulation_board.winner == player_color)
            while node is not None:
                node.visits += 1
                if won:
                    node.wins += 1
                node = node.parent

        if not root.children:
            return random.choice(self.get_legal_moves(board))

        best_child = sorted(root.children, key=lambda c: c.visits)[-1]
        return best_child.move

    def get_legal_moves(self, board):
        moves = []
        for r in range(board.size):
            for c in range(board.size):
                if board.board[r][c] == 0:
                    moves.append((r, c))
        return moves


class MCTSNode:
    def __init__(self, parent=None, move=None):
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = None

    def uct_select_child(self, exploration_weight=1.41):
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            exploit = child.wins / child.visits
            explore = math.sqrt(math.log(self.visits) / child.visits)
            score = exploit + exploration_weight * explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child


# --- MENU SYSTEM ---
class GameMenu:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.SysFont('Arial', 20)
        self.title_font = pygame.font.SysFont('Arial', 32, bold=True)
        self.small_font = pygame.font.SysFont('Arial', 16)

        # Game settings
        self.game_mode = "Human vs AI"  # "Human vs AI" or "AI vs AI"
        self.red_agent = "Human"  # "Human", "Random", "Dijkstra", "MCTS"
        self.blue_agent = "MCTS"
        self.red_iterations = 1000
        self.blue_iterations = 1000

        # UI elements
        self.buttons = {}
        self.sliders = {}
        self.setup_ui()

    def setup_ui(self):
        # Game mode buttons
        self.buttons['human_vs_ai'] = pygame.Rect(50, 100, 200, 40)
        self.buttons['ai_vs_ai'] = pygame.Rect(270, 100, 200, 40)

        # Red player agent buttons
        self.buttons['red_human'] = pygame.Rect(50, 200, 100, 35)
        self.buttons['red_random'] = pygame.Rect(160, 200, 100, 35)
        self.buttons['red_dijkstra'] = pygame.Rect(270, 200, 100, 35)
        self.buttons['red_mcts'] = pygame.Rect(380, 200, 100, 35)

        # Blue player agent buttons
        self.buttons['blue_human'] = pygame.Rect(50, 300, 100, 35)
        self.buttons['blue_random'] = pygame.Rect(160, 300, 100, 35)
        self.buttons['blue_dijkstra'] = pygame.Rect(270, 300, 100, 35)
        self.buttons['blue_mcts'] = pygame.Rect(380, 300, 100, 35)

        # Iteration sliders
        self.sliders['red_iter'] = {
            'rect': pygame.Rect(50, 400, 400, 20),
            'handle': pygame.Rect(50, 395, 15, 30),
            'min': 100,
            'max': 10000,
            'value': 1000
        }

        self.sliders['blue_iter'] = {
            'rect': pygame.Rect(50, 480, 400, 20),
            'handle': pygame.Rect(50, 475, 15, 30),
            'min': 100,
            'max': 10000,
            'value': 1000
        }

        # Start button
        self.buttons['start'] = pygame.Rect(300, 530, 200, 50)

    def draw_button(self, name, rect, text, selected=False):
        mouse_pos = pygame.mouse.get_pos()
        hover = rect.collidepoint(mouse_pos)

        if selected:
            color = BUTTON_SELECTED
        elif hover:
            color = BUTTON_HOVER
        else:
            color = BUTTON_COLOR

        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, TEXT_COLOR, rect, 2)

        text_surf = self.font.render(text, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=rect.center)
        self.screen.blit(text_surf, text_rect)

    def draw_slider(self, name, slider_data, label):
        rect = slider_data['rect']
        handle = slider_data['handle']
        value = slider_data['value']

        # Draw slider track
        pygame.draw.rect(self.screen, BUTTON_COLOR, rect)
        pygame.draw.rect(self.screen, TEXT_COLOR, rect, 2)

        # Draw handle
        pygame.draw.rect(self.screen, BUTTON_SELECTED, handle)
        pygame.draw.rect(self.screen, TEXT_COLOR, handle, 2)

        # Draw label and value
        label_surf = self.font.render(label, True, TEXT_COLOR)
        self.screen.blit(label_surf, (rect.x, rect.y - 30))

        value_surf = self.small_font.render(f"Iterations: {value}", True, TEXT_COLOR)
        self.screen.blit(value_surf, (rect.x + rect.width + 10, rect.y))

    def update_slider_handle(self, slider_name):
        slider = self.sliders[slider_name]
        value_range = slider['max'] - slider['min']
        position = (slider['value'] - slider['min']) / value_range
        slider['handle'].x = slider['rect'].x + int(position * slider['rect'].width) - 7

    def handle_slider_drag(self, slider_name, mouse_x):
        slider = self.sliders[slider_name]
        rect = slider['rect']

        # Clamp mouse position to slider bounds
        relative_x = max(0, min(mouse_x - rect.x, rect.width))
        position = relative_x / rect.width

        # Calculate value
        value_range = slider['max'] - slider['min']
        value = int(slider['min'] + position * value_range)
        value = (value // 100) * 100  # Round to nearest 100

        slider['value'] = max(slider['min'], min(value, slider['max']))
        self.update_slider_handle(slider_name)

    def draw(self):
        self.screen.fill(BG_COLOR)

        # Title
        title = self.title_font.render("HEX GAME SETUP", True, TEXT_COLOR)
        self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 30))

        # Game Mode Section
        mode_label = self.font.render("Game Mode:", True, TEXT_COLOR)
        self.screen.blit(mode_label, (50, 70))

        self.draw_button('human_vs_ai', self.buttons['human_vs_ai'],
                         "Human vs AI", self.game_mode == "Human vs AI")
        self.draw_button('ai_vs_ai', self.buttons['ai_vs_ai'],
                         "AI vs AI", self.game_mode == "AI vs AI")

        # Red Player Section
        red_label = self.font.render("Red Player (Top-Bottom):", True, P1_COLOR)
        self.screen.blit(red_label, (50, 165))

        self.draw_button('red_human', self.buttons['red_human'],
                         "Human", self.red_agent == "Human")
        self.draw_button('red_random', self.buttons['red_random'],
                         "Random", self.red_agent == "Random")
        self.draw_button('red_dijkstra', self.buttons['red_dijkstra'],
                         "Dijkstra", self.red_agent == "Dijkstra")
        self.draw_button('red_mcts', self.buttons['red_mcts'],
                         "MCTS", self.red_agent == "MCTS")

        # Blue Player Section
        blue_label = self.font.render("Blue Player (Left-Right):", True, P2_COLOR)
        self.screen.blit(blue_label, (50, 265))

        self.draw_button('blue_human', self.buttons['blue_human'],
                         "Human", self.blue_agent == "Human")
        self.draw_button('blue_random', self.buttons['blue_random'],
                         "Random", self.blue_agent == "Random")
        self.draw_button('blue_dijkstra', self.buttons['blue_dijkstra'],
                         "Dijkstra", self.blue_agent == "Dijkstra")
        self.draw_button('blue_mcts', self.buttons['blue_mcts'],
                         "MCTS", self.blue_agent == "MCTS")

        # MCTS Iteration Sliders
        if self.red_agent == "MCTS":
            self.draw_slider('red_iter', self.sliders['red_iter'],
                             "Red MCTS Iterations:")
            self.red_iterations = self.sliders['red_iter']['value']

        if self.blue_agent == "MCTS":
            self.draw_slider('blue_iter', self.sliders['blue_iter'],
                             "Blue MCTS Iterations:")
            self.blue_iterations = self.sliders['blue_iter']['value']

        # Start button
        self.draw_button('start', self.buttons['start'], "START GAME")

        pygame.display.flip()

    def handle_click(self, pos):
        # Game mode
        if self.buttons['human_vs_ai'].collidepoint(pos):
            self.game_mode = "Human vs AI"
            self.red_agent = "Human"
            return False
        elif self.buttons['ai_vs_ai'].collidepoint(pos):
            self.game_mode = "AI vs AI"
            if self.red_agent == "Human":
                self.red_agent = "Dijkstra"
            if self.blue_agent == "Human":
                self.blue_agent = "MCTS"
            return False

        # Red agent
        if self.buttons['red_human'].collidepoint(pos) and self.game_mode == "Human vs AI":
            self.red_agent = "Human"
        elif self.buttons['red_random'].collidepoint(pos):
            self.red_agent = "Random"
        elif self.buttons['red_dijkstra'].collidepoint(pos):
            self.red_agent = "Dijkstra"
        elif self.buttons['red_mcts'].collidepoint(pos):
            self.red_agent = "MCTS"

        # Blue agent
        if self.buttons['blue_human'].collidepoint(pos) and self.game_mode == "Human vs AI":
            self.blue_agent = "Human"
        elif self.buttons['blue_random'].collidepoint(pos):
            self.blue_agent = "Random"
        elif self.buttons['blue_dijkstra'].collidepoint(pos):
            self.blue_agent = "Dijkstra"
        elif self.buttons['blue_mcts'].collidepoint(pos):
            self.blue_agent = "MCTS"

        # Start button
        if self.buttons['start'].collidepoint(pos):
            return True

        return False

    def run(self):
        clock = pygame.time.Clock()
        running = True
        dragging_slider = None

        # Initialize slider handles
        self.update_slider_handle('red_iter')
        self.update_slider_handle('blue_iter')

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.handle_click(event.pos):
                        return self.get_config()

                    # Check slider clicks
                    if self.red_agent == "MCTS" and self.sliders['red_iter']['handle'].collidepoint(event.pos):
                        dragging_slider = 'red_iter'
                    elif self.blue_agent == "MCTS" and self.sliders['blue_iter']['handle'].collidepoint(event.pos):
                        dragging_slider = 'blue_iter'
                    elif self.red_agent == "MCTS" and self.sliders['red_iter']['rect'].collidepoint(event.pos):
                        self.handle_slider_drag('red_iter', event.pos[0])
                    elif self.blue_agent == "MCTS" and self.sliders['blue_iter']['rect'].collidepoint(event.pos):
                        self.handle_slider_drag('blue_iter', event.pos[0])

                elif event.type == pygame.MOUSEBUTTONUP:
                    dragging_slider = None

                elif event.type == pygame.MOUSEMOTION:
                    if dragging_slider:
                        self.handle_slider_drag(dragging_slider, event.pos[0])

            self.draw()
            clock.tick(30)

        return None

    def get_config(self):
        return {
            'game_mode': self.game_mode,
            'red_agent': self.red_agent,
            'blue_agent': self.blue_agent,
            'red_iterations': self.red_iterations,
            'blue_iterations': self.blue_iterations
        }


# --- UI CLASS ---
class HexUI:
    def __init__(self, game, config):
        self.game = game
        self.config = config
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Hex Game")
        self.font = pygame.font.SysFont('Arial', 24, bold=True)
        self.small_font = pygame.font.SysFont('Arial', 18)

        self.start_x = WINDOW_WIDTH // 4 + 50
        self.start_y = WINDOW_HEIGHT // 4 + 30

        # Create agents
        self.agents = {}
        self.agents[1] = self.create_agent(config['red_agent'], config['red_iterations'])
        self.agents[2] = self.create_agent(config['blue_agent'], config['blue_iterations'])

    def create_agent(self, agent_type, iterations):
        if agent_type == "Human":
            return HumanAgent()
        elif agent_type == "Random":
            return RandomAgent()
        elif agent_type == "Dijkstra":
            return DijkstraAgent()
        elif agent_type == "MCTS":
            return MCTSAgent(iterations=iterations)
        return None

    def hex_to_pixel(self, row, col):
        x = self.start_x + (HEX_RADIUS * math.sqrt(3) * (col + 0.5 * row))
        y = self.start_y + (HEX_RADIUS * 1.5 * row)
        return x, y

    def pixel_to_hex(self, x, y):
        row = (y - self.start_y) / (HEX_RADIUS * 1.5)
        col = ((x - self.start_x) / (HEX_RADIUS * math.sqrt(3))) - 0.5 * row
        return int(round(row)), int(round(col))

    def draw_hexagon(self, surface, center_x, center_y, fill_color, row, col):
        points = []
        for i in range(6):
            angle_deg = 60 * i + 30
            angle_rad = math.pi / 180 * angle_deg
            px = center_x + HEX_RADIUS * math.cos(angle_rad)
            py = center_y + HEX_RADIUS * math.sin(angle_rad)
            points.append((px, py))

        pygame.draw.polygon(surface, fill_color, points)

        edge_colors = [BORDER_COLOR] * 6
        size = self.game.size

        if row == 0:
            edge_colors[3] = RED_BORDER
            edge_colors[4] = RED_BORDER
        if row == size - 1:
            edge_colors[0] = RED_BORDER
            edge_colors[1] = RED_BORDER
        if col == 0:
            edge_colors[1] = BLUE_BORDER
            edge_colors[2] = BLUE_BORDER
        if col == size - 1:
            edge_colors[4] = BLUE_BORDER
            edge_colors[5] = BLUE_BORDER

        for i in range(6):
            start = points[i]
            end = points[(i + 1) % 6]
            pygame.draw.line(surface, edge_colors[i], start, end, 4)

    def draw(self):
        self.screen.fill(BG_COLOR)

        # Status Text
        if self.game.winner:
            winner_name = "RED" if self.game.winner == 1 else "BLUE"
            agent_name = self.config['red_agent'] if self.game.winner == 1 else self.config['blue_agent']
            status = f"WINNER: {winner_name} ({agent_name})!"
            color = P1_COLOR if self.game.winner == 1 else P2_COLOR
        else:
            current_agent = self.config['red_agent'] if self.game.turn == 1 else self.config['blue_agent']
            current_color = "RED" if self.game.turn == 1 else "BLUE"
            status = f"Turn: {current_color} ({current_agent})"
            if isinstance(self.agents[self.game.turn], HumanAgent):
                status += " - Your move!"
            else:
                status += " - Thinking..."
            color = TEXT_COLOR

        text_surf = self.font.render(status, True, color)
        self.screen.blit(text_surf, (20, 20))

        # Draw hexagons
        for r in range(self.game.size):
            for c in range(self.game.size):
                x, y = self.hex_to_pixel(r, c)
                fill = EMPTY_COLOR
                if self.game.board[r][c] == 1:
                    fill = P1_COLOR
                elif self.game.board[r][c] == 2:
                    fill = P2_COLOR
                self.draw_hexagon(self.screen, x, y, fill, r, c)

        # Labels
        self.screen.blit(self.small_font.render("RED: Top to Bottom", True, RED_BORDER), (20, 60))
        self.screen.blit(self.small_font.render("BLUE: Left to Right", True, BLUE_BORDER), (20, 85))

        pygame.display.flip()

    def run(self):
        running = True
        clock = pygame.time.Clock()

        while running:
            current_agent = self.agents[self.game.turn]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if isinstance(current_agent, HumanAgent) and not self.game.winner:
                        mx, my = pygame.mouse.get_pos()
                        r, c = self.pixel_to_hex(mx, my)
                        if 0 <= r < self.game.size and 0 <= c < self.game.size:
                            self.game.place_stone(r, c)

            # AI turn
            if not isinstance(current_agent, HumanAgent) and not self.game.winner:
                self.draw()
                pygame.time.wait(100)
                move = current_agent.get_move(self.game, self.game.turn)
                if move:
                    self.game.place_stone(move[0], move[1])

            self.draw()
            clock.tick(30)

        pygame.quit()


# --- MAIN ---
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

    menu = GameMenu(screen)
    config = menu.run()

    if config:
        game = HexBoard(BOARD_SIZE)
        ui = HexUI(game, config)
        ui.run()

    pygame.quit()