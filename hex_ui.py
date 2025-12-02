import pygame
import math
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
P1_COLOR = (255, 50, 50)  # Red (You)
P2_COLOR = (50, 50, 255)  # Blue (AI)
TEXT_COLOR = (255, 255, 255)
BORDER_COLOR = (50, 50, 50)
RED_BORDER = (220, 20, 20)
BLUE_BORDER = (20, 20, 255)


# --- LOGIC CLASS ---
class HexBoard:
    def __init__(self, size):
        self.size = size
        # 0: Empty, 1: Player 1 (Red), 2: Player 2 (Blue)
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

class RandomAgent(Agent):
    """
    Baseline AI: Selects a move completely at random from available empty spots.
    """

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
    """
    Greedy AI that minimizes its own shortest path while maximizing the opponent's.
    """

    def get_shortest_path_length(self, board, player):
        size = board.size
        pq = []
        visited = set()

        # Initialize Virtual Start Nodes
        if player == 1:  # Top -> Bottom
            for c in range(size):
                state = board.board[0][c]
                if state == player:
                    cost = 0
                elif state == 0:
                    cost = 1
                else:
                    continue
                heapq.heappush(pq, (cost, 0, c))
        else:  # Left -> Right
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

            # Check if reached target edge
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

        # Optimization: First move center if available
        center = board.size // 2
        if board.board[center][center] == 0:
            return (center, center)

        # Evaluate moves
        for r, c in empty_cells:
            # Simulate
            board.board[r][c] = player

            my_path = self.get_shortest_path_length(board, player)
            op_path = self.get_shortest_path_length(board, opponent)

            # Undo
            board.board[r][c] = 0

            if my_path == float('inf'):
                score = -float('inf')
            else:
                # Heuristic: Difference in path lengths
                # Higher is better (I am closer than opponent)
                score = op_path - my_path

            if score > best_score:
                best_score = score
                best_move = (r, c)

        return best_move


# --- UI CLASS ---
class HexUI:
    def __init__(self, game):
        self.game = game
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Hex AI Challenge - You are RED (Top-Bottom)")
        self.font = pygame.font.SysFont('Arial', 24, bold=True)

        self.start_x = WINDOW_WIDTH // 4 + 50
        self.start_y = WINDOW_HEIGHT // 4 + 30

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

        # Fill
        pygame.draw.polygon(surface, fill_color, points)

        # Determine border colors per edge
        edge_colors = [BORDER_COLOR] * 6

        size = self.game.size

        # Top row → top edge red
        if row == 0:
            edge_colors[3] = RED_BORDER  # top edge
            edge_colors[4] = RED_BORDER  # top edge

        # Bottom row → bottom edge red
        if row == size - 1:
            edge_colors[0] = RED_BORDER  # bottom edge
            edge_colors[1] = RED_BORDER  # bottom edge

        # Left column → left edges blue
        if col == 0:
            edge_colors[1] = BLUE_BORDER  # southwest
            edge_colors[2] = BLUE_BORDER  # northwest

        # Right column → right edges blue
        if col == size - 1:
            edge_colors[4] = BLUE_BORDER  # northeast
            edge_colors[5] = BLUE_BORDER  # southeast

        # Draw each edge with its color
        for i in range(6):
            start = points[i]
            end = points[(i + 1) % 6]
            pygame.draw.line(surface, edge_colors[i], start, end, 4)

    def draw(self):
        self.screen.fill(BG_COLOR)

        # Status Text
        if self.game.winner:
            status = f"WINNER: {'RED (You)' if self.game.winner == 1 else 'BLUE (AI)'}!"
            color = P1_COLOR if self.game.winner == 1 else P2_COLOR
        else:
            status = f"Turn: {'RED (You)' if self.game.turn == 1 else 'BLUE (AI Thinking...)'}"
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

        # Optional: Add labels
        label_font = pygame.font.SysFont('Arial', 20)
        self.screen.blit(label_font.render("RED starts at TOP", True, RED_BORDER), (20, 70))
        self.screen.blit(label_font.render("BLUE starts at LEFT", True, BLUE_BORDER), (20, 100))

        pygame.display.flip()

    def run(self):
        running = True
        clock = pygame.time.Clock()
        # --- AGENT SELECTION ---
        # ai_agent = DijkstraAgent()
        ai_agent = RandomAgent()
        ai_player = 2       # Blue
        human_player = 1    # Red

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.game.turn == human_player and not self.game.winner:
                        mx, my = pygame.mouse.get_pos()
                        r, c = self.pixel_to_hex(mx, my)
                        if 0 <= r < self.game.size and 0 <= c < self.game.size:
                            self.game.place_stone(r, c)

            if self.game.turn == ai_player and not self.game.winner:
                self.draw()  # Show "Thinking..."
                pygame.time.wait(300)  # Small delay for visual feedback
                move = ai_agent.get_move(self.game, ai_player)
                if move:
                    self.game.place_stone(move[0], move[1])

            self.draw()
            clock.tick(30)

        pygame.quit()


if __name__ == "__main__":
    game = HexBoard(BOARD_SIZE)
    ui = HexUI(game)
    ui.run()