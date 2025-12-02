import pygame
import math
from collections import deque
import heapq
import random

# --- CONFIGURATION ---
BOARD_SIZE = 9
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
HEX_RADIUS = 25
BG_COLOR = (30, 30, 30)
EMPTY_COLOR = (200, 200, 200)
P1_COLOR = (255, 50, 50)  # Red (Connects Top-Bottom)
P2_COLOR = (50, 50, 255)  # Blue (Connects Left-Right)
TEXT_COLOR = (255, 255, 255)


class Agent:
    def get_move(self, board, player_color):
        """
        Input: Board object, player_color (1 or 2)
        Output: (row, col)
        """
        raise NotImplementedError


class RandomAgent(Agent):
    def get_move(self, board, player_color):
        empty_cells = []
        for r in range(board.size):
            for c in range(board.size):
                if board.board[r][c] == 0:
                    empty_cells.append((r, c))
        return random.choice(empty_cells) if empty_cells else None


class DijkstraAgent(Agent):
    def get_shortest_path_length(self, board, player):
        """
        Runs Dijkstra to find the cost of the path for 'player'.
        Player 1: Top -> Bottom
        Player 2: Left -> Right
        """
        size = board.size
        pq = []  # Priority Queue: (cost, r, c)
        visited = set()

        # Initialize Virtual Start Nodes
        # If P1: Add all cells in Top Row (Row 0)
        # If P2: Add all cells in Left Col (Col 0)
        if player == 1:
            for c in range(size):
                state = board.board[0][c]
                if state == player:
                    cost = 0
                elif state == 0:
                    cost = 1
                else:
                    continue  # Opponent blocked
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

            # Check if we reached the Target Edge
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

        return float('inf')  # No path possible

    def get_move(self, board, player):
        """
        Evaluates every empty cell.
        Metric: Minimize (My_Path) AND Maximize (Opponent_Path).
        """
        best_move = None
        best_score = -float('inf')
        opponent = 3 - player

        # Get all empty cells
        empty_cells = []
        for r in range(board.size):
            for c in range(board.size):
                if board.board[r][c] == 0:
                    empty_cells.append((r, c))

        # Simple optimization: If it's the first move, play center (strategy standard)
        if len(empty_cells) == board.size * board.size:
            return (board.size // 2, board.size // 2)

        # Evaluate each candidate move
        for r, c in empty_cells:
            # Simulate Move
            board.board[r][c] = player

            # Calculate heuristics
            my_path = self.get_shortest_path_length(board, player)
            op_path = self.get_shortest_path_length(board, opponent)

            # Undo Move
            board.board[r][c] = 0

            # Scoring Function:
            # We want 'my_path' small, 'op_path' large.
            # If my_path is inf (blocked), score is terrible.
            if my_path == float('inf'):
                score = -float('inf')
            else:
                score = op_path - my_path

            if score > best_score:
                best_score = score
                best_move = (r, c)

        return best_move

class HexBoard:
    """
    Manages the game state and logic.
    Board is represented as a rhombus grid (2D array).
    """

    def __init__(self, size):
        self.size = size
        # 0: Empty, 1: Player 1 (Red), 2: Player 2 (Blue)
        self.board = [[0 for _ in range(size)] for _ in range(size)]
        self.turn = 1  # Player 1 starts
        self.winner = None

    def is_valid_move(self, row, col):
        return 0 <= row < self.size and 0 <= col < self.size and self.board[row][col] == 0

    def place_stone(self, row, col):
        if self.is_valid_move(row, col) and self.winner is None:
            self.board[row][col] = self.turn
            if self.check_win(self.turn):
                self.winner = self.turn
            else:
                self.turn = 3 - self.turn  # Toggle between 1 and 2
            return True
        return False

    def get_neighbors(self, r, c):
        """
        Returns valid neighbors for a cell (r, c) in a Hex grid.
        Hex neighbors in a rhombus grid representation:
        (r-1, c), (r-1, c+1), (r, c-1), (r, c+1), (r+1, c-1), (r+1, c)
        """
        directions = [
            (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0)
        ]
        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                neighbors.append((nr, nc))
        return neighbors

    def check_win(self, player):
        """
        BFS to check if 'player' has connected their sides.
        Player 1 (Red): Top (row 0) -> Bottom (row size-1)
        Player 2 (Blue): Left (col 0) -> Right (col size-1)
        """
        queue = deque()
        visited = set()

        # Initialize BFS with all stones on the starting edge
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

            # Check win condition based on player
            if player == 1 and r == target_row: return True
            if player == 2 and c == target_col: return True

            for nr, nc in self.get_neighbors(r, c):
                if self.board[nr][nc] == player and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        return False


class HexUI:
    """
    Handles the Graphical User Interface using Pygame.
    """

    def __init__(self, game):
        self.game = game
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Hex Game (9x9) - P1: Red (Top-Down), P2: Blue (Left-Right)")
        self.font = pygame.font.SysFont('Arial', 24)

        # Calculate board offset to center it
        board_pixel_width = 3 / 2 * HEX_RADIUS * game.size
        self.start_x = WINDOW_WIDTH // 4
        self.start_y = WINDOW_HEIGHT // 4

    def hex_to_pixel(self, row, col):
        """Converts grid coordinates to pixel (x, y) for drawing."""
        # Uses standard "pointy-topped" hex math or skewed rect math
        # For Rhombus board:
        # x = size * 3/2 * (col * 1 + row * 0.5) -- Wait, let's use standard layout
        # x = radius * sqrt(3) * (col + 0.5 * row)
        # y = radius * 3/2 * row
        x = self.start_x + (HEX_RADIUS * math.sqrt(3) * (col + 0.5 * row))
        y = self.start_y + (HEX_RADIUS * 3 / 2 * row)
        return x, y

    def pixel_to_hex(self, x, y):
        """Approximate conversion from mouse click to grid coordinates."""
        # This is a simple approximation; for precise clicking a better algo is needed
        # but this works well for clicking near centers.
        row = (y - self.start_y) / (HEX_RADIUS * 3 / 2)
        col = ((x - self.start_x) / (HEX_RADIUS * math.sqrt(3))) - 0.5 * row
        return int(round(row)), int(round(col))

    def draw_hexagon(self, surface, color, x, y):
        points = []
        for i in range(6):
            angle_deg = 60 * i + 30
            angle_rad = math.pi / 180 * angle_deg
            px = x + HEX_RADIUS * math.cos(angle_rad)
            py = y + HEX_RADIUS * math.sin(angle_rad)
            points.append((px, py))

        pygame.draw.polygon(surface, color, points)
        pygame.draw.polygon(surface, (50, 50, 50), points, 2)  # Border

    def draw(self):
        self.screen.fill(BG_COLOR)

        # Draw Status
        status_text = f"Turn: {'Red (P1)' if self.game.turn == 1 else 'Blue (P2)'}"
        if self.game.winner:
            status_text = f"WINNER: {'Red' if self.game.winner == 1 else 'Blue'}!"

        text_surf = self.font.render(status_text, True, TEXT_COLOR)
        self.screen.blit(text_surf, (20, 20))

        # Draw Board
        for r in range(self.game.size):
            for c in range(self.game.size):
                x, y = self.hex_to_pixel(r, c)
                color = EMPTY_COLOR
                if self.game.board[r][c] == 1:
                    color = P1_COLOR
                elif self.game.board[r][c] == 2:
                    color = P2_COLOR

                self.draw_hexagon(self.screen, color, x, y)

        pygame.display.flip()

    def run(self):
        running = True
        clock = pygame.time.Clock()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if not self.game.winner:
                        mx, my = pygame.mouse.get_pos()
                        r, c = self.pixel_to_hex(mx, my)
                        self.game.place_stone(r, c)

            self.draw()
            clock.tick(30)

        pygame.quit()


if __name__ == "__main__":
    game = HexBoard(BOARD_SIZE)
    ui = HexUI(game)
    ui.run()
