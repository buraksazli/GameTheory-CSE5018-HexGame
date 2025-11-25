import random
import sys

BOARD_SIZES = [5, 7, 9, 11]
NUM_GAMES = 1000

sys.setrecursionlimit(2000)

class HexGame:
    def __init__(self, size):
        self.size = size
       
        self.board = [[0 for _ in range(size)] for _ in range(size)]
        self.current_player = 1
        self.moves_made = 0
 
        self.neighbors = [
            (-1, 0), (-1, 1),  
            (0, -1), (0, 1),  
            (1, -1), (1, 0)   
        ]

    def is_valid_move(self, row, col):
        return 0 <= row < self.size and 0 <= col < self.size and self.board[row][col] == 0

    def make_move(self, row, col):
        if self.is_valid_move(row, col):
            self.board[row][col] = self.current_player
            self.current_player = 3 - self.current_player 
            self.moves_made += 1
            return True
        return False

    def get_empty_cells(self):
        return [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r][c] == 0]

    def check_winner(self):
       
        visited = set()
        for col in range(self.size):
            if self.board[0][col] == 1: 
                if self.dfs(0, col, 1, visited):
                    return 1
        
        visited = set()
        for row in range(self.size):
            if self.board[row][0] == 2: 
                if self.dfs(row, 0, 2, visited):
                    return 2

        return 0 

    def dfs(self, r, c, player, visited):
       
        if player == 1 and r == self.size - 1: 
            return True
        if player == 2 and c == self.size - 1: 
            return True
        
        visited.add((r, c))
        
        for dr, dc in self.neighbors:
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < self.size and 0 <= nc < self.size:

                if self.board[nr][nc] == player and (nr, nc) not in visited:
                    if self.dfs(nr, nc, player, visited):
                        return True
        return False

def play_random_game(size):
    game = HexGame(size)
    
    while True:
        empty_cells = game.get_empty_cells()
        if not empty_cells:
            break 
            
        move = random.choice(empty_cells)
        game.make_move(move[0], move[1])
        
        winner = game.check_winner()
        if winner != 0:
            return winner
            
    return 0

def run_experiment():
    print("\n--- Preliminary Experiment: Random Hex Simulation ---")
    print(f"{'Board Size':<12} | {'Games':<10} | {'P1 Wins':<10} | {'P2 Wins':<10} | {'P1 Win Rate'}")
    print("-" * 65)
    
    for size in BOARD_SIZES:
        p1_wins = 0
        p2_wins = 0
        
        for _ in range(NUM_GAMES):
            winner = play_random_game(size)
            if winner == 1:
                p1_wins += 1
            elif winner == 2:
                p2_wins += 1
        
        win_rate = (p1_wins / NUM_GAMES) * 100
        print(f"{size}x{size:<10} | {NUM_GAMES:<10} | {p1_wins:<10} | {p2_wins:<10} | %{win_rate:.2f}")

if __name__ == "__main__":
    run_experiment()