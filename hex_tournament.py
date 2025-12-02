import time
from hex_ui import HexBoard, RandomAgent, DijkstraAgent, MCTSAgent


def play_match(p1_agent, p2_agent, size=9):
    """
    Runs a single game between two agents.
    Returns: The winner (1 or 2)
    """
    board = HexBoard(size)

    while board.winner is None:
        # Determine current player and agent
        current_player = board.turn
        agent = p1_agent if current_player == 1 else p2_agent

        # Get move
        move = agent.get_move(board, current_player)

        # Handle resignation or error
        if move is None:
            print(f"Player {current_player} has no moves left!")
            return 3 - current_player  # The other player wins

        # Apply move
        board.place_stone(move[0], move[1])

    return board.winner


def run_tournament(agent1_cls, agent1_name, agent2_cls, agent2_name, num_games=100, board_size=9):
    print(f"--- STARTING TOURNAMENT: {agent1_name} vs {agent2_name} ---")
    print(f"Board Size: {board_size}x{board_size} | Games: {num_games}")

    p1_wins = 0
    p2_wins = 0
    start_time = time.time()

    for i in range(num_games):
        # Instantiate fresh agents for each game (important for MCTS tree reset)
        # You can adjust MCTS iterations here, e.g., MCTSAgent(iterations=500)
        p1 = agent1_cls()
        p2 = agent2_cls()

        # Swap colors every game to ensure fairness?
        # For this function, let's keep Agent 1 as Red (P1) always to measure First Player Advantage.
        winner = play_match(p1, p2, board_size)

        if winner == 1:
            p1_wins += 1
        else:
            p2_wins += 1

        # Progress bar (optional)
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_games} games...")

    total_time = time.time() - start_time

    print("\n--- RESULTS ---")
    print(f"{agent1_name} (Red/First) Wins: {p1_wins} ({p1_wins / num_games * 100:.1f}%)")
    print(f"{agent2_name} (Blue/Second) Wins: {p2_wins} ({p2_wins / num_games * 100:.1f}%)")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Avg Time per Game: {total_time / num_games:.2f}s")
    print("------------------------------------------------\n")


if __name__ == "__main__":
    # EXPERIMENT 1: BASELINE
    # Check if Random vs Random is roughly 50/50 (or shows 1st player advantage)
    run_tournament(RandomAgent, "Random", RandomAgent, "Random", num_games=100)

    # EXPERIMENT 2: INTELLIGENCE CHECK
    # Dijkstra should completely destroy Random
    run_tournament(DijkstraAgent, "Dijkstra", RandomAgent, "Random", num_games=50)

    # EXPERIMENT 3: THE MAIN EVENT
    # MCTS vs Dijkstra.
    # Note: MCTS is slower, so we run fewer games.
    # To pass arguments like iterations, you might need a lambda or wrapper class,
    # but for now assume default iterations in your class or modify the loop above.
    print("Running Heavyweight Match (MCTS vs Dijkstra)...")


    # Custom wrapper to increase MCTS power
    class StrongMCTS(MCTSAgent):
        def __init__(self):
            super().__init__(iterations=1000)


    run_tournament(StrongMCTS, "MCTS (1k)", DijkstraAgent, "Dijkstra", num_games=20)
