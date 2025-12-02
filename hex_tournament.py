import time
import matplotlib.pyplot as plt
from hex_ui import HexBoard, RandomAgent, DijkstraAgent, MCTSAgent


def play_match(p1_agent, p2_agent, size=9):
    """
    Runs a single game between two agents.
    Returns: The winner (1 or 2)
    """
    board = HexBoard(size)

    while board.winner is None:
        current_player = board.turn
        agent = p1_agent if current_player == 1 else p2_agent

        move = agent.get_move(board, current_player)

        if move is None:
            return 3 - current_player  # Current player has no moves left

        board.place_stone(move[0], move[1])

    return board.winner


def plot_results(p1_name, p1_wins, p2_name, p2_wins):
    """
    Creates a window with a Pie Chart and a Bar Chart of the results.
    """
    labels = [f'{p1_name} (P1)', f'{p2_name} (P2)']
    wins = [p1_wins, p2_wins]
    colors = ['#ff6666', '#6666ff']  # Red and Blue

    # Create a figure with 2 subplots (side by side)
    plt.figure(figsize=(12, 6))

    # 1. Pie Chart
    plt.subplot(1, 2, 1)
    plt.pie(wins, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, shadow=True)
    plt.title(f'Win Rate Distribution\n({sum(wins)} Games)', loc="left")

    # 2. Bar Chart
    plt.subplot(1, 2, 2)
    bars = plt.bar(labels, wins, color=colors)
    plt.title('Total Games Won')
    plt.ylabel('Wins')
    plt.ylim(0, sum(wins) + 1)  # Set y-axis limit slightly above total games

    # Add numbers on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

    plt.suptitle(f'Tournament Results: {p1_name} vs {p2_name}', fontsize=16)
    plt.tight_layout()
    plt.show()


def run_tournament(agent1_cls, agent1_name, agent2_cls, agent2_name, num_games=100, board_size=9, visualize=True):
    print(f"--- STARTING TOURNAMENT: {agent1_name} vs {agent2_name} ---")
    print(f"Board Size: {board_size}x{board_size} | Games: {num_games}")

    p1_wins = 0
    p2_wins = 0
    start_time = time.time()

    for i in range(num_games):
        p1 = agent1_cls()
        p2 = agent2_cls()

        winner = play_match(p1, p2, board_size)

        if winner == 1:
            p1_wins += 1
        else:
            p2_wins += 1

        # Simple progress indicator
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{num_games} games finished...")

    total_time = time.time() - start_time

    print("\n--- RESULTS ---")
    print(f"{agent1_name} (Red): {p1_wins} wins")
    print(f"{agent2_name} (Blue): {p2_wins} wins")
    print(f"Total Time: {total_time:.2f}s")
    print("------------------------------------------------\n")

    if visualize:
        plot_results(agent1_name, p1_wins, agent2_name, p2_wins)


if __name__ == "__main__":
    MCTS_ITERATIONS = 1000

    # EXPERIMENT 1: BASELINE
    # Check if Random vs Random is roughly 50/50 (or shows 1st player advantage)
    # This usually shows a slight advantage for Player 1 (Red)
    run_tournament(RandomAgent, "Random", RandomAgent, "Random", num_games=50, visualize=True)

    # EXPERIMENT 2: INTELLIGENCE CHECK
    # Dijkstra should completely destroy Random
    run_tournament(DijkstraAgent, "Dijkstra", RandomAgent, "Random", num_games=20, visualize=True)

    # EXPERIMENT 3: THE MAIN EVENT
    # MCTS vs Dijkstra.
    # Note: MCTS is slower, so we run fewer games.

    # Custom MCTS wrapper for stronger AI
    class StrongMCTS(MCTSAgent):
        def __init__(self):
            super().__init__(iterations=MCTS_ITERATIONS)


    print("Running Heavyweight Match (MCTS vs Dijkstra)...")
    run_tournament(StrongMCTS, f"MCTS (it.: {MCTS_ITERATIONS})", DijkstraAgent, "Dijkstra", num_games=10, visualize=True)
