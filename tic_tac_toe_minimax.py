import random
import matplotlib.pyplot as plt

X = 'X'
O = 'O'
SPACE = ' '
MAX_LEN = 5 # nguoi choi dat duoc 5X hoac 5O thi se thang

def get_symbol(moved_count: int) -> str:
    return ((moved_count + 1) % 2) * X + (moved_count % 2) * O

def print_state(state: list[list]):
    for row in state:
        print(row)

def input_caro_board():
    n = int(input())
    moved_count = int(input())
    state = [[SPACE for i in range(n)] for i in range(n)]
    outer_region = [[0 for i in range(n)] for i in range(n)]
    empty_positions = set([(i, j) for i in range(n) for j in range(n)])
    
    for i in range(moved_count):
        x = int(input())
        y = int(input())
        state[y][x] = get_symbol(i)
        adjust_outer_region(outer_region, (x, y), 2)
        empty_positions.remove((x, y))
    
    if (moved_count == 0):
        outer_region[n // 2][n // 2] = 1

    return n, moved_count, state, empty_positions, outer_region
    
def check_row_end(state: list[list], last_move: tuple[int, int]) -> bool:
    count = 0
    x1 = max(last_move[0] - (MAX_LEN - 1), 0)
    x2 = min(last_move[0] + (MAX_LEN - 1), len(state) - 1)

    for x in range(x1, x2 + 1):
        if state[last_move[1]][x] == state[last_move[1]][last_move[0]]:
            count += 1
        else:
            count = 0

        if (count >= MAX_LEN):
            return True

    return False

def check_column_end(state: list[list], last_move: tuple[int, int]) -> bool:
    count = 0
    y1 = max(last_move[1] - (MAX_LEN - 1), 0)
    y2 = min(last_move[1] + (MAX_LEN - 1), len(state) - 1)
    
    for y in range(y1, y2 + 1):
        if state[y][last_move[0]] == state[last_move[1]][last_move[0]]:
            count += 1
        else: 
            count = 0

        if (count >= MAX_LEN):
            return True

    return False

def check_diagonal_main_end(state: list[list], last_move: tuple[int, int]) -> bool:
    count = 0
    x1, x2, y1, y2 = 0, 0, 0, 0
    if (last_move[0] <= last_move[1]):
        x1 = max(last_move[0] - (MAX_LEN - 1), 0)
        y1 = last_move[1] - (last_move[0] - x1)
        y2 = min(last_move[1] + (MAX_LEN - 1), len(state) - 1)
        x2 = last_move[0] + (y2 - last_move[1])
    else:
        y1 = max(last_move[1] - (MAX_LEN - 1), 0)
        x1 = last_move[0] - (last_move[1] - y1)
        x2 = min(last_move[0] + (MAX_LEN - 1), len(state) - 1)
        y2 = last_move[1] + (x2 - last_move[0])

    y = y1
    for x in range(x1, x2 + 1):
        if state[y][x] == state[last_move[1]][last_move[0]]:
            count += 1
        else:
            count = 0
        y += 1
        
        if count >= MAX_LEN:
            return True        

    return False

def check_diagonal_secondary(state: list[list], last_move: tuple[int, int]) -> bool:
    count = 0
    x1, x2, y1, y2 = 0, 0, 0, 0
    if (last_move[0] <= len(state) - 1 - last_move[1]):
        x1 = max(last_move[0] - (MAX_LEN - 1), 0)
        y2 = last_move[1] + (last_move[0] - x1)
        y1 = max(last_move[1] - (MAX_LEN - 1), 0)
        x2 = last_move[0] + (last_move[1] - y1)
    else:
        y2 = min(last_move[1] + (MAX_LEN - 1), len(state) - 1)
        x1 = last_move[0] - (y2 - last_move[1])
        x2 = min(last_move[0] + (MAX_LEN - 1), len(state) - 1)
        y1 = last_move[1] - (x2 - last_move[0])

    y = y2
    for x in range(x1, x2 + 1):
        if state[y][x] == state[last_move[1]][last_move[0]]:
            count += 1
        else:
            count = 0
        y -= 1

        if count >= MAX_LEN:
            return True
        
    return False

def check_game_end(state: list[list], last_move: tuple[int, int]) -> bool:
    return (check_row_end(state, last_move) or
            check_column_end(state, last_move) or
            check_diagonal_main_end(state, last_move) or
            check_diagonal_secondary(state, last_move))  

def adjust_outer_region(outer_region: list[list], move: tuple[int, int], margin: int = 2) -> list:
    last = []
    x1 = max(0, move[0] - margin)
    x2 = min(move[0] + margin, len(outer_region) - 1)
    y1 = max(0, move[1] - margin)
    y2 = min(move[1] + margin, len(outer_region) - 1)
    for x in range(x1, x2 + 1):
        for y in range(y1, y2 + 1):
            if (outer_region[y][x] == 0):
                outer_region[y][x] = 1
                last.append((x, y))
    if (margin == 1) and (len(last) == 0):
        last = adjust_outer_region(outer_region, move, 2)
    return last

def restore_outer_region(outer_region: list[list], last: list):
    for i in last:
        outer_region[i[1]][i[0]] = 0

def minimax(state: list[list], depth: int, moved_count: int, empty_positions: set[tuple[int, int]], outer_region: list[list], last_move: tuple[int, int] = (-1, -1), alpha = -float("inf"), beta = float("inf")) -> int:    
    score = 0
    if (last_move != (-1, -1)):
        game_ended = check_game_end(state, last_move)
        score = game_ended * 100 * (1 - 2*(depth % 2)) 

    if (score == 100):
        return 100
    
    if (score == -100):
        return -100
    
    if (depth >= 2): # dieu chinh do sau
        return 0
    
    if (depth % 2 == 0):
        best = float("inf")
        empty_positions_copy = empty_positions.copy()
        while len(empty_positions_copy) > 0:
            empty_position = empty_positions_copy.pop()
            if (not outer_region[empty_position[1]][empty_position[0]]):
                continue

            empty_positions.remove(empty_position)

            state[empty_position[1]][empty_position[0]] = get_symbol(moved_count)
            last = adjust_outer_region(outer_region, empty_position, 2)
            best = min(best, minimax(state, depth + 1, moved_count + 1, empty_positions, outer_region, empty_position, alpha, beta))
            restore_outer_region(outer_region, last)
            
            empty_positions.add(empty_position)
            state[empty_position[1]][empty_position[0]] = SPACE
            beta = min(beta, best)
            if (alpha >= beta):
                break
        if (best == float("inf")):
            return 0
        return best
    else:
        best = -float("inf")
        empty_positions_copy = empty_positions.copy()
        while len(empty_positions_copy) > 0:
            empty_position = empty_positions_copy.pop()
            if (not outer_region[empty_position[1]][empty_position[0]]):
                continue

            empty_positions.remove(empty_position)

            state[empty_position[1]][empty_position[0]] = get_symbol(moved_count)
            last = adjust_outer_region(outer_region, empty_position, 2)
            best = max(best, minimax(state, depth + 1, moved_count + 1, empty_positions, outer_region, empty_position, alpha, beta))
            restore_outer_region(outer_region, last)

            empty_positions.add(empty_position)
            state[empty_position[1]][empty_position[0]] = SPACE
            alpha = max(alpha, best)
            if (alpha >= beta):
                break
        if (best == -float("inf")):
            return 0
        return best          

def solution(state: list[list], moved_count: int, empty_positions: set[tuple[int, int]], outer_region: list[list]) -> tuple[int, int]:
    best_move = (-1, -1)
    best_val = -float("inf")

    empty_positions_copy = empty_positions.copy()
    while len(empty_positions_copy) > 0:
        empty_position = empty_positions_copy.pop()
        if (not outer_region[empty_position[1]][empty_position[0]]):
            continue
        empty_positions.remove(empty_position)
        state[empty_position[1]][empty_position[0]] = get_symbol(moved_count)

        last = adjust_outer_region(outer_region, empty_position, 2)
        move_val = minimax(state, 0, moved_count + 1, empty_positions, outer_region, empty_position)
        # print(move_val, empty_position)
        restore_outer_region(outer_region, last)
        
        empty_positions.add(empty_position)
        state[empty_position[1]][empty_position[0]] = SPACE
        if (move_val > best_val):
            best_val = move_val
            best_move = empty_position
        if best_val > 0:
            break

    return best_move

def plot_training_results(results):
    x = [i for i in range(len(results["X"]))]
    plt.plot(x, results["X"], label="X Wins")
    plt.plot(x, results["O"], label="O Wins")
    plt.plot(x, results["draw"], label="Draws")
    plt.xlabel("Games")
    plt.ylabel("Wins/Draws")
    plt.title("Training Results")
    plt.legend()
    plt.show()

# hàm tự chơi với random. máy luôn cầm X. Sau đó trả về kết quả người thắng, nếu ko ai thắng thì trả về None
def play_game(state: list[list], moved_count: int, empty_positions: set[tuple[int, int]], outer_region: list[list]) -> tuple[int, int]:
    players = ["X", "O"]
    current_player_idx = 0
    winner = None

    while True:
        current_player = players[current_player_idx]
        
        # Agent chọn hành động

        if current_player == "X":  # Chỉ agent "X" học Q-learning
            action = solution(state, moved_count, empty_positions, outer_region)
        else:
            action = random.choice(list(empty_positions))  # Người chơi "O" chỉ chọn ngẫu nhiên

        state[action[1]][action[0]] = get_symbol(moved_count)
        moved_count += 1
        adjust_outer_region(outer_region, action, 2)
        empty_positions.remove(action)

        # Kiểm tra nếu ai đó thắng
        if check_game_end(state, action):
            winner = current_player
            if current_player == "X":
                break

        # Nếu hòa
        if len(empty_positions) == 0:
            break

        current_player_idx = 1 - current_player_idx  

    return winner

def play(n, num_games) -> dict:
    results = {"X": 0, "O": 0, "draw": 0}

    for _ in range(num_games):
        outer_region = [[0 for i in range(n)] for j in range(n)]
        outer_region[n // 2][n // 2] = 1

        winner = play_game([[' ' for i in range(n)] for j in range(n)], 0, set([(i, j) for i in range(n) for j in range(n)]), outer_region)
        if winner:
            results[winner] += 1
        else:
            results["draw"] += 1

    return results

def main():
    n = 5
   
    num_games = 10
    results = {"X": [], "O": [], "draw": []}

    for i in range(0, num_games, 1):
        game_results = play(n, num_games) 
        results["X"].append(game_results["X"])
        results["O"].append(game_results["O"])
        results["draw"].append(game_results["draw"])

    plot_training_results(results)




if __name__ == "__main__":
    main()