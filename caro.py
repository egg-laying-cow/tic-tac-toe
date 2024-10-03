import random
import matplotlib.pyplot as plt
from collections import defaultdict

X = 'X'
O = 'O'
SPACE = ' '

def get_symbol(moved_count: int) -> str:
    return ((moved_count + 1) % 2) * X + (moved_count % 2) * O

class Caro:
    def __init__(self, max_len = 5):
        self.max_len = max_len
        self.moved_count = 0
        self.last_move = None
        self.board = [[SPACE for _ in range(self.max_len)] for _ in range(self.max_len)]
        self.empty_positions = [(i, j) for i in range(max_len) for j in range(max_len)]

    def move(self, x, y):
        if self.board[y][x] != SPACE:
            return False

        symbol = get_symbol(self.moved_count)
        self.board[y][x] = symbol
        self.last_move = (x, y)
        self.empty_positions.remove((x, y))
        self.moved_count += 1
        return True

    def print_board(self):
        for row in self.board:
            print(row)
        print()

    def check_game_end(self) -> bool:
        return (self._check_row_end() or
                self._check_column_end() or
                self._check_diagonal_main_end() or
                self._check_diagonal_secondary())  
    
    def _check_row_end(self) -> bool:
        count = 0
        x1 = max(self.last_move[0] - (self.max_len - 1), 0)
        x2 = min(self.last_move[0] + (self.max_len - 1), len(self.board) - 1)

        for x in range(x1, x2 + 1):
            if self.board[self.last_move[1]][x] == self.board[self.last_move[1]][self.last_move[0]]:
                count += 1
            else:
                count = 0

            if (count >= self.max_len):
                return True

        return False

    def _check_column_end(self) -> bool:
        count = 0
        y1 = max(self.last_move[1] - (self.max_len - 1), 0)
        y2 = min(self.last_move[1] + (self.max_len - 1), len(self.board) - 1)
        
        for y in range(y1, y2 + 1):
            if self.board[y][self.last_move[0]] == self.board[self.last_move[1]][self.last_move[0]]:
                count += 1
            else: 
                count = 0

            if (count >= self.max_len):
                return True

        return False

    def _check_diagonal_main_end(self) -> bool:
        count = 0
        x1, x2, y1, y2 = 0, 0, 0, 0
        if (self.last_move[0] <= self.last_move[1]):
            x1 = max(self.last_move[0] - (self.max_len - 1), 0)
            y1 = self.last_move[1] - (self.last_move[0] - x1)
            y2 = min(self.last_move[1] + (self.max_len - 1), len(self.board) - 1)
            x2 = self.last_move[0] + (y2 - self.last_move[1])
        else:
            y1 = max(self.last_move[1] - (self.max_len - 1), 0)
            x1 = self.last_move[0] - (self.last_move[1] - y1)
            x2 = min(self.last_move[0] + (self.max_len - 1), len(self.board) - 1)
            y2 = self.last_move[1] + (x2 - self.last_move[0])

        y = y1
        for x in range(x1, x2 + 1):
            if self.board[y][x] == self.board[self.last_move[1]][self.last_move[0]]:
                count += 1
            else:
                count = 0
            y += 1
            
            if count >= self.max_len:
                return True        

        return False

    def _check_diagonal_secondary(self) -> bool:
        count = 0
        x1, x2, y1, y2 = 0, 0, 0, 0
        if (self.last_move[0] <= len(self.board) - 1 - self.last_move[1]):
            x1 = max(self.last_move[0] - (self.max_len - 1), 0)
            y2 = self.last_move[1] + (self.last_move[0] - x1)
            y1 = max(self.last_move[1] - (self.max_len - 1), 0)
            x2 = self.last_move[0] + (self.last_move[1] - y1)
        else:
            y2 = min(self.last_move[1] + (self.max_len - 1), len(self.board) - 1)
            x1 = self.last_move[0] - (y2 - self.last_move[1])
            x2 = min(self.last_move[0] + (self.max_len - 1), len(self.board) - 1)
            y1 = self.last_move[1] - (x2 - self.last_move[0])

        y = y2
        for x in range(x1, x2 + 1):
            if self.board[y][x] == self.board[self.last_move[1]][self.last_move[0]]:
                count += 1
            else:
                count = 0
            y -= 1

            if count >= self.max_len:
                return True
            
        return False
    
    def to_string(self):
        return ''.join([''.join(row) for row in self.board])
    
    def get_valid_actions(self):
        return self.empty_positions
    
    def is_full(self):
        return len(self.empty_positions) == 0
    
    def reset(self):
        self.moved_count = 0
        self.last_move = None
        self.board = [[SPACE for _ in range(self.max_len)] for _ in range(self.max_len)]
        self.empty_positions = [(i, j) for i in range(self.max_len) for j in range(self.max_len)]

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, board_size=5, q_table=defaultdict(lambda: defaultdict(float))):
        self.alpha = alpha  # Tốc độ học
        self.gamma = gamma  # hệ số giảm
        self.epsilon = epsilon  # Xác suất chọn hành động ngẫu nhiên (exploration)
        self.q_table = q_table
        self.board_size = board_size

    def choose_action(self, board: Caro):
        state = board.to_string()
        valid_actions = board.get_valid_actions()
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_actions)  # Chọn hành động ngẫu nhiên
        else:
            q_values = [self.q_table[state][action] for action in valid_actions]
            max_q = max(q_values)
            best_actions = [valid_actions[i] for i in range(len(valid_actions)) if q_values[i] == max_q]
            return random.choice(best_actions)
        
    def update_q_value(self, board, action, reward):
        state = board.to_string()
        next_valid_actions = board.get_valid_actions()
        current_q = self.q_table[state][action]
        
        if len(next_valid_actions) > 0:
            future_q = max([self.q_table[state][next_action] for next_action in next_valid_actions])
        else:
            future_q = 0
        
        self.q_table[state][action] += self.alpha * (reward + self.gamma * future_q - current_q)


def play_game(board: Caro, agent: QLearningAgent, train=True):
    players = ["X", "O"]
    current_player_idx = 0
    winner = None

    while True:
        current_player = players[current_player_idx]
        valid_actions = board.get_valid_actions()

        # Agent chọn hành động

        if current_player == "X":  # Chỉ agent "X" học Q-learning
            action = agent.choose_action(board)
        else:
            action = random.choice(valid_actions)  # Người chơi "O" chỉ chọn ngẫu nhiên

        board.move(action[0], action[1])

        # Kiểm tra nếu ai đó thắng
        if board.check_game_end():
            winner = current_player
            if train:
                if current_player == "X":
                    agent.update_q_value(board, action, 1)  # Thắng thì nhận phần thưởng 1
            break

        # Nếu hòa
        if board.is_full():
            if train and current_player == "X":
                agent.update_q_value(board, action, 0.5)  # Hòa thì nhận phần thưởng 0.5
            break
        
        if train and current_player == "X":
            agent.update_q_value(board, action, 0)

        current_player_idx = 1 - current_player_idx  

    return winner

def train_agent(board, agent, num_games):
    results = {"X": 0, "O": 0, "draw": 0}

    for _ in range(num_games):
        winner = play_game(board, agent, train=True)
        if winner:
            results[winner] += 1
        else:
            results["draw"] += 1
        board.reset()

    return results

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

# Thiết lập và chạy mô phỏng
board_size = 5  
board = Caro(max_len=board_size)
q_table = defaultdict(lambda: defaultdict(float))
agent = QLearningAgent(board_size=board_size, q_table=q_table)

num_games = 1000
results = {"X": [], "O": [], "draw": []}

# Huấn luyện và lưu kết quả mỗi 100 trận
for i in range(0, num_games, 100):
    game_results = train_agent(board, agent, 100)
    results["X"].append(game_results["X"])
    results["O"].append(game_results["O"])
    results["draw"].append(game_results["draw"])

# print(agent.q_table)
plot_training_results(results)

        