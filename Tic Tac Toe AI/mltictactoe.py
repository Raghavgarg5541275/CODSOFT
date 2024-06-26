import tkinter as tk
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import beta


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        self.successes = defaultdict(int)
        self.failures = defaultdict(int)

    @staticmethod
    def get_state(board):
        return str(board)

    def choose_action(self, state, available_moves):
        if np.random.rand() < self.epsilon:
            return random.choice(available_moves)
        samples = [beta.rvs(self.successes[(state, move)] + 1, self.failures[(state, move)] + 1) for move in available_moves]
        max_sample = max(samples)
        actions_with_max_sample = [move for move, sample in zip(available_moves, samples) if sample == max_sample]
        return random.choice(actions_with_max_sample)

    def update_q_value(self, state, action, reward, next_state, available_moves):
        max_q_next = max([self.q_table.get((next_state, next_action), 0) for next_action in available_moves], default=0)
        current_q = self.q_table.get((state, action), 0)
        self.q_table[(state, action)] = current_q + self.alpha * (reward + self.gamma * max_q_next - current_q)
        if reward > 0:
            self.successes[(state, action)] += 1
        elif reward < 0:
            self.failures[(state, action)] += 1

    def train_from_dataset(self, filepath):
        data = pd.read_csv(filepath)
        for index, row in data.iterrows():
            board = [' '] * 9
            moves = row[:-1]
            result = row[-1]
            reward = 1 if result == 'win' else -1 if result == 'loss' else 0
            for i, move in enumerate(moves):
                if move != '?':
                    board[int(move)] = 'X' if i % 2 == 0 else 'O'
                    state = self.get_state(board)
                    action = (int(move) // 3, int(move) % 3)
                    if i < len(moves) - 1 and moves[i + 1] != '?':
                        next_move = moves[i + 1]
                        board[int(next_move)] = 'X' if (i + 1) % 2 == 0 else 'O'
                        next_state = self.get_state(board)
                    else:
                        next_state = self.get_state(board)
                    available_moves = [(r // 3, r % 3) for r in range(9) if board[r] == ' ']
                    self.update_q_value(state, action, reward, next_state, available_moves)


class TicTacToe:
    def __init__(self, root, agent):
        self.root = root
        self.root.title('Tic Tac Toe')
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        self.create_buttons()
        self.message = tk.Label(root, text="Your turn!", font=("Arial", 16))
        self.message.grid(row=3, column=0, columnspan=3)
        self.agent = agent
        self.game_count = 0
        self.game_counter_label = tk.Label(root, text="Games Played: 0", font=("Arial", 12))
        self.game_counter_label.grid(row=4, column=0, columnspan=3)
        self.new_game_button = tk.Button(root, text="New Game", font=("Arial", 16), command=self.new_game)
        self.new_game_button.grid(row=5, column=0, columnspan=3)

    def create_buttons(self):
        for i in range(3):
            for j in range(3):
                self.buttons[i][j] = tk.Button(self.root, text=' ', font=("Arial", 24), width=5, height=2,
                                               command=lambda row=i, col=j: self.human_move(row, col))
                self.buttons[i][j].grid(row=i, column=j)

    def human_move(self, row, col):
        if self.board[row][col] == ' ' and self.current_player == 'X':
            self.board[row][col] = 'X'
            self.buttons[row][col].config(text='X')
            if self.is_winner('X'):
                self.message.config(text="You win!")
                self.disable_buttons()
                return
            if self.is_full():
                self.message.config(text="It's a draw!")
                return
            self.current_player = 'O'
            self.message.config(text="AI's turn!")
            self.root.after(500, self.ai_move)

    def ai_move(self):
        state = self.agent.get_state(self.board)
        available_moves = self.get_available_moves()
        move = self.agent.choose_action(state, available_moves)
        if move:
            row, col = move
            self.board[row][col] = 'O'
            self.buttons[row][col].config(text='O')
            if self.is_winner('O'):
                self.message.config(text="AI wins!")
                self.disable_buttons()
                return
            if self.is_full():
                self.message.config(text="It's a draw!")
                return
            self.current_player = 'X'
            self.message.config(text="Your turn!")

    def is_winner(self, player):
        win_states = [
            [self.board[0][0], self.board[0][1], self.board[0][2]],
            [self.board[1][0], self.board[1][1], self.board[1][2]],
            [self.board[2][0], self.board[2][1], self.board[2][2]],
            [self.board[0][0], self.board[1][0], self.board[2][0]],
            [self.board[0][1], self.board[1][1], self.board[2][1]],
            [self.board[0][2], self.board[1][2], self.board[2][2]],
            [self.board[0][0], self.board[1][1], self.board[2][2]],
            [self.board[2][0], self.board[1][1], self.board[0][2]],
        ]
        return [player, player, player] in win_states

    def is_full(self):
        return all(cell != ' ' for row in self.board for cell in row)

    def get_available_moves(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == ' ']

    def disable_buttons(self):
        for i in range(3):
            for j in range(3):
                self.buttons[i][j].config(state=tk.DISABLED)

    def new_game(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
        for i in range(3):
            for j in range(3):
                self.buttons[i][j].config(text=' ', state=tk.NORMAL)
        self.message.config(text="Your turn!")
        self.game_count += 1
        self.game_counter_label.config(text=f"Games Played: {self.game_count}")


# Dataset:
agent = QLearningAgent()
agent.train_from_dataset('./ttt.csv')
root = tk.Tk()
game = TicTacToe(root, agent)
root.mainloop()