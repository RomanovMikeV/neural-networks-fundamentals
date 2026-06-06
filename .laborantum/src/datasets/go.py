import numpy as np
import torch


class GoPositionDataset(torch.utils.data.Dataset):
    def __init__(self, n_positions=1024, board_size=9, seed=0):
        self.n_positions = int(n_positions)
        self.board_size = int(board_size)
        self.rng = np.random.default_rng(seed)
        self.boards = []
        self.value_targets = []
        self.policy_targets = []

        ## YOUR CODE HERE
        if not self.boards:
            board = np.zeros((self.board_size, self.board_size), dtype=np.int64)
            self.boards.append(board)
            self.value_targets.append(0.0)
            self.policy_targets.append(0)

    def _sample_board(self):
        probabilities = np.array([0.70, 0.15, 0.15])
        values = np.array([0, 1, -1])
        board = self.rng.choice(values, size=(self.board_size, self.board_size), p=probabilities)
        if not np.any(board == 0):
            board[self.rng.integers(0, self.board_size), self.rng.integers(0, self.board_size)] = 0
        return board.astype(np.int64)

    def _neighbors(self, row, col):
        for drow, dcol in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nrow = row + drow
            ncol = col + dcol
            if 0 <= nrow < self.board_size and 0 <= ncol < self.board_size:
                yield nrow, ncol

    def _liberty_score(self, board, color):
        liberties = 0
        for row in range(self.board_size):
            for col in range(self.board_size):
                if board[row, col] == color:
                    liberties += sum(board[nrow, ncol] == 0 for nrow, ncol in self._neighbors(row, col))
        return liberties

    def _evaluate_position(self, board):
        black_stones = np.sum(board == 1)
        white_stones = np.sum(board == -1)
        black_liberties = self._liberty_score(board, 1)
        white_liberties = self._liberty_score(board, -1)
        raw_score = (black_stones - white_stones) + 0.25 * (black_liberties - white_liberties)
        return float(np.tanh(raw_score / self.board_size))

    def _move_score(self, board, row, col):
        if board[row, col] != 0:
            return -1.0e9
        own_neighbors = 0
        opponent_neighbors = 0
        empty_neighbors = 0
        for nrow, ncol in self._neighbors(row, col):
            own_neighbors += board[nrow, ncol] == 1
            opponent_neighbors += board[nrow, ncol] == -1
            empty_neighbors += board[nrow, ncol] == 0
        center = (self.board_size - 1) / 2
        center_bonus = -0.04 * (abs(row - center) + abs(col - center))
        return 1.5 * opponent_neighbors + own_neighbors + 0.25 * empty_neighbors + center_bonus

    def _best_move(self, board):
        best_index = 0
        best_score = -1.0e9
        for row in range(self.board_size):
            for col in range(self.board_size):
                score = self._move_score(board, row, col)
                index = row * self.board_size + col
                if score > best_score:
                    best_score = score
                    best_index = index
        return int(best_index)

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, index):
        board = self.boards[index]
        black = board == 1
        white = board == -1
        empty = board == 0
        features = np.stack([black, white, empty]).astype(np.float32)
        return {
            'board': torch.tensor(features, dtype=torch.float32),
            'value': torch.tensor(self.value_targets[index], dtype=torch.float32),
            'move': torch.tensor(self.policy_targets[index], dtype=torch.long),
            'legal_mask': torch.tensor(empty.reshape(-1), dtype=torch.bool),
        }
