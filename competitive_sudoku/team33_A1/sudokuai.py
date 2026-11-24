#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import math
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard
import competitive_sudoku.sudokuai

# points for completing 0, 1, 2 or 3 regions in one move
POINTS = [0, 1, 3, 7]

# weight of the mobility term in the evaluation (0 turns it off)
# with 0.1, ten extra reachable cells count as one point
MOBILITY_WEIGHT = 0.1


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    def compute_best_move(self, game_state: GameState) -> None:
        # copy of the game that we can play moves on and undo them
        state = SearchState(game_state)

        legal_moves = state.legal_moves()
        if not legal_moves:
            return

        # propose a move right away in case we run out of time
        self.propose_move(to_move(legal_moves[0]))

        # index of our score and the opponent's score in the scores list
        me = game_state.current_player - 1
        opponent = 1 - me

        def evaluate(state):
            # score difference from our point of view
            value = state.scores[me] - state.scores[opponent]

            # mobility: if you get boxed in you have to skip turns and the
            # opponent gets the rest of the board, so having more cells to
            # play in than the opponent is good
            # (not used in classic mode since everyone can play everywhere)
            if MOBILITY_WEIGHT and not state.is_classic:
                my_cells = state.reachable[me + 1]
                their_cells = state.reachable[opponent + 1]
                value += MOBILITY_WEIGHT * (my_cells - their_cells)
            return value

        # minimax with alpha-beta pruning, returns (value, best move)
        # first_move = best move of the previous depth, gets searched first
        def minimax(state, depth, alpha, beta, maximizing, first_move=None):
            # at depth 0 we only need the evaluation, no need to generate moves
            if depth == 0:
                return evaluate(state), None

            # moves that score points are already sorted to the front
            moves = state.legal_moves()
            if not moves:
                # no moves means the player skips their turn, the game is not over
                state.pass_turn()
                if not state.legal_moves():
                    # nobody can move anymore
                    state.pass_turn()
                    return evaluate(state), None
                # skipping isn't a real move so depth stays the same
                score, _ = minimax(state, depth, alpha, beta, not maximizing)
                state.pass_turn()
                return score, None

            if first_move is not None and first_move in moves:
                moves.remove(first_move)
                moves.insert(0, first_move)

            best_move = None

            if maximizing:
                best_score = -math.inf
                for move in moves:
                    points = state.play(move)
                    score, _ = minimax(state, depth - 1, alpha, beta, False)
                    state.undo(move, points)
                    if score > best_score:
                        best_score, best_move = score, move
                    alpha = max(alpha, best_score)
                    if alpha >= beta:
                        break  # prune
                return best_score, best_move
            else:
                best_score = math.inf
                for move in moves:
                    points = state.play(move)
                    score, _ = minimax(state, depth - 1, alpha, beta, True)
                    state.undo(move, points)
                    if score < best_score:
                        best_score, best_move = score, move
                    beta = min(beta, best_score)
                    if alpha >= beta:
                        break  # prune
                return best_score, best_move

        # iterative deepening: search depth 1, 2, 3, ... and propose the best move
        # after each depth. we don't know the time limit, the framework just stops
        # us and uses the last proposed move. no point going deeper than the
        # number of empty cells.
        previous_best = None
        for depth in range(1, state.empty_cells + 1):
            _, best_move = minimax(state, depth, -math.inf, math.inf, True, previous_best)
            if best_move is not None:
                self.propose_move(to_move(best_move))
                previous_best = best_move


def to_move(move) -> Move:
    """(row, col, value) tuple -> Move"""
    row, col, value = move
    return Move((row, col), value)


class SearchState:
    """
    Copy of the game state used during the search. Moves are played with play()
    and taken back with undo(), so we don't have to copy the board every time.

    For every row, column and block it keeps the values that are used and the
    number of empty cells, so checking moves and points is fast.

    Moves are (row, col, value) tuples.
    """

    def __init__(self, game_state: GameState):
        board = game_state.board
        self.N = N = board.N
        self.m = board.m      # block height (rows)
        self.n = board.n      # block width (columns)

        # board as a 2D list, 0 = empty
        self.grid = [[board.get((i, j)) for j in range(N)] for i in range(N)]

        self.row_used = [set() for _ in range(N)]
        self.col_used = [set() for _ in range(N)]
        self.block_used = [set() for _ in range(N)]
        self.row_empty = [0] * N
        self.col_empty = [0] * N
        self.block_empty = [0] * N
        self.empty_cells = 0

        for i in range(N):
            for j in range(N):
                b = self.block(i, j)
                value = self.grid[i][j]
                if value == SudokuBoard.empty:
                    self.row_empty[i] += 1
                    self.col_empty[j] += 1
                    self.block_empty[b] += 1
                    self.empty_cells += 1
                else:
                    self.row_used[i].add(value)
                    self.col_used[j].add(value)
                    self.block_used[b].add(value)

        # taboo moves don't change during the search, so build the set once
        self.taboo = {(t.square[0], t.square[1], t.value) for t in game_state.taboo_moves}

        # copied so we don't change the real game state
        self.scores = list(game_state.scores)
        self.current_player = game_state.current_player
        # indexed by player number (index 0 not used)
        self.start_squares = [None, game_state.allowed_squares1, game_state.allowed_squares2]
        self.is_classic = game_state.allowed_squares1 is None and game_state.allowed_squares2 is None
        self.occupied = [None,
                         list(game_state.occupied_squares1 or []),
                         list(game_state.occupied_squares2 or [])]

        # neighbours of every cell (also diagonal)
        self.neighbors = [[[(i + di, j + dj)
                            for di in (-1, 0, 1) for dj in (-1, 0, 1)
                            if (di or dj) and 0 <= i + di < N and 0 <= j + dj < N]
                           for j in range(N)] for i in range(N)]

        # for mobility we keep track of how many empty cells each player can reach,
        # updated in play()/undo() so we don't have to count it at every position
        # reach[p][i][j] = number of reasons player p can play in (i, j)
        #                  (start region or next to one of their cells)
        # reachable[p] = number of empty cells player p can reach
        if not self.is_classic:
            self.reach = [None, [[0] * N for _ in range(N)], [[0] * N for _ in range(N)]]
            for p in (1, 2):
                for (i, j) in self.start_squares[p]:
                    self.reach[p][i][j] += 1
                for (i, j) in self.occupied[p]:
                    for (r, c) in self.neighbors[i][j]:
                        self.reach[p][r][c] += 1
            self.reachable = [0] + [
                sum(1 for i in range(N) for j in range(N)
                    if self.grid[i][j] == SudokuBoard.empty and self.reach[p][i][j] > 0)
                for p in (1, 2)]

    def block(self, i, j):
        """block number of cell (i, j)"""
        return (i // self.m) * self.m + (j // self.n)

    def allowed_squares(self, player=None):
        """
        Empty cells the player is allowed to play in (start region + cells next to
        their own cells). Uses the current player if no player is given.
        In classic mode all empty cells are allowed.
        """
        if player is None:
            player = self.current_player
        N, grid = self.N, self.grid
        if self.start_squares[player] is None:
            return [(i, j) for i in range(N) for j in range(N) if grid[i][j] == SudokuBoard.empty]

        squares = {(i, j) for (i, j) in self.start_squares[player] if grid[i][j] == SudokuBoard.empty}
        for (i, j) in self.occupied[player]:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    r, c = i + di, j + dj
                    if 0 <= r < N and 0 <= c < N and grid[r][c] == SudokuBoard.empty:
                        squares.add((r, c))
        return sorted(squares)    # sorted so the order is always the same

    def points_for(self, i, j):
        """points for filling (i, j), a region is completed if this is its last empty cell"""
        completed = ((self.row_empty[i] == 1) +
                     (self.col_empty[j] == 1) +
                     (self.block_empty[self.block(i, j)] == 1))
        return POINTS[completed]

    def legal_moves(self):
        """
        All legal moves for the current player: allowed empty cell, value not in the
        row/column/block yet, and not taboo. Moves that score more points come first.
        """
        moves = []
        for (i, j) in self.allowed_squares():
            used = self.row_used[i] | self.col_used[j] | self.block_used[self.block(i, j)]
            points = self.points_for(i, j)
            for value in range(1, self.N + 1):
                if value not in used and (i, j, value) not in self.taboo:
                    moves.append((points, (i, j, value)))
        # most points first (sort is stable so ties keep their order)
        moves.sort(key=lambda item: -item[0])
        return [move for _, move in moves]

    def play(self, move):
        """play a move for the current player, returns the points scored"""
        i, j, value = move
        b = self.block(i, j)
        points = self.points_for(i, j)      # has to be before filling the cell
        player = self.current_player

        self.grid[i][j] = value
        self.row_used[i].add(value)
        self.col_used[j].add(value)
        self.block_used[b].add(value)
        self.row_empty[i] -= 1
        self.col_empty[j] -= 1
        self.block_empty[b] -= 1
        self.empty_cells -= 1
        self.occupied[player].append((i, j))
        self.scores[player - 1] += points

        if not self.is_classic:
            # cell is filled so it's not reachable anymore for either player
            for p in (1, 2):
                if self.reach[p][i][j] > 0:
                    self.reachable[p] -= 1
            # the player can now reach the empty neighbours of this cell
            for (r, c) in self.neighbors[i][j]:
                self.reach[player][r][c] += 1
                if self.reach[player][r][c] == 1 and self.grid[r][c] == SudokuBoard.empty:
                    self.reachable[player] += 1

        self.current_player = 3 - player
        return points

    def undo(self, move, points):
        """undo a move made with play() (same steps in reverse)"""
        i, j, value = move
        b = self.block(i, j)
        self.current_player = 3 - self.current_player
        player = self.current_player

        if not self.is_classic:
            for (r, c) in self.neighbors[i][j]:
                self.reach[player][r][c] -= 1
                if self.reach[player][r][c] == 0 and self.grid[r][c] == SudokuBoard.empty:
                    self.reachable[player] -= 1
            for p in (1, 2):
                if self.reach[p][i][j] > 0:
                    self.reachable[p] += 1   # cell becomes empty again

        self.scores[player - 1] -= points
        self.occupied[player].pop()
        self.empty_cells += 1
        self.block_empty[b] += 1
        self.col_empty[j] += 1
        self.row_empty[i] += 1
        self.block_used[b].discard(value)
        self.col_used[j].discard(value)
        self.row_used[i].discard(value)
        self.grid[i][j] = SudokuBoard.empty

    def pass_turn(self):
        """current player skips their turn"""
        self.current_player = 3 - self.current_player
