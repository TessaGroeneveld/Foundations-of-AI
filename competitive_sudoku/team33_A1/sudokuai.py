#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
from typing import List
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard
import competitive_sudoku.sudokuai


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    def compute_best_move(self, game_state: GameState) -> None:
        legal_moves = self.get_legal_moves(game_state)
        if not legal_moves:
            return 

        # First move to ensure that we don't run out of time before the first move is made.
        self.propose_move(legal_moves[0])

        # I guess that this needs to be changed with our algorithm, but for now, we just propose random legal moves.
        while True:
            time.sleep(0.2)
            self.propose_move(random.choice(legal_moves))

    def get_legal_moves(self, game_state: GameState) -> list[Move]:
        """
        Returns a list of all legal moves for the current game state."""
        board = game_state.board
        N = board.N
        m = board.m
        n = board.n

        allowed = game_state.player_squares()
        # classic mode
        if allowed is None: 
            candidate_squares = [(i, j) for i in range(N) for j in range(N) if board.get((i, j)) == SudokuBoard.empty]
        else: 
            candidate_squares = [sq for sq in allowed if board.get(sq) == SudokuBoard.empty]

        # Create a set of taboo moves for faster lookup
        taboo_set = {
            (move.square[0], move.square[1], move.value)
            for move in game_state.taboo_moves
        }

        legal_moves: List[Move] = []

        for i, j in candidate_squares:
            # Find all values that are already used in the same row, column, and box
            used_values = set()

            for col in range(N):
                val = board.get((i, col))
                if val != SudokuBoard.empty:
                    used_values.add(val)

            for row in range(N):
                val = board.get((row, j))
                if val != SudokuBoard.empty:
                    used_values.add(val)

            start_i = (i // m) * m
            start_j = (j // n) * n
            for row in range(start_i, start_i + m):
                for col in range(start_j, start_j + n):
                    val = board.get((row, col))
                    if val != SudokuBoard.empty:
                        used_values.add(val)

            # Generate moves for the available values from 1 to N
            for value in range(1, N + 1):
                if value not in used_values and (i, j, value) not in taboo_set:
                    legal_moves.append(Move((i, j), value))

        return legal_moves

    def apply_move(self, game_state: GameState, move: Move) -> GameState:
        """
        Applies a move to the game state and returns the new game state.
        """
        m = game_state.board.m
        n = game_state.board.n
        new_board = SudokuBoard(m, n)
        new_board.squares = game_state.board.squares[:]
        new_board.put(move.square, move.value)

        i, j = move.square
        N = new_board.N

        completed_units = 0

        if all(new_board.get((i, col)) != SudokuBoard.empty for col in range(N)):
            completed_units += 1

        if all(new_board.get((row, j)) != SudokuBoard.empty for row in range(N)):
            completed_units += 1

        start_i = (i // m) * m
        start_j = (j // n) * n
        if all(
            new_board.get((row, col)) != SudokuBoard.empty
            for row in range(start_i, start_i + m)
            for col in range(start_j, start_j + n)
        ):
            completed_units += 1

        points_table = {0: 0, 1: 1, 2: 3, 3: 7}
        earned_points = points_table[completed_units]

        new_scores = list(game_state.scores)
        new_scores[game_state.current_player - 1] += earned_points

        # Update occupied squares for the current player
        occ1 = (
            list(game_state.occupied_squares1)
            if game_state.occupied_squares1 is not None
            else None
        )
        occ2 = (
            list(game_state.occupied_squares2)
            if game_state.occupied_squares2 is not None
            else None
        )

        if game_state.current_player == 1 and occ1 is not None:
            occ1.append(move.square)
        elif game_state.current_player == 2 and occ2 is not None:
            occ2.append(move.square)

        next_player = 2 if game_state.current_player == 1 else 1

        return GameState(
            initial_board=game_state.initial_board,
            board=new_board,
            taboo_moves=list(game_state.taboo_moves),
            moves=list(game_state.moves) + [move],
            scores=new_scores,
            current_player=next_player,
            allowed_squares1=game_state.allowed_squares1,
            allowed_squares2=game_state.allowed_squares2,
            occupied_squares1=occ1,
            occupied_squares2=occ2,
        )
