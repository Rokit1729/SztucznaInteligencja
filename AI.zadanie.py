import math
import random
from copy import deepcopy


class Game:
    n_rows: int
    n_columns: int
    winning_length: int
    board: list[list[int]]
    move_history: list[int]
    current_player: int

    def __init__(self, n_rows: int = 7, n_columns: int = 7, winning_length: int = 4):
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.winning_length = winning_length
        self.current_player = 0
        self.board = [[] for _ in range(self.n_columns)]
        self.move_history = []


class Player:
    team_name = "Purple Cheese"
    team_members = ["Wiktor Niedźwiedzki", "Filip Michewicz", "Mateucz Broczkowski"]
    DEPTH = 4

    @staticmethod
    def make_move(game: Game) -> int:
        maximizing = game.current_player == 1
        col, _, _ = Player.alfabeta(game, Player.DEPTH, -math.inf, math.inf, maximizing)
        return col

    @staticmethod
    def alfabeta(game: Game, depth: int, alpha: float, beta: float, maximizingPlayer: bool):
        nodes = 1

        if Player.is_full(game) or Player.winning_move(game, 0) or Player.winning_move(game, 1) or depth == 0:
            score = Player.score_position(game, 1 if maximizingPlayer else 0)
            return (None, score, nodes)

        valid_cols = Player.valid_moves(game)
        if not valid_cols:
            return (None, 0.0, nodes)

        if maximizingPlayer:
            value = -math.inf
            chosen_col = random.choice(valid_cols)
            for col in valid_cols:
                child = Player.copy(game)
                Player.drop_piece(child, col, 1)
                _, score_child, child_nodes = Player.alfabeta(child, depth - 1, alpha, beta, False)
                nodes += child_nodes
                if score_child > value:
                    value = score_child
                    chosen_col = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return (chosen_col, value, nodes)
        else:
            value = math.inf
            chosen_col = random.choice(valid_cols)
            for col in valid_cols:
                child = Player.copy(game)
                Player.drop_piece(child, col, 0)
                _, score_child, child_nodes = Player.alfabeta(child, depth - 1, alpha, beta, True)
                nodes += child_nodes
                if score_child < value:
                    value = score_child
                    chosen_col = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return (chosen_col, value, nodes)

    @staticmethod
    def copy(game: Game) -> Game:
        new_game = Game(game.n_rows, game.n_columns, game.winning_length)
        new_game.current_player = game.current_player
        new_game.move_history = list(game.move_history)
        new_game.board = [list(col) for col in game.board]
        return new_game

    @staticmethod
    def valid_moves(game: Game):
        return [col for col in range(game.n_columns) if len(game.board[col]) < game.n_rows]

    @staticmethod
    def is_full(game: Game):
        return all(len(game.board[col]) >= game.n_rows for col in range(game.n_columns))

    @staticmethod
    def cell(game: Game, row: int, col: int):
        if row < len(game.board[col]):
            return game.board[col][row]
        return None

    @staticmethod
    def drop_piece(game: Game, column: int, symbol: int):
        if column not in Player.valid_moves(game):
            return False
        game.board[column].append(symbol)
        return True

    @staticmethod
    def winning_move(game: Game, player: int):
        L, R, C = game.winning_length, game.n_rows, game.n_columns

        # 1) Poziomo
        for row in range(R):
            for col_start in range(C - L + 1):
                if all(Player.cell(game, row, col_start + i) == player for i in range(L)):
                    return True

        # 2) Pionowo
        for col in range(C):
            for row_start in range(R - L + 1):
                if all(Player.cell(game, row_start + i, col) == player for i in range(L)):
                    return True
        
        # 3) Przekątna w dół-prawo (\)
        for row_start in range(R - L + 1):
            for col_start in range(C - L + 1):
                if all(Player.cell(game, row_start + i, col_start + i) == player for i in range(L)):
                    return True

        # 4) Przekątna w górę-prawo (/)
        for row_start in range(L - 1, R):
            for col_start in range(C - L + 1):
                if all(Player.cell(game, row_start - i, col_start + i) == player for i in range(L)):
                    return True

        return False

    @staticmethod
    def evaluate_window(window, symbol):
        score = 0
        opp = 1 - symbol
        L = len(window)

        count_self = window.count(symbol)
        count_opp = window.count(opp)
        count_empty = window.count(None)

        # 1) Jeżeli mamy 4 w linii – wygrana
        if count_self == L:
            score += 1e6

        # 2) 3 w linii + 1 puste
        elif count_self == L - 1 and count_empty == 1:
            score += 500

        # 3) 2 w linii + 2 puste, przy czym oba końce puste
        elif count_self == L - 2 and count_empty == 2:
            if window[0] is None and window[-1] is None:
                score += 300

        # 4) 1 pionek i 3 puste – minimalna szansa
        elif count_self == 1 and count_empty == L - 1:
            score += 50

        # 5) Przeciwnik ma 4 w linii – natychmiastowa kara
        if count_opp == L:
            score -= 1e6

        # 6) Przeciwnik ma 3 w linii + 1 puste – groźba wygranej
        elif count_opp == L - 1 and count_empty == 1:
            score -= 600

        # 7) Przeciwnik ma 2 w linii + 2 puste, oba końce puste
        elif count_opp == L - 2 and count_empty == 2:
            if window[0] is None and window[-1] is None:
                score -= 350

        return score

    @staticmethod
    def score_position(game: Game, symbol: int):
        score = 0
        R, C, L = game.n_rows, game.n_columns, game.winning_length

        center_col = C // 2
        center_array = [Player.cell(game, r, center_col) for r in range(R)]
        center_count = center_array.count(symbol)
        score += center_count * 6

        # Bonus za środkową kolumnę
        for row in range(R):
            for col in range(C - L + 1):
                window = [Player.cell(game, row, col + i) for i in range(L)]
                score += Player.evaluate_window(window, symbol)

        # Sprawdź wszystkie okna poziome, pionowe i skośne
        for col in range(C):
            for row in range(R - L + 1):
                window = [Player.cell(game, row + i, col) for i in range(L)]
                score += Player.evaluate_window(window, symbol)

        for row in range(R - L + 1):
            for col in range(C - L + 1):
                window = [Player.cell(game, row + i, col + i) for i in range(L)]
                score += Player.evaluate_window(window, symbol)

        for row in range(L - 1, R):
            for col in range(C - L + 1):
                window = [Player.cell(game, row - i, col + i) for i in range(L)]
                score += Player.evaluate_window(window, symbol)

        return score




game = Game()

game.board[0].append(0)  
game.board[1].append(1)  
game.board[0].append(0)  

game.current_player = 1


print(Player.make_move(game))

