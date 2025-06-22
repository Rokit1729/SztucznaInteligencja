import math
import random

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
    team_members = ["Wiktor Niedźwiedzki", "Filip Michewicz", "Mateusz Broczkowski"]
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

        # 3) 2 w linii + 2 puste
        elif count_self == L - 2 and count_empty == 2:
            score += 300

        # 4) 1 w linii i 3 puste
        elif count_self == 1 and count_empty == L - 1:
            score += 1

        # 5) Przeciwnik ma 4 w linii – przegrana
        if count_opp == L:
            score -= 1e6

        # 6) Przeciwnik ma 3 w linii + 1 puste
        elif count_opp == L - 1 and count_empty == 1:
            score -= 600

        # 7) Przeciwnik ma 2 w linii + 2 puste
        elif count_opp == L - 2 and count_empty == 2:
            score -= 350

        # 8) 1 w linii i 3 puste
        elif count_opp == 1 and count_empty == L - 1:
            score -= 1

        return score

    @staticmethod
    def evaluate_window_longer(window, symbol):
        score = 0
        opp = 1 - symbol
        L = len(window)

        count_self = window.count(symbol)
        count_opp = window.count(opp)
        count_empty = window.count(None)

        # 1) Jeżeli mamy 3 w linii i dwa po bokach puste
        if count_self == L - 2 and count_empty == 2:
            if window[0] == None and window[-1] == None:
                score += 300

        # 2) Jeżeli przeciwnik ma 3 w linii i dwa po bokach puste
        if count_opp == L - 2 and count_empty == 2:
            if window[0] == None and window[-1] == None:
                score -= 300

        return score

    @staticmethod
    def score_position(game: Game, symbol: int):
        score = 0
        R, C, L = game.n_rows, game.n_columns, game.winning_length

        # 1) Okna poziome
        for row in range(R):
            for col in range(C - L + 1):
                window = [Player.cell(game, row, col + i) for i in range(L)]
                score += Player.evaluate_window(window, symbol)

        # 2) Okna pionowe
        for col in range(C):
            for row in range(R - L + 1):
                window = [Player.cell(game, row + i, col) for i in range(L)]
                score += Player.evaluate_window(window, symbol)

        # 3) Okna przekątne w dół-prawo (\)
        for row in range(R - L + 1):
            for col in range(C - L + 1):
                window = [Player.cell(game, row + i, col + i) for i in range(L)]
                score += Player.evaluate_window(window, symbol)

        # 4) Okna przekątne w górę-prawo (/)
        for row in range(L - 1, R):
            for col in range(C - L + 1):
                window = [Player.cell(game, row - i, col + i) for i in range(L)]
                score += Player.evaluate_window(window, symbol)

        L_extra = L + 1

        # 5) Okna poziome - długości 5
        for row in range(R):
            for col in range(C - L_extra + 1):
                window = [Player.cell(game, row, col + i) for i in range(L_extra)]
                score += Player.evaluate_window_longer(window, symbol)

        # 6) Okna pionowe - długości 5
        for col in range(C):
            for row in range(R - L_extra + 1):
                window = [Player.cell(game, row + i, col) for i in range(L_extra)]
                score += Player.evaluate_window_longer(window, symbol)

        # 7) Okna przekątne w dół-prawo (\) - długości 5
        for row in range(R - L_extra + 1):
            for col in range(C - L_extra + 1):
                window = [Player.cell(game, row + i, col + i) for i in range(L_extra)]
                score += Player.evaluate_window_longer(window, symbol)

        # 8) Okna przekątne w górę-prawo (/) - długości 5
        for row in range(L_extra - 1, R):
            for col in range(C - L_extra + 1):
                window = [Player.cell(game, row - i, col + i) for i in range(L_extra)]
                score += Player.evaluate_window_longer(window, symbol)

        return score


    @staticmethod
    # Do wizualizacji gry z botami
    def print_board(game: Game):
        for r in reversed(range(game.n_rows)):
            row_symbols = []
            for c in range(game.n_columns):
                cell = Player.cell(game, r, c)
                row_symbols.append(str(cell) if cell is not None else ".")
            print("| " + " | ".join(row_symbols) + " |")
        print("-" * (4 * game.n_columns + 1))


def predict_next_move(move_history: list[int]) -> int:
    game = Game()
    
    #for move in move_history:
    #    if not Player.make_move(move):
    #        raise ValueError(f"Niepoprawny ruch w historii: kolumna {move} jest pełna lub nie istnieje.")

    for i in range(7):
        ile = move_history.count(i)
        if ile > 6: raise ValueError(f"Niepoprawny ruch w historii: kolumna {i} jest pełna lub nie istnieje.")
    
    current_player = len(move_history)%2
    maximizing = (current_player == 1)
    col, score, nodes = Player.alfabeta(game, Player.DEPTH, -math.inf, math.inf, maximizing)

    print(f"Obecny stan planszy (gracz {current_player} na ruchu):")
    Player.print_board(game)
    print(f"Ocena pozycji: {score}, Liczba odwiedzonych węzłów: {nodes}")
    return col



if __name__ == "__main__":
    choice = input("Czy chcesz zobaczyć rozgrywkę Bot vs Bot? (1 = tak, 0 = lista ruchów): ")
    if choice != "1" and choice != "0":
        print("Błędna opcja, domyślnie lista ruchów")
    
    if choice == "1":
        game = Game()
        turn = 0
        move_number = 0
        while True:
            move_number += 1
            print(f"\nTura {move_number} - Bot {turn}")
            col, score, nodes = Player.alfabeta(
                game, Player.DEPTH, -math.inf, math.inf, maximizingPlayer=(turn == 1)
            )
            Player.drop_piece(game, col, turn)
            game.current_player = 1 - game.current_player  # zmiana gracza
            print(f"Bot {turn} wybiera kolumnę {col}")
            Player.print_board(game)

            if Player.winning_move(game, turn):
                print(f"\nBot {turn} wygrywa!")
                break
            if Player.is_full(game):
                print("\nRemis!")
                break

            turn = 1 - turn

    else:
        print("Podaj historię ruchów jako listę numerów kolumn, oddzielonych przecinkami (np. 0,3,1,1,0,1):")
        wejscie = input("Historia ruchów: ")
        try:
            historia = [int(x.strip()) for x in wejscie.split(",") if x.strip() != ""]
            nastepny_ruch = predict_next_move(historia)
            print(f"Zalecany następny ruch: kolumna {nastepny_ruch}")
        except ValueError as e:
            print(f"Błąd: {e}")
        except Exception as e:
            print(f"Wystąpił nieoczekiwany błąd: {e}")

