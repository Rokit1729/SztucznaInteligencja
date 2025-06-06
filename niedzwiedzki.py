import random
import math
import time


class Game:
    """
    Klasa reprezentująca rozgrywkę naszej gry. Zawiera następujące pola:

    n_rows, n_columns : 
        liczba wierszy i kolumn na planszy (domyślnie 7)
    winning_length : 
        liczba symboli w jednej linii potrzebna, aby wygrać (domyślnie 4)
    board : 
        plansza, reprezentowana jako lista list. Każda lista to jedna kolumna, a symbole na tej liście 
        to symbole w tej kolumnie, od najniższego do najwyższego. Np. jeśli plansza w grze 4x4 wygląda tak:

        +-+-+-+-+
        | | | | |
        +-+-+-+-+
        | |1| | |
        +-+-+-+-+
        |0|1| | |
        +-+-+-+-+
        |0|0| |1|
        +-+-+-+-+
        
        to zmienna board ma wartość [[0,0], [0,1,1], [], [1]]

        Gracze zawsze reprezentowani są symbolami 0 i 1, zaczyna gracz 0.

    current_player : 
        gracz, na którego ruch aktualnie czekamy. Może to być tylko wartość 0 lub 1, na początku gry jest to 0.

    move_history : 
        historia ruchów od początku rozgrywki, w postaci listy kolumn, w których kolejno były umieszczane symbole. 
        Np. plansza z przykładu wyżej mogłaby odpowiadać następującej wartości zmiennej move_history : [0,3,1,1,0,1]
    """
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

    def make_move(self, column):
        """
        Wstawia pionek aktualnego gracza (self.current_player) do kolumny 'column'.
        Jeżeli ruch jest nieprawidłowy (kolumna pełna lub indeks spoza zakresu), zwraca False.
        W przeciwnym razie:
          1) dopisuje symbol (0 lub 1) do self.board[column] na koniec listy,
          2) dodaje kolumnę do move_history,
          3) zamienia current_player = 1 - current_player,
          4) zwraca True.
        """
        if column < 0 or column >= self.n_columns:
            return False
        if len(self.board[column]) >= self.n_rows:
            return False

        self.board[column].append(self.current_player)
        self.move_history.append(column)
        self.current_player = 1 - self.current_player
        return True
    
    def valid_moves(self):
        return [col for col in range(self.n_columns) if len(self.board[col]) < self.n_rows]
    
    def is_full(self):
        return all(len(self.board[col]) >= self.n_rows for col in range(self.n_columns))

    def cell(self, row, col):
        if row < len(self.board[col]):
            return self.board[col][row]
        return None
    
    def drop_piece(self, column, symbol):
        if column not in self.valid_moves():
            return False
        self.board[column].append(symbol)
        return True

    def winning_move(self, player):
        L = self.winning_length
        R = self.n_rows
        C = self.n_columns

        # 1) Poziomo
        for row in range(R):
            for col_start in range(C - L + 1):
                if all(self.cell(row, col_start + i) == player for i in range(L)):
                    return True

        # 2) Pionowo
        for col in range(C):
            for row_start in range(R - L + 1):
                if all(self.cell(row_start + i, col) == player for i in range(L)):
                    return True

        # 3) Przekątna w dół-prawo (\)
        for row_start in range(R - L + 1):
            for col_start in range(C - L + 1):
                if all(self.cell(row_start + i, col_start + i) == player for i in range(L)):
                    return True

        # 4) Przekątna w górę-prawo (/)
        for row_start in range(L - 1, R):
            for col_start in range(C - L + 1):
                if all(self.cell(row_start - i, col_start + i) == player for i in range(L)):
                    return True

        return False

    def game_over(self):
        return self.winning_move(0) or self.winning_move(1) or self.is_full()

    def print_board(self):
        for r in reversed(range(self.n_rows)):
            row_symbols = []
            for c in range(self.n_columns):
                cell = self.cell(r, c)
                row_symbols.append(str(cell) if cell is not None else ".")
            print("| " + " | ".join(row_symbols) + " |")
        print("-" * (4 * self.n_columns + 1))

    def copy(self):
        new_game = Game(self.n_rows, self.n_columns, self.winning_length)
        new_game.current_player = self.current_player
        new_game.move_history = self.move_history.copy()
        new_game.board = [col.copy() for col in self.board]
        return new_game

    def evaluate_window(self, window, symbol):
        score = 0.0
        opp = 1 - symbol
        L = self.winning_length

        count_self = window.count(symbol)
        count_opp = window.count(opp)
        count_empty = window.count(None)

        # 1) Jeżeli mamy 4 w linii
        if count_self == L:
            score += 1e6
        # 2) 3 w linii + 1 puste
        elif count_self == L - 1 and count_empty == 1:
            score += 500
        # 3) 2 w linii + 2 puste
        elif count_self == L - 2 and count_empty == 2:
            score += 200

        # Premia, jeśli 3 w linii i oba końce puste
        # Do poprawki
        #if count_self == L - 2 and window[0] is None and window[-1] is None:
        #    score += 300

        # Blokowanie przeciwnika
        if count_opp == L:
            score -= 1e10
        elif count_opp == L - 1 and count_empty == 1:
            score -= 400
        elif count_opp == L - 2 and count_empty == 2:
            score -= 100

        return score

    def score_position(self, symbol):
        score = 0.0
        R = self.n_rows
        C = self.n_columns
        L = self.winning_length
        opp = 1 - symbol

        # 1) Kolumny bliżej środka dają premię
        # Wstępnie napisać na kartce które pozycje dają ile możliwości na wygraną i je punktować np. dla środowego kwadratu jest to 16 a dla narożnika tylko 3
        for col in range(C):
            distance = min(col, C - 1 - col)
            count_self = sum(1 for row in range(R) if self.cell(row, col) == symbol)
            count_opp = sum(1 for row in range(R) if self.cell(row, col) == opp)
            score += distance * count_self
            score -= distance * count_opp

        # 2) Okna poziome
        for row in range(R):
            for col_start in range(C - L + 1):
                window = [self.cell(row, col_start + i) for i in range(L)]
                score += self.evaluate_window(window, symbol)

        # 3) Okna pionowe
        for col in range(C):
            for row_start in range(R - L + 1):
                window = [self.cell(row_start + i, col) for i in range(L)]
                score += self.evaluate_window(window, symbol)

        # 4) Okna przekątne w dół-prawo (\)
        for row_start in range(R - L + 1):
            for col_start in range(C - L + 1):
                window = [self.cell(row_start + i, col_start + i) for i in range(L)]
                score += self.evaluate_window(window, symbol)

        # 5) Okna przekątne w górę-prawo (/)
        for row_start in range(L - 1, R):
            for col_start in range(C - L + 1):
                window = [self.cell(row_start - i, col_start + i) for i in range(L)]
                score += self.evaluate_window(window, symbol)

        return score


class Player:
    team_name = "Purple Cheese"
    team_members = ["Wiktor Niedźwiedzki", "Filip Michewicz", "Mateucz Broczkowski"]
    DEPTH = 4

    def alfabeta(game: Game, depth: int, alpha: float, beta: float, maximizingPlayer: bool):
        nodes = 1

        if game.is_full(): # Remis
            return (None, 0.0, nodes)
        if depth == 0:
            return (None, game.score_position(1), nodes)

        valid_cols = game.valid_moves()
        if not valid_cols:
            return (None, 0.0, nodes)

        if maximizingPlayer:
            value = -math.inf
            chosen_col = random.choice(valid_cols)
            for col in valid_cols:
                child_game = game.copy()
                child_game.drop_piece(col, 1)
                _, score_child, child_nodes = Player.alfabeta(child_game, depth - 1, alpha, beta, False)
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
                child_game = game.copy()
                child_game.drop_piece(col, 0)
                _, score_child, child_nodes = Player.alfabeta(child_game, depth - 1, alpha, beta, True)
                nodes += child_nodes
                if score_child < value:
                    value = score_child
                    chosen_col = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return (chosen_col, value, nodes)

    def make_move(self, game: Game) -> int:
        col, _, _ = Player.alfabeta(game, Player.DEPTH, -math.inf, math.inf, True)
        return col

if __name__ == "__main__":
    PLAYER = 0
    BOT = 1
    game = Game()
    bot = Player()

    # 1) Wybór trybu gry
    print("Wybierz tryb gry:")
    print("1 = Gracz vs Bot")
    print("2 = Bot vs Bot")
    choice = input("Twoja opcja (1 lub 2): ")

    if choice == '1':
        human_vs_bot = True
        print("Wybrano: Gracz vs Bot")
        # 2) Wybór, kto zaczyna: Gracz czy Bot
        first = input("Kto zaczyna? (1 = Gracz, 2 = Bot): ")
        if first == '1':
            turn = PLAYER
        elif first == '2':
            turn = BOT
        else:
            print("Niepoprawny wybór. Domyślnie zaczyna Gracz.")
            turn = PLAYER
    else:
        human_vs_bot = False
        print("Wybrano: Bot vs Bot")
        turn = PLAYER

    print()

    tura = 0
    while True:
        tura += 1
        print(f"Tura {tura}")

        # Gracz vs Bot
        if human_vs_bot and turn == PLAYER:
            game.print_board()
            try:
                col = int(input(f"Twój ruch (0-{game.n_columns}): "))
            except ValueError:
                print("Podaj poprawny numer kolumny.")
                continue

            if col in game.valid_moves():
                game.make_move(col)
                print(f"Gracz wybiera kolumnę {col}\n")
                if game.winning_move(PLAYER):
                    game.print_board()
                    print("Gracz wygrał!")
                    break
                turn = BOT

        # Bot vs Bot
        else:
            start = time.time()
            maximizing = (turn == BOT)
            col, score, nodes = Player.alfabeta(game, Player.DEPTH, -math.inf, math.inf, maximizing)
            end = time.time()
            speed = int(nodes / (end - start)) if (end - start) > 0 else float('inf')

            game.make_move(col)
            if human_vs_bot:
                print(f"Bot wybiera kolumnę {col}")
                print(f"Głębokość: {Player.DEPTH}, Odwiedzone węzły: {nodes}, "
                      f"Szybkość: {speed} węzłów/s, Ocena: {score}\n")
            else:
                gracze = {PLAYER: "Bot 0", BOT: "Bot 1"}
                print(f"{gracze[turn]} wybiera kolumnę {col}")
                print(f"Głębokość: {Player.DEPTH}, Odwiedzone węzły: {nodes}, "
                      f"Szybkość: {speed} węzłów/s, Ocena: {score}\n")

            game.print_board()

            if game.winning_move(turn):
                if human_vs_bot:
                    print("Bot wygrał!")
                else:
                    gracze = {PLAYER: "Bot 0", BOT: "Bot 1"}
                    print(f"{gracze[turn]} wygrał!")
                break

            turn = 1 - turn

        # Sprawdzenie remisu
        if game.is_full():
            print("Remis!")
            break

    # Wyświetlenie historii ruchów
    print("Historia ruchów:")
    print([m for m in game.move_history])
