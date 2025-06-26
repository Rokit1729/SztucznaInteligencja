import math
import random
import time

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
    DEPTH = 6

    @staticmethod
    def alfabeta(game: Game, depth: int, alpha: float, beta: float, maximizingPlayer: bool):
        """
        Wykonuje algorytm minimax z cięciami alfa-beta.

        Args:
            game (Game): bieżący stan gry.
            depth (int): pozostała głębokość przeszukiwania.
            alpha (float): najlepsza dotychczas znaleziona wartość dla gracza maksymalizującego.
            beta (float): najlepsza dotychczas znaleziona wartość dla gracza minimalizującego.
            maximizingPlayer (bool): True, jeśli ruch maksymalizuje ocenę (gracz 1), False dla gracza 0.

        Returns:
            tuple[int|None, float, int]: (kolumna, wartość oceny).
        """
        
        # 1) Na każdym węźle najpierw sprawdza czy może już wygrał
        if Player.winning_move(game, 1 if maximizingPlayer else 0):
            if maximizingPlayer:
                return (None, math.inf)
            else:
                return (None, -math.inf)
        
        # 2) Jeżeli doszło do remisu
        valid_cols = Player.valid_moves(game)
        if not valid_cols:
            return (None, 0.0)
        
        # 3) Na najgłębszym węźle dokonuje walidacji
        if depth == 0:
            score = Player.score_position(game, 1 if maximizingPlayer else 0)
            return (None, score)

        # 4) Na pozostałych węzła dokonuje klasycznej alfabety z prunningiem
        if maximizingPlayer:
            value = -math.inf
            chosen_col = random.choice(valid_cols)
            for col in valid_cols:
                child = Player.copy(game)
                Player.drop_piece(child, col, 1)
                _, score_child = Player.alfabeta(child, depth - 1, alpha, beta, False)
                if score_child > value:
                    value = score_child
                    chosen_col = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return (chosen_col, value)
        else:
            value = math.inf
            chosen_col = random.choice(valid_cols)
            for col in valid_cols:
                child = Player.copy(game)
                Player.drop_piece(child, col, 0)
                _, score_child = Player.alfabeta(child, depth - 1, alpha, beta, True)
                if score_child < value:
                    value = score_child
                    chosen_col = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return (chosen_col, value)

    @staticmethod
    def copy(game: Game) -> Game:
        """
        Tworzy głęboką kopię obiektu Game:
        - Kopiuje wymiary (n_rows, n_columns, winning_length) i current_player.
        - Kopiuje historię ruchów (listę).
        - Kopiuje każdą kolumnę planszy jako nową listę.
        - Zwraca nowy obiekt Game.
        """
        new_game = Game(game.n_rows, game.n_columns, game.winning_length)
        new_game.current_player = game.current_player
        new_game.move_history = list(game.move_history)
        new_game.board = [list(col) for col in game.board]
        return new_game

    @staticmethod
    def valid_moves(game: Game):
        """
        Zwraca listę indeksów kolumn, do których można wciąż wrzucić pionek.
        """
        return [col for col in range(game.n_columns) if len(game.board[col]) < game.n_rows]

    @staticmethod
    def is_full(game: Game):
        """
        Sprawdza, czy plansza jest w pełni zapełniona.
        """
        return all(len(game.board[col]) >= game.n_rows for col in range(game.n_columns))

    @staticmethod
    def cell(game: Game, row: int, col: int):
        """
        Zwraca wartość komórki na danym wierszu i kolumnie:
        - Jeśli w danej kolumnie nie ma jeszcze takiego wiersza (lista krótsza), zwraca None.
        - W przeciwnym razie zwraca wartość symbolu (0 lub 1).
        """
        if row < len(game.board[col]):
            return game.board[col][row]
        return None

    @staticmethod
    def drop_piece(game: Game, column: int, symbol: int):
        """
        Wrzuca pionek o danym symbolu (0 lub 1) do wybranej kolumny:
        - Jeśli kolumna jest pełna lub niedopuszczalna, zwraca False.
        - W przeciwnym razie dodaje symbol na szczyt kolumny i zwraca True.
        """
        
        if column not in Player.valid_moves(game):
            return False
        game.board[column].append(symbol)
        return True

    @staticmethod
    def winning_move(game: Game, player: int):
        """
        Sprawdza, czy dany gracz (player: 0 lub 1) ma zwycięską sekwencję długości winning_length:
        - Sprawdza wszystkie możliwe kierunki:
            1) Poziomo (rząd stały, kolumna zmienna)
            2) Pionowo (kolumna stała, rząd zmienny)
            3) Przekątna w dół-prawo (\)
            4) Przekątna w górę-prawo (/)
        - Dla każdego możliwego startu sekwencji długości L sprawdza, czy wszystkie komórki są równe player.
        - Jeśli znajdzie sekwencję, zwraca True, w przeciwnym razie False.
        """
        
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
        """
        Ocena krótkiego "okna" (lista długości L) pod kątem określonego symbolu:
        - window: lista wartości [0,1,None]; symbol: 0 lub 1
        - count_self = liczba wystąpień naszego symbolu w oknie
        - count_opp = liczba wystąpień przeciwnika
        - count_empty = liczba None (wolnych miejsc)
        - Przyznaje punkty:
            1) 3 własne + 1 puste: +2575
            2) 2 własne + 2 puste: +50
            3) 1 własne + 3 puste: +1
            4) Przeciwnik 3 + 1 puste: -2575
            5) Przeciwnik 2 + 2 puste: -50
            6) Przeciwnik 1 + 3 puste: -1
        - Zwraca wartość score (dodatnią lub ujemną).
        """
        
        score = 0
        opp = 1 - symbol
        L = len(window)

        count_self = window.count(symbol)
        count_opp = window.count(opp)
        count_empty = window.count(None)

        # 1) 3 w linii + 1 puste
        if count_self == L - 1 and count_empty == 1:
            score += 2575
        # 2) 2 w linii + 2 puste
        elif count_self == L - 2 and count_empty == 2:
            score += 50
        # 3) 1 w linii i 3 puste
        elif count_self == 1 and count_empty == L - 1:
            score += 1

        # 4) Przeciwnik 3 w linii + 1 puste
        if count_opp == L - 1 and count_empty == 1:
            score -= 2575
        # 5) Przeciwnik 2 w linii + 2 puste
        elif count_opp == L - 2 and count_empty == 2:
            score -= 50
        # 6) Przeciwnik 1 w linii i 3 puste
        elif count_opp == 1 and count_empty == L - 1:
            score -= 1

        return score

    @staticmethod
    def evaluate_window_longer(window, symbol):
        """
        Ocena dłuższego "okna" (np. długości winning_length+1):
        - Sprawdza sytuacje, gdy mamy 3 w linii i obydwa boki puste:
            1) Jeśli my mamy count_self == L-2 i oba końce None: +250
            2) Jeśli przeciwnik ma count_opp == L-2 i oba końce None: -250
        - Przydatne do wykrywania ukrytych możliwości („wąskie” okna z pustymi bokami).
        """
        
        score = 0
        opp = 1 - symbol
        L = len(window)

        count_self = window.count(symbol)
        count_opp = window.count(opp)
        count_empty = window.count(None)

        # 1) Jeżeli mamy 3 w linii i dwa po bokach puste
        if count_self == L - 2 and count_empty == 2:
            if window[0] is None and window[-1] is None:
                score += 250

        # 2) Przeciwnik ma 3 w linii i dwa po bokach puste
        if count_opp == L - 2 and count_empty == 2:
            if window[0] is None and window[-1] is None:
                score -= 250

        return score

    @staticmethod
    def score_position(game: Game, symbol: int):
        """
        Sumuje oceny wszystkich możliwych "okien" dla danego symbolu na planszy:
        - Przechodzi przez wszystkie kierunki i pozycje startowe:
            1) Okna poziome długości winning_length
            2) Okna pionowe długości winning_length
            3) Okna przekątne w dół-prawo (\) długości winning_length
            4) Okna przekątne w górę-prawo (/) długości winning_length
        - Dodatkowo:
            5) Okna poziome długości winning_length+1 (wywołuje evaluate_window_longer)
            6) Okna pionowe długości winning_length+1
            7) Okna przekątne w dół-prawo długości winning_length+1
            8) Okna przekątne w górę-prawo długości winning_length+1
        - Dla każdego okna zbiera odpowiednią ocenę i sumuje do całkowitego wyniku.
        - Zwraca skumulowany wynik (im wyższy, tym lepiej dla danego symbolu).
        """
        
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

        # Okna dłuższe o 1 (winning_length + 1)
        L_extra = L + 1

        # 5) Okna poziome - długości L+1
        for row in range(R):
            for col in range(C - L_extra + 1):
                window = [Player.cell(game, row, col + i) for i in range(L_extra)]
                score += Player.evaluate_window_longer(window, symbol)

        # 6) Okna pionowe - długości L+1
        for col in range(C):
            for row in range(R - L_extra + 1):
                window = [Player.cell(game, row + i, col) for i in range(L_extra)]
                score += Player.evaluate_window_longer(window, symbol)

        # 7) Okna przekątne w dół-prawo (\) - długości L+1
        for row in range(R - L_extra + 1):
            for col in range(C - L_extra + 1):
                window = [Player.cell(game, row + i, col + i) for i in range(L_extra)]
                score += Player.evaluate_window_longer(window, symbol)

        # 8) Okna przekątne w górę-prawo (/) - długości L+1
        for row in range(L_extra - 1, R):
            for col in range(C - L_extra + 1):
                window = [Player.cell(game, row - i, col + i) for i in range(L_extra)]
                score += Player.evaluate_window_longer(window, symbol)

        return score

    @staticmethod
    def make_move(game: Game) -> int:
        """
        Na podstawie dotychczasowej historii ruchów wybiera kolejny ruch.

        1. Walidacja zakresów i przepełnienia kolumn.
        2. Odtworzenie stanu planszy z historii.
        3. Sprawdzenie natychmiastowego zwycięstwa.
        4. Wywołanie alfabety z domyślną głębokością DEPTH.
        """
        
        # Walidacja historii
        for col in game.move_history:
            if col < 0 or col >= game.n_columns:
                raise ValueError(f"Niepoprawny ruch: kolumna {col} poza zakresem 0..{game.n_columns-1}")
        for col in range(game.n_columns):
            if game.move_history.count(col) > game.n_rows:
                raise ValueError(f"Niepoprawny ruch: kolumna {col} przepełniona")
        
        # Odtwarzamy ruchy
        new_game = Game(game.n_rows, game.n_columns, game.winning_length)
        for idx, col in enumerate(game.move_history):
            player = idx % 2
            Player.drop_piece(new_game, col, player)
        new_game.move_history = list(game.move_history)
        current = len(game.move_history) % 2
        new_game.current_player = current

        # Natychmiastowe zwycięstwo
        for col in Player.valid_moves(new_game):
            child = Player.copy(new_game)
            Player.drop_piece(child, col, current)
            if Player.winning_move(child, current):
                return col

        col, _ = Player.alfabeta(new_game, Player.DEPTH, -math.inf, math.inf, maximizingPlayer=(current == 1))
        return col


def draw_board(history, rows, columns):
    """
    Rysuje planszę na podstawie historii ruchów:
    - history: lista numerów kolumn w kolejności wrzucania pionków.
    - rows, columns: wymiary planszy.
    """
    
    board_copy = [[] for _ in range(columns)]

    player = 0
    for i in history:
        board_copy[i].append(str(player))
        player = 1 - player
    
    # Uzupełnienie pustymi miejscami
    for column in board_copy:
        while len(column) < rows:
            column.append("-")
    
    # Transpozycja: tworzymy wiersze for rysowania
    board = [[] for _ in range(rows)]
    row = 0
    for i in board:
        for j in range(columns):
            i.append(board_copy[j][row])
        row += 1
    
    # Rysujemy od góry w dół
    for row in board[::-1]:
        print('| ' + ' | '.join(row) + ' |')

if __name__ == "__main__":
    game = Game()
    print("Podaj historię ruchów jako listę numerów kolumn, oddzielonych przecinkami (np. 0,3,1,1,0,1):")
    wejscie = input("Historia ruchów: ")
    try:
        historia = [int(x.strip()) for x in wejscie.split(",") if x.strip()]
        game.move_history = historia
        start = time.time()
        next_col = Player.make_move(game)
        end = time.time()
        draw_board(historia, game.n_rows, game.n_columns)
        print(f"Zalecany następny ruch: kolumna {next_col}")
        print(f"Czas ruchu: {end - start:.4f} sek.")
    except ValueError as e:
        print(f"Błąd: {e}")
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd: {e}")