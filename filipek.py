import math
import random

class Game:
    """
    Logika gry typu Connect Four (7×7, wygrana przy 4 w linii).
    Plansza reprezentowana kolumnami: każda kolumna to lista ruchów (0 lub 1).
    current_player ∈ {0, 1}.
    """
    def __init__(self, n_rows: int = 7, n_columns: int = 7, winning_length: int = 4):
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.winning_length = winning_length
        self.current_player = 0
        # Reprezentacja: lista kolumn, każda kolumna to lista int (0 lub 1)
        self.board = [[] for _ in range(self.n_columns)]
    
    def copy(self):
        """Tworzy głęboką kopię obiektu Game."""
        new_game = Game(self.n_rows, self.n_columns, self.winning_length)
        new_game.current_player = self.current_player
        new_game.board = [list(col) for col in self.board]
        return new_game
    
    def valid_moves(self):
        """Zwraca listę indeksów kolumn, do których można wrzucić pionek."""
        return [col for col in range(self.n_columns) if len(self.board[col]) < self.n_rows]
    
    def is_full(self):
        """Sprawdza, czy plansza jest pełna (remis)."""
        return all(len(self.board[col]) >= self.n_rows for col in range(self.n_columns))
    
    def drop_piece(self, column: int, symbol: int) -> bool:
        """
        Wrzuca pionek symbolu (0 lub 1) do podanej kolumny.
        Zwraca False, jeżeli ruch nielegalny.
        """
        if column not in self.valid_moves():
            return False
        self.board[column].append(symbol)
        return True
    
    def cell(self, row: int, col: int):
        """
        Zwraca zawartość komórki (row, col):
        - jeśli w kolumnie jest mniej elementów niż row+1: None,
        - w przeciwnym razie 0 lub 1.
        """
        if row < len(self.board[col]):
            return self.board[col][row]
        return None
    
    def winning_move(self, player: int) -> bool:
        """
        Sprawdza, czy gracz `player` ma sekwencję długości winning_length:
        - poziomo, pionowo, przekątne (góra→prawo i dół→prawo).
        """
        L = self.winning_length
        R = self.n_rows
        C = self.n_columns
        # Poziomo
        for row in range(R):
            for col_start in range(C - L + 1):
                if all(self.cell(row, col_start + i) == player for i in range(L)):
                    return True
        # Pionowo
        for col in range(C):
            for row_start in range(R - L + 1):
                if all(self.cell(row_start + i, col) == player for i in range(L)):
                    return True
        # Przekątna w dół→prawo (\)
        for row_start in range(R - L + 1):
            for col_start in range(C - L + 1):
                if all(self.cell(row_start + i, col_start + i) == player for i in range(L)):
                    return True
        # Przekątna w górę→prawo (/)
        for row_start in range(L - 1, R):
            for col_start in range(C - L + 1):
                if all(self.cell(row_start - i, col_start + i) == player for i in range(L)):
                    return True
        return False

class Bot:
    """
    Bot z wagami A, B, C, D przechowywanymi jako int (>=0).
    Implementuje alfabeta z oceną heurystyczną opartą na własnych wagach.
    """
    def __init__(self, A: int, B: int, C: int, D: int, depth: int = 4):
        self.A = int(A)
        self.B = int(B)
        self.C = int(C)
        self.D = int(D)
        self.depth = depth  # głębokość przeszukiwania minimax/alfabeta
    
    def make_move(self, game: Game) -> int:
        """
        Zwraca kolumnę do zagrania. Używa alfabeta.
        Jeśli alfabeta zwróci None lub nielegalny ruch, wybiera losowy legalny ruch.
        """
        maximizing = (game.current_player == 1)
        col, _, _ = self.alfabeta(game, self.depth, -math.inf, math.inf, maximizing)
        if col is None or col not in game.valid_moves():
            valid = game.valid_moves()
            if not valid:
                return None
            return random.choice(valid)
        return col
    
    def alfabeta(self, game: Game, depth: int, alpha: float, beta: float, maximizingPlayer: bool):
        """
        Zwraca (wybrana_kolumna, ocena, liczba_przeszukanych_węzłów).
        Jeśli stan terminalny lub depth == 0, oblicza ocenę heurystyczną.
        """
        nodes = 1
        # Stan terminalny lub osiągnięcie głębokości
        if game.is_full() or game.winning_move(0) or game.winning_move(1) or depth == 0:
            symbol = 1 if maximizingPlayer else 0
            score = self.score_position(game, symbol)
            return (None, score, nodes)
        
        valid_cols = game.valid_moves()
        if not valid_cols:
            return (None, 0.0, nodes)
        
        if maximizingPlayer:
            value = -math.inf
            chosen_col = random.choice(valid_cols)
            for col in valid_cols:
                child = game.copy()
                # gra jako 1
                child.current_player = 1
                legal = child.drop_piece(col, 1)
                if not legal:
                    continue
                child.current_player = 0
                _, score_child, child_nodes = self.alfabeta(child, depth - 1, alpha, beta, False)
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
                child = game.copy()
                # gra jako 0
                child.current_player = 0
                legal = child.drop_piece(col, 0)
                if not legal:
                    continue
                child.current_player = 1
                _, score_child, child_nodes = self.alfabeta(child, depth - 1, alpha, beta, True)
                nodes += child_nodes
                if score_child < value:
                    value = score_child
                    chosen_col = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return (chosen_col, value, nodes)
    
    def evaluate_window(self, window: list, symbol: int) -> float:
        """
        Ocena fragmentu długości L = winning_length:
        - count_self == L  → +A
        - L-1 + 1 puste   → +B
        - L-2 + 2 puste   → +C
        - 1 + (L-1) puste → +1
        Odpowiednio przeciwnik: minus te wartości.
        Wagi A,B,C są int, konwertowane na float.
        """
        score = 0.0
        opp = 1 - symbol
        L = len(window)
        count_self = window.count(symbol)
        count_opp = window.count(opp)
        count_empty = window.count(None)
        if count_self == L:
            score += float(self.A)
        elif count_self == L - 1 and count_empty == 1:
            score += float(self.B)
        elif count_self == L - 2 and count_empty == 2:
            score += float(self.C)
        elif count_self == 1 and count_empty == L - 1:
            score += 1.0
        if count_opp == L:
            score -= float(self.A)
        elif count_opp == L - 1 and count_empty == 1:
            score -= float(self.B)
        elif count_opp == L - 2 and count_empty == 2:
            score -= float(self.C)
        elif count_opp == 1 and count_empty == L - 1:
            score -= 1.0
        return score
    
    def evaluate_window_longer(self, window: list, symbol: int) -> float:
        """
        Ocena okien długości L+1:
        Jeśli count_self == L-2 i dwa puste na końcach → +D,
        Analogicznie przeciwnik: -D.
        Waga D jest int, używana jako float.
        """
        score = 0.0
        opp = 1 - symbol
        L = len(window)
        count_self = window.count(symbol)
        count_opp = window.count(opp)
        count_empty = window.count(None)
        if count_self == L - 2 and count_empty == 2:
            if window[0] is None and window[-1] is None:
                score += float(self.D)
        if count_opp == L - 2 and count_empty == 2:
            if window[0] is None and window[-1] is None:
                score -= float(self.D)
        return score
    
    def score_position(self, game: Game, symbol: int) -> float:
        """
        Sumuje oceny wszystkich okien w planszy dla danego symbolu.
        """
        score = 0.0
        R = game.n_rows
        C = game.n_columns
        L = game.winning_length
        # Okna poziome
        for row in range(R):
            for col in range(C - L + 1):
                window = [game.cell(row, col + i) for i in range(L)]
                score += self.evaluate_window(window, symbol)
        # Okna pionowe
        for col in range(C):
            for row in range(R - L + 1):
                window = [game.cell(row + i, col) for i in range(L)]
                score += self.evaluate_window(window, symbol)
        # Przekątne \ długość L
        for row in range(R - L + 1):
            for col in range(C - L + 1):
                window = [game.cell(row + i, col + i) for i in range(L)]
                score += self.evaluate_window(window, symbol)
        # Przekątne / długość L
        for row in range(L - 1, R):
            for col in range(C - L + 1):
                window = [game.cell(row - i, col + i) for i in range(L)]
                score += self.evaluate_window(window, symbol)
        # Okna długości L+1
        L_extra = L + 1
        # Poziome
        for row in range(R):
            for col in range(C - L_extra + 1):
                window = [game.cell(row, col + i) for i in range(L_extra)]
                score += self.evaluate_window_longer(window, symbol)
        # Pionowe
        for col in range(C):
            for row in range(R - L_extra + 1):
                window = [game.cell(row + i, col) for i in range(L_extra)]
                score += self.evaluate_window_longer(window, symbol)
        # Przekątne \ długość L+1
        for row in range(R - L_extra + 1):
            for col in range(C - L_extra + 1):
                window = [game.cell(row + i, col + i) for i in range(L_extra)]
                score += self.evaluate_window_longer(window, symbol)
        # Przekątne / długość L+1
        for row in range(L_extra - 1, R):
            for col in range(C - L_extra + 1):
                window = [game.cell(row - i, col + i) for i in range(L_extra)]
                score += self.evaluate_window_longer(window, symbol)
        return score

def apply_initial_moves(game: Game, moves: list[int]) -> bool:
    """
    Nakłada sekwencję początkowych ruchów; sprawdza legalność i brak wygranej w trakcie randomizacji.
    Zwraca False, jeśli nielegalny ruch lub wygrana w fazie randomizacji → odrzucamy tę sekwencję.
    """
    for move in moves:
        if move not in game.valid_moves():
            return False
        game.drop_piece(move, game.current_player)
        if game.winning_move(0) or game.winning_move(1):
            return False
        game.current_player = 1 - game.current_player
    return True

def generate_random_initial_moves(game: Game, max_moves: int = 6) -> list[int]:
    """
    Generuje losową sekwencję początkowych ruchów długości od 1 do max_moves,
    próbując maks. 100 razy, aż nie doprowadzi do wygranej w trakcie randomizacji.
    Zwraca listę ruchów lub pustą listę, jeśli nie udało się wygenerować.
    """
    for _ in range(100):
        length = random.randint(1, max_moves)
        temp_game = game.copy()
        moves = []
        ok = True
        for _ in range(length):
            valid = temp_game.valid_moves()
            if not valid:
                ok = False
                break
            move = random.choice(valid)
            moves.append(move)
            if not apply_initial_moves(temp_game, [move]):
                ok = False
                break
        if ok:
            return moves
    return []

def play_game(bot1: Bot, bot2: Bot, starting_player: int, initial_moves: list[int] = None) -> int:
    """
    Przeprowadza jedną partię między bot1 i bot2.
    starting_player: 0 → bot1 zaczyna, 1 → bot2 zaczyna.
    Jeśli initial_moves nie None: nakłada tę sekwencję przed grą.
    Zwraca 1 jeśli wygra bot1, 2 jeśli bot2, 0 jeśli remis.
    """
    game = Game()
    game.current_player = starting_player
    if initial_moves:
        _ = apply_initial_moves(game, initial_moves)
    while True:
        cp = game.current_player
        if cp == 0:
            col = bot1.make_move(game)
        else:
            col = bot2.make_move(game)
        if col is None:
            return 0
        legal = game.drop_piece(col, cp)
        if not legal:
            return 0
        if game.winning_move(cp):
            return 1 if cp == 0 else 2
        if game.is_full():
            return 0
        game.current_player = 1 - cp

def initialize_population_custom(size_each: int = 10) -> list[Bot]:
    """
    Tworzy populację: 5 zestawów wag, po `size_each` botów każdego zestawu.
    """
    base_weights = [
        (100, 50, 10, 30),
        (1000, 500, 100, 300),
        (1000000, 5000, 1000, 3000),
        (1000000, 500, 100, 300),
        (40, 30, 20, 45),
    ]
    pop = []
    for (A0, B0, C0, D0) in base_weights:
        for _ in range(size_each):
            bot = Bot(A=A0, B=B0, C=C0, D=D0, depth=4)
            pop.append(bot)
    return pop

def evolve_population(population: list[Bot],
                      iterations: int = 10000,
                      tie_randomize_max_moves: int = 6,
                      randomize_prob: float = 0.05,
                      output_filename: str = "zwycięskie_wagi.txt"):
    """
    Ewolucja populacji:
    - Po każdej serii starcia (2 mecze + ewentualny rematch) dopisujemy do pliku tylko wagi bota, który wygrał serię.
    - Plik otwierany w trybie 'a', bez nadpisywania wcześniejszych wpisów.
    - Jeśli seria zakończy się remisem nawet po rematch, nie zapisujemy nic i pomijamy mutację tej pary w tej iteracji.
    - Po ustaleniu zwycięzcy serii: zapisujemy f.write(f"{winner.A} {winner.B} {winner.C} {winner.D}\n"), a następnie mutujemy przegranego.
    """
    pop_size = len(population)

    # Otwieranie pliku w trybie append.
    # Jeśli konieczne jest wyczyszczenie pliku przed pierwszym uruchomieniem, 
    # należy przed wywołaniem tej funkcji wykonać: open(output_filename, 'w').close()
    try:
        f = open(output_filename, 'a', encoding='utf-8')
    except Exception as e:
        print(f"Błąd otwarcia pliku do dopisywania: {e}")
        return

    try:
        for it in range(1, iterations + 1):
            # Wybór dwóch różnych botów
            i, j = random.sample(range(pop_size), 2)
            bot1 = population[i]
            bot2 = population[j]

            # Opcjonalna randomizacja startu z pewnym prawdopodobieństwem
            initial_moves = None
            if random.random() < randomize_prob:
                moves = generate_random_initial_moves(Game(), max_moves=tie_randomize_max_moves)
                if moves:
                    initial_moves = moves

            # --- Pierwszy mecz: bot1 zaczyna ---
            result1 = play_game(bot1, bot2, starting_player=0, initial_moves=initial_moves)
            # --- Drugi mecz: bot2 zaczyna ---
            result2 = play_game(bot1, bot2, starting_player=1, initial_moves=initial_moves)

            # Obliczenie wyników serii
            score1 = (1 if result1 == 1 else 0) + (1 if result2 == 1 else 0)
            score2 = (1 if result1 == 2 else 0) + (1 if result2 == 2 else 0)

            winner = None
            loser = None
            if score1 > score2:
                winner, loser = bot1, bot2
            elif score2 > score1:
                winner, loser = bot2, bot1
            else:
                # Remis w serii → rematch z randomizacją startu
                moves = generate_random_initial_moves(Game(), max_moves=tie_randomize_max_moves)
                if moves:
                    # Rematch 1: bot1 zaczyna
                    rem1 = play_game(bot1, bot2, starting_player=0, initial_moves=moves)
                    # Rematch 2: bot2 zaczyna
                    rem2 = play_game(bot1, bot2, starting_player=1, initial_moves=moves)

                    # Ponowne obliczenie wyników rematch-serii
                    score1 = (1 if rem1 == 1 else 0) + (1 if rem2 == 1 else 0)
                    score2 = (1 if rem1 == 2 else 0) + (1 if rem2 == 2 else 0)
                    if score1 > score2:
                        winner, loser = bot1, bot2
                    elif score2 > score1:
                        winner, loser = bot2, bot1
                    else:
                        # Nadal remis po rematch → pomijamy zapis i mutację
                        continue
                else:
                    # Brak poprawnej randomizacji → pomijamy zapis i mutację
                    continue

            # Jeśli tu mamy winner i loser, to zapisujemy wyłącznie zwycięskie wagi:
            try:
                f.write(f"{winner.A} {winner.B} {winner.C} {winner.D}\n")
                f.flush()
            except Exception as e:
                print(f"Błąd zapisu zwycięskich wag w iteracji {it}: {e}")
                # Pomimo błędu zapisu, przeprowadzimy mutację w pamięci, ale warto zwrócić uwagę na I/O

            # Mutacja przegranego: relatywna ±10%, zaokrąglona do int, klipowana do >=0
            delta_A = random.uniform(-0.1, 0.1)
            delta_B = random.uniform(-0.1, 0.1)
            delta_C = random.uniform(-0.1, 0.1)
            delta_D = random.uniform(-0.1, 0.1)

            new_A = int(round(winner.A * (1 + delta_A)))
            new_B = int(round(winner.B * (1 + delta_B)))
            new_C = int(round(winner.C * (1 + delta_C)))
            new_D = int(round(winner.D * (1 + delta_D)))
            if new_A < 0: new_A = 0
            if new_B < 0: new_B = 0
            if new_C < 0: new_C = 0
            if new_D < 0: new_D = 0

            loser.A = new_A
            loser.B = new_B
            loser.C = new_C
            loser.D = new_D

            # Raport w konsoli o serii (opcjonalnie)
            print(f"Iteracja {it}: starcie botów index {i} i {j}, zwycięzca wagi = "
                  f"{winner.A}, {winner.B}, {winner.C}, {winner.D}")

    finally:
        f.close()

def main():
    # Parametry ewolucji:
    SIZE_EACH = 5           # liczba botów dla każdego zestawu wag
    POP_SIZE = 5 * SIZE_EACH # 50 botów
    ITERATIONS = 5000        # liczba par rozgrywek; dostosować wg mocy obliczeniowej
    TIE_RANDOM_MAX = 6       # maks. długość randomizacji przy remisie
    RANDOMIZE_PROB = 0.05    # prawdopodobieństwo randomizacji startu w normalnym meczu
    OUTPUT_FILE = "wagi.txt"

    # Jeśli chcesz wyczyścić plik na starcie, odkomentuj poniższą linię:
    # open(OUTPUT_FILE, 'w').close()

    # Inicjalizacja populacji
    population = initialize_population_custom(size_each=SIZE_EACH)
    print(f"Inicjalizacja: {len(population)} botów.")

    # Uruchomienie ewolucji; plik OUTPUT_FILE będzie dopisywany tylko zwycięskimi wagami serii
    evolve_population(population,
                      iterations=ITERATIONS,
                      tie_randomize_max_moves=TIE_RANDOM_MAX,
                      randomize_prob=RANDOMIZE_PROB,
                      output_filename=OUTPUT_FILE)

    print(f"Ewolucja zakończona. Zapisano wagi zwycięzców serii do pliku: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
