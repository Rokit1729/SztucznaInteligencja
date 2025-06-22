import math
import random
import copy

class Game:
    """
    Logika gry w stylu Connect Four:
    - n_rows: liczba wierszy
    - n_columns: liczba kolumn
    - winning_length: długość wymaganej sekwencji do wygranej
    - board: lista kolumn, każda kolumna to lista symboli (0 lub 1)
    - current_player: 0 lub 1, wskazuje kto jest na ruchu
    """
    def __init__(self, n_rows: int = 7, n_columns: int = 7, winning_length: int = 4):
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.winning_length = winning_length
        self.current_player = 0
        self.board = [[] for _ in range(self.n_columns)]
    
    def copy(self):
        """ Tworzy głęboką kopię stanu gry """
        new_game = Game(self.n_rows, self.n_columns, self.winning_length)
        new_game.current_player = self.current_player
        new_game.board = [list(col) for col in self.board]
        return new_game
    
    def valid_moves(self):
        """Zwraca listę indeksów kolumn, w które można jeszcze wrzucić pionek"""
        return [col for col in range(self.n_columns) if len(self.board[col]) < self.n_rows]
    
    def is_full(self):
        """Sprawdza, czy plansza jest pełna (remis)"""
        return all(len(self.board[col]) >= self.n_rows for col in range(self.n_columns))
    
    def drop_piece(self, column: int, symbol: int) -> bool:
        """Wrzuca pionek symbolu (0 lub 1) do danej kolumny. Zwraca False, jeśli ruch nielegalny."""
        if column not in self.valid_moves():
            return False
        self.board[column].append(symbol)
        return True
    
    def cell(self, row: int, col: int):
        """
        Zwraca zawartość komórki (row, col): 
        - jeśli w danej kolumnie jest mniej elementów niż row+1, to None, 
        - inaczej 0 lub 1.
        """
        if row < len(self.board[col]):
            return self.board[col][row]
        return None
    
    def winning_move(self, player: int) -> bool:
        """Sprawdza, czy dany gracz (0 lub 1) ma wygraną sekwencję długości winning_length."""
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

class Bot:
    """
    Bot z wagami A,B,C,D >= 0. Implementuje metodę alfabeta do wyboru ruchu,
    a ocena pozycji korzysta z wag tego bota.
    """
    def __init__(self, A: float, B: float, C: float, D: float, depth: int = 4):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.depth = depth  # głębokość przeszukiwania
        # można ewentualnie losować DEPTH, ale tu stała
    def make_move(self, game: Game) -> int:
        """
        Zwraca kolumnę do zagrania, licząc od 0, stosując alfabeta.
        current_player w obiekcie Game mówi, czy bot gra jako 0 czy 1.
        """
        maximizing = (game.current_player == 1)
        col, _, _ = self.alfabeta(game, self.depth, -math.inf, math.inf, maximizing)
        # Jeśli alfabeta zwróci None (np. gdy brak ruchów), możemy wybrać losowy ruch legalny:
        if col is None:
            valid = game.valid_moves()
            if not valid:
                return None
            return random.choice(valid)
        return col
    
    def alfabeta(self, game: Game, depth: int, alpha: float, beta: float, maximizingPlayer: bool):
        """
        Zwraca (wybrana_kolumna, ocena, liczba_przeszukanych_węzłów).
        Jeśli depth==0 lub stan terminalny → ocena heurystyczna.
        """
        nodes = 1
        # Stan terminalny?
        if game.is_full() or game.winning_move(0) or game.winning_move(1) or depth == 0:
            # Jeśli terminal lub osiągnięto depth=0 → ocena
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
                child.current_player = game.current_player
                # bot gra jako symbol 1?
                PlayerSymbol = 1
                child.drop_piece(col, PlayerSymbol)
                child.current_player = 1 - child.current_player
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
                child.current_player = game.current_player
                # bot gra jako symbol 0
                PlayerSymbol = 0
                child.drop_piece(col, PlayerSymbol)
                child.current_player = 1 - child.current_player
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
        Ocena fragmentu długości L = winning_length, lub dłuższego (w evaluate_window_longer).
        Zasady:
        - count_self == L → +A
        - L-1 + 1 puste → +B
        - L-2 + 2 puste → +C
        - 1 + (L-1) puste → +1
        Odpowiednio przeciwnik → minus te wartości.
        """
        score = 0.0
        opp = 1 - symbol
        L = len(window)
        count_self = window.count(symbol)
        count_opp = window.count(opp)
        count_empty = window.count(None)
        # Własne sekwencje
        if count_self == L:
            score += self.A
        elif count_self == L - 1 and count_empty == 1:
            score += self.B
        elif count_self == L - 2 and count_empty == 2:
            score += self.C
        elif count_self == 1 and count_empty == L - 1:
            score += 1.0
        # Przeciwnik
        if count_opp == L:
            score -= self.A
        elif count_opp == L - 1 and count_empty == 1:
            score -= self.B
        elif count_opp == L - 2 and count_empty == 2:
            score -= self.C
        elif count_opp == 1 and count_empty == L - 1:
            score -= 1.0
        return score
    
    def evaluate_window_longer(self, window: list, symbol: int) -> float:
        """
        Ocena okien długości L+1. Jeśli jest L-2 własnych i dwa puste na końcach → +D.
        Podobnie przeciwnik.
        """
        score = 0.0
        opp = 1 - symbol
        L = len(window)
        count_self = window.count(symbol)
        count_opp = window.count(opp)
        count_empty = window.count(None)
        # Własne
        if count_self == L - 2 and count_empty == 2:
            if window[0] is None and window[-1] is None:
                score += self.D
        # Przeciwnik
        if count_opp == L - 2 and count_empty == 2:
            if window[0] is None and window[-1] is None:
                score -= self.D
        return score
    
    def score_position(self, game: Game, symbol: int) -> float:
        """
        Sumuje oceny wszystkich okien w planszy dla danego symbolu.
        """
        score = 0.0
        R = game.n_rows
        C = game.n_columns
        L = game.winning_length
        # Okna poziome (długość L)
        for row in range(R):
            for col in range(C - L + 1):
                window = [game.cell(row, col + i) for i in range(L)]
                score += self.evaluate_window(window, symbol)
        # Okna pionowe (długość L)
        for col in range(C):
            for row in range(R - L + 1):
                window = [game.cell(row + i, col) for i in range(L)]
                score += self.evaluate_window(window, symbol)
        # Przekątne w dół-prawo (\) długość L
        for row in range(R - L + 1):
            for col in range(C - L + 1):
                window = [game.cell(row + i, col + i) for i in range(L)]
                score += self.evaluate_window(window, symbol)
        # Przekątne w górę-prawo (/) długość L
        for row in range(L - 1, R):
            for col in range(C - L + 1):
                window = [game.cell(row - i, col + i) for i in range(L)]
                score += self.evaluate_window(window, symbol)
        # Okna długości L+1 (evaluate_window_longer)
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
        # Przekątne w dół-prawo (\)
        for row in range(R - L_extra + 1):
            for col in range(C - L_extra + 1):
                window = [game.cell(row + i, col + i) for i in range(L_extra)]
                score += self.evaluate_window_longer(window, symbol)
        # Przekątne w górę-prawo (/)
        for row in range(L_extra - 1, R):
            for col in range(C - L_extra + 1):
                window = [game.cell(row - i, col + i) for i in range(L_extra)]
                score += self.evaluate_window_longer(window, symbol)
        return score

def apply_initial_moves(game: Game, moves: list[int]) -> bool:
    """
    Nakłada na grę sekwencję początkowych ruchów.
    Ruchy są naprzemiennie wykonywane przez graczy 0 i 1 według current_player.
    Zwraca False, jeśli w trakcie sekwencji nastąpił ruch nielegalny lub wygrana.
    """
    for move in moves:
        if move not in game.valid_moves():
            return False
        game.drop_piece(move, game.current_player)
        # Po każdym ruchu można sprawdzić wygraną; ale dla randomizacji startu
        # zwykle pomijamy: jeśli wygrywa ktoś w fazie randomizacji, 
        # to traktujemy to jako nieudana randomizacja i generujemy inną sekwencję.
        if game.winning_move(0) or game.winning_move(1):
            return False
        game.current_player = 1 - game.current_player
    return True

def generate_random_initial_moves(game: Game, max_moves: int = 6) -> list[int]:
    """
    Generuje losową sekwencję ruchów początkowych o długości od 1 do max_moves.
    Tylko ruchy legalne i przerywa, jeśli nie można pociągnąć dalej.
    Jeśli próbuje dłużej, odrzuca i generuje od nowa, aż uzyska poprawną sekwencję.
    """
    for _ in range(100):  # maksymalnie 100 prób
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
    return []  # oznacza: nie udało się wygenerować randomizacji

def play_game(bot1: Bot, bot2: Bot, starting_player: int, initial_moves: list[int] = None) -> int:
    """
    Przeprowadza jedną partię między bot1 i bot2.
    starting_player: 0 → bot1 zaczyna, 1 → bot2 zaczyna.
    Jeśli initial_moves podana i nie None: przed właściwą grą nakładamy te ruchy.
    Zwraca:
      1, jeśli wygrywa bot1
      2, jeśli wygrywa bot2
      0, jeśli remis
    """
    game = Game()
    # ustawienie current_player zgodnie ze starting_player:
    game.current_player = starting_player
    # Nakładamy initial_moves, jeśli podano
    if initial_moves:
        ok = apply_initial_moves(game, initial_moves)
        if not ok:
            # jeśli sekwencja doprowadziła do wygranej lub nielegalnego ruchu, traktujemy jak brak randomizacji
            pass
    # Rozgrywamy grę do końca, bez drukowania planszy
    while True:
        cp = game.current_player
        if cp == 0:
            col = bot1.make_move(game)
        else:
            col = bot2.make_move(game)
        if col is None:
            # brak ruchów → remis
            return 0
        # wykonanie ruchu
        legal = game.drop_piece(col, cp)
        if not legal:
            # teoretycznie nie powinno się zdarzyć (make_move powinno zwracać tylko legalne), 
            # ale jeśli tak, traktujemy jako remis
            return 0
        # sprawdzenie wygranej
        if game.winning_move(cp):
            # zwycięzca
            return 1 if cp == 0 else 2
        if game.is_full():
            return 0
        game.current_player = 1 - cp

def evolve_population(population: list[Bot],
                      iterations: int = 10000,
                      mutation_scale: float = 0.1,
                      tie_randomize_max_moves: int = 6,
                      randomize_prob: float = 0.05):
    """
    Główna pętla ewolucji:
    - population: lista Bot (początkowo 100 botów z losowymi wagami)
    - iterations: liczba par gier do rozegrania
    - mutation_scale: skala odchylenia standardowego przy mutacji wag
    - tie_randomize_max_moves: maksymalna długość sekwencji randomizacji startu w razie remisu
    - randomize_prob: z prawdopodobieństwem randomize_prob dokonujemy randomizacji startu 
      nawet przed normalnymi rozgrywkami, by eksplorować bardziej.
    """
    pop_size = len(population)
    for it in range(1, iterations + 1):
        # losowo dwa różne indexy
        i, j = random.sample(range(pop_size), 2)
        bot1 = population[i]
        bot2 = population[j]
        # wybór randomizacji startu z prawdopodobieństwem randomize_prob
        initial_moves = None
        if random.random() < randomize_prob:
            moves = generate_random_initial_moves(Game(), max_moves=tie_randomize_max_moves)
            if moves:
                initial_moves = moves
        # Rozgrywamy dwie gry: raz bot1 zaczyna, raz bot2 zaczyna
        result1 = play_game(bot1, bot2, starting_player=0, initial_moves=initial_moves)
        result2 = play_game(bot1, bot2, starting_player=1, initial_moves=initial_moves)
        # Obliczamy wyniki: +1 dla każdego zwycięstwa
        score1 = 1 if result1 == 1 else 0
        score1 += 1 if result2 == 1 else 0
        score2 = 1 if result1 == 2 else 0
        score2 += 1 if result2 == 2 else 0
        # Decyzja: jeśli jeden ma wyraźnie wyższy wynik, to on wygrywa starcie
        winner = None
        loser = None
        if score1 > score2:
            winner = bot1
            loser = bot2
        elif score2 > score1:
            winner = bot2
            loser = bot1
        else:
            # remis w serii → randomizacja startu i ponowna para gier
            # Generujemy randomizację:
            moves = generate_random_initial_moves(Game(), max_moves=tie_randomize_max_moves)
            if moves:
                result1 = play_game(bot1, bot2, starting_player=0, initial_moves=moves)
                result2 = play_game(bot1, bot2, starting_player=1, initial_moves=moves)
                score1 = 1 if result1 == 1 else 0
                score1 += 1 if result2 == 1 else 0
                score2 = 1 if result1 == 2 else 0
                score2 += 1 if result2 == 2 else 0
                if score1 > score2:
                    winner = bot1; loser = bot2
                elif score2 > score1:
                    winner = bot2; loser = bot1
                else:
                    # nadal remis → pomijamy tę parę, przejdź dalej
                    continue
            else:
                # nie udało się randomizacji → pomiń
                continue
        # Mamy zwycięzcę i przegranego: mutujemy przegranego na bazie wag zwycięzcy
        # W funkcji evolve_population, gdy mamy winner oraz loser:
        # Zakładamy, że winner.A to bieżąca wartość wagi A zwycięzcy.

        delta_A = random.uniform(-0.1, 0.1)  # E[delta_A] = 0
        new_A = max(0.0, winner.A * (1 + delta_A))

        # Podobnie dla pozostałych wag:
        delta_B = random.uniform(-0.1, 0.1)
        new_B = max(0.0, winner.B * (1 + delta_B))

        delta_C = random.uniform(-0.1, 0.1)
        new_C = max(0.0, winner.C * (1 + delta_C))

        delta_D = random.uniform(-0.1, 0.1)
        new_D = max(0.0, winner.D * (1 + delta_D))

        loser.A = new_A
        loser.B = new_B
        loser.C = new_C
        loser.D = new_D

        # Opcjonalnie możemy również dostosować głębokość, ale pomijamy tu.
        # Raport: wypisz wagi zwycięzcy (po mutacji? czy przed? Lepiej przed mutacją, bo mutowaliśmy przegranego)
        # Jeśli chcielibyśmy śledzić najlepsze wagi w populacji, można też co pewien czas znaleźć
        # bota o najwyższym współczynniku wygranych w ostatnim czasie, ale tu drukujemy zwycięskie wagi.
        print(f"Iteracja {it}: zwycięzca między indexami {i},{j}: wagi A,B,C,D = {winner.A:.4f}, {winner.B:.4f}, {winner.C:.4f}, {winner.D:.4f}")
    # Koniec pętli ewolucyjnej

def initialize_population(size: int = 100, init_scale: float = 1.0) -> list[Bot]:
    """
    Tworzy początkową populację botów z losowymi wagami:
    A, B, C, D ~ U(0, init_scale)
    """
    pop = []
    for _ in range(size):
        A = random.random() * init_scale
        B = random.random() * init_scale
        C = random.random() * init_scale
        D = random.random() * init_scale
        bot = Bot(A=A, B=B, C=C, D=D, depth=4)
        pop.append(bot)
    return pop

if __name__ == "__main__":
    # Parametry ewolucji:
    POP_SIZE = 10
    ITERATIONS = 1000000         # liczba par rozgrywek; można dostosować
    MUTATION_SCALE = 0.1      # skala mutacji wag
    TIE_RANDOM_MAX = 6        # maksymalna długość randomizacji w razie remisu
    RANDOMIZE_PROB = 0.1     # prawdopodobieństwo randomizacji startu również bez remisu
    
    # Inicjalizacja populacji
    population = initialize_population(size=POP_SIZE, init_scale=1000)
    # Uruchamiamy ewolucję
    evolve_population(population,
                      iterations=ITERATIONS,
                      mutation_scale=MUTATION_SCALE,
                      tie_randomize_max_moves=TIE_RANDOM_MAX,
                      randomize_prob=RANDOMIZE_PROB)
    # Po zakończeniu można na przykład zebrać najlepsze wagi według dodatkowych kryteriów,
    # albo po prostu zapisać całą populację do pliku itp.
