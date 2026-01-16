import numpy as np

M = 1000


def macierz(c, A, b, signs):
    #c = - współczynniki funkcji celu (zysk z x1, x2, x3)
    #A = macierz 3x3 - zmienne przy ograniczeniach lewa strona
    #b = - prawostronne wartości ograniczeń
    #signs = ["<=", ">=", "="] - typy ograniczeń
    
    m, n = A.shape          #m = liczba wierszy macierzy A = 3 (ograniczenia) n kolumny
    #zasada
    # <= + 1 zmienna luzu
    # >= - 1 zmienna nadmiarowa + 1 zmienna sztuczna
    # =  + 1 zmienna sztuczna
    # w przypadku zmiennych sztucznych (>> lub =) w funkcji celu dajemy funkcje np -M*x_n+1 (M duża liczba) kary

        #obliczanie liczby nowych zmiennych
    luzowa = sum(1 for s in signs if s == "<=")  #dodaj liczbe 1 dla kazdego ograniczenia typu <=
    nadmiarowa = sum(1 for s in signs if s == ">=") #dodaj liczbe 1 dla kazdego ograniczenia typu >=
    sztuczna = sum(1 for s in signs if s in (">=", "=")) #dodaj liczbe 1 dla kazdego ograniczenia typu >= lub =

    wszystkie = n + luzowa + nadmiarowa + sztuczna #suma podstawowych zmiennych + zmiennych sztucznych + zmiennych nadmiarowych
    #tworzenie nowych macierzy i wektorów (oryginalne macierze np. A są za małe)
    A_new = np.zeros((m, wszystkie)) # wiersze , kolumny
    C_new = np.zeros(wszystkie) #kolumny
    C_new[:n] = c           #wypelnienie wspl f celu z zdania na poczatek wektora C_new [3, 5, 2, 0, 0, 0, 0]
    names = [f"x{i+1}" for i in range(wszystkie)] # tworzy liste nazw zmiennych x1,x2,...,xn
    indeksy_zmiennych_bazowych = [] #lista numerów kolumn, które są obecnie w bazie

    #etap 1 dodawania zmiennych luzu, nadmiarowych i sztucznych

    col = n    # bo pierwsze n zajete przez zmienne podstawowe x1,x2,x3
    for i in range(m):  # 0,1,2 - iteracja po wierszach macierzy A, i dodaje sie najpierw w 1 wierszu potem 2 itd
        A_new[i, :n] = A[i]  # kopiuje n elementow z kazdego i-tego wiersza
                            

        if signs[i] == "<=":    #jesli pierwsze ograniczenie to <= to...
            A_new[i, col] = 1   # dodajemy zmienna luzu (slack variable) 0,1,2,3 więc w 4 kolumnie (indeks 3) dajemy 1
            indeksy_zmiennych_bazowych.append(col)
            col += 1  # przesuwamy sie do nastepnej kolumny (3->4)

        elif signs[i] == ">=":
                #dodawanie zarowno ujemnej nadmiarowej jak i dodatniej sztucznej
            A_new[i, col] = -1  # odejmujemy zmienna nadmiarowa, 
            col += 1 # przesuwamy sie do nastepnej kolumny  

            A_new[i, col] = 1  # dodajemy zmienna sztuczna
            C_new[col] = -M    #dajemy karę -M w funkcji celu

            indeksy_zmiennych_bazowych.append(col)      
            col += 1

        elif signs[i] == "=":
            A_new[i, col] = 1
            C_new[col] = -M

            indeksy_zmiennych_bazowych.append(col)
            col += 1

    return A_new, b, C_new, indeksy_zmiennych_bazowych, names # zwraca nowa macierz A, wektor b, wektor C, indeksy zmiennych bazowych, nazwy zmiennych

    #etap 2: Simpleks
# =====================================================
def simpleks(A, b, C, indeksy_zmiennych_bazowych, names):
    m, n = A.shape
    T = np.hstack((A, b.reshape(-1, 1)))    #dodaje kolumne b do tabeli simpleksowej , T to cała tabela simpleksowa 
    #polacz poziomo a i odwrot b w kolumne w T
    
    
    while True:
        Zj = np.zeros(n + 1)    #wartości Zj +1 bo jeszcze kolumna b
        for i in range(m):
            Zj = Zj + C[indeksy_zmiennych_bazowych[i]] * T[i]   #obliczanie Zj: suma kosztów bazowych (wspolczynnikow wierszów dodanych) razy wiersze tabeli simpleksowej (Współczynniki zmiennych w tabeli simpleksowej)
        Cj_Zj = np.append(C, 0) - Zj   #c i 0     #Cj - Zj : różnica między wspolczynnikami funkcji celu a Zj czyli suma kosztów bazowych razy wiersze tabeli simpleksowej
        #
        print("Cj - Zj:", np.round(Cj_Zj[:-1], 1).tolist())# bez ostatniej bo nie testuhemy kolumny b
        #
        if np.all(Cj_Zj[:-1] <= 0):         #jeśli wszystkie wartości Cj - Zj są mniejsze lub równe zero, to rozwiązanie jest optymalne, jesli nie to kontynuujemy
            break
        
        pivot_col = np.argmax(Cj_Zj[:-1])           #wchodzaca, wybór kolumny przestawiania (kolumna z największą wartością Cj - Zj)
        ratios = []         
        for i in range(m):
            if T[i, pivot_col] > 0:     #jeśli współczynnik w kolumnie w której była największa wartość Cj - Zj jest > 0, 
                ratios.append(T[i, -1] / T[i, pivot_col]) #to kolumne b dzielimy przez kolumne gdzie byl wspolczynnik wybrany
            else:
                ratios.append(np.inf)           #jeśli współczynnik jest <= 0, to stosunek jest nieskończony, dodaje znak nieskończoności do listy stosunków
                #dzielenie przez ujemna dalo by rozwiązania spoza obszaru rozwiazan
        pivot_row = np.argmin(ratios)       #wychodzaca, wybór wiersza przestawiania (wiersz z najmniejszym stosunkiem w kolumnie b po dzieleniu)
        
        pivot = T[pivot_row, pivot_col] # na przecieciu jest pivot, w tej kolumnie nalezy wyzerowac wszystkie inne liczby poza pivotem
        T[pivot_row] /= pivot # dzielenie wiersza przestawiania przez pivot, aby pivot stał się 1
        
        #eliminacja gaussowskaae
        
        for i in range(m): #czyli robimy jedynke w pivocie i potem odejmujemy od innych wierszy ten wiersz razy jeden
            if i != pivot_row: #jesli wiersz 0 != wiersz pivot rob  to , potem 1 =1 wiec pomin, i 2=/=1 
                wspolczynnik = T[i, pivot_col]   #współczynnik do wyzerowania 
                wiersz_pivot = T[pivot_row]          #wiersz przestawiania z nasza jedynką
                do_odjecia = wspolczynnik * wiersz_pivot  #obliczanie wartości do odjęcia, aby wyzerować współczynnik w kolumnie przestawiania
                T[i] = T[i] - do_odjecia  #odejmowanie wartości od wiersza, aby wyzerować współczynnik w kolumnie przestawiania
        indeksy_zmiennych_bazowych[pivot_row] = pivot_col # zmiana nazwy z zmienna wychodzaca na wchodzaca
    
    solution = np.zeros(n)
# sprawdza zmienne bazowe w jakiej są kolejności w tabeli simpleksowej i
# przypisuje ich wartości z kolumny b do odpowiednich miejsc w wektorze solution, tak by byly miedzy zerami w odpowiednich miejscach
    for numer_wiersza in range(m): # dla każdego wiersza w tabeli simpleksowej
        numer_zmiennej = indeksy_zmiennych_bazowych[numer_wiersza] #ustala miejsce zmiennej bazowej w tabeli simpleksowej
        
        wartosc_zmiennej = T[numer_wiersza, -1] #okresla wartosc w kolumnie b dla tego wiersza
                            #-1 bo bedzie pionowo
        solution[numer_zmiennej] = wartosc_zmiennej #przypisuje z b do odp. miejsca w wektorze rozw.
    


    # KROK 3: Oblicz funkcję celu Z
    value = np.dot(C, solution) #np.dot to iloczyn skalarny 
    # czyli mnozymy rozwiazanie przez wspolczynniki funkcji celu

    return solution, value

def solve_lp_simple(c, A, b, signs, sense='max'):
    c_copy = c.copy()
    if sense == 'min':
        c_copy = -c_copy  #Simpleks tylko maksymalizuje jesli min Z = max(-Z)
            #więc zmieniamy znaki współczynników funkcji celu

    transformacja = macierz(c_copy, A, b, signs) # python po przeslaniu c_copy, A, b, signs wykonuje poprzednio zadaane kroki
    # robi A_new, C_new, baza, names  za pomoca wprowadzonych danych w funkcji macierz
    # BEZPOŚREDNI DOSTĘP
    A2 = transformacja[0]     # macierz A rozszerzona
    b2 = transformacja[1]     # wektor b  
    C2 = transformacja[2]     # współczynniki C rozszerzone
    base = transformacja[3]   # zmienne bazowe startowe
    names = transformacja[4]  # nazwy zmiennych    
    print("Macierz po transformacji:")
    print(A2)
    print(C2)
    print("Baza startowa:", [names[i] for i in base])
    
    solution, value = simpleks(A2, b2, C2, base, names)
    
    if sense == 'min':
        value = -value
    
    return solution, value, names

# PRZYKŁAD UŻYCIA
C = np.array([3, 5, 2])
A = np.array([[1, 1, 1], [2, 1, -1], [1, -1, 1]])
b = np.array([7, 2, 3])
signs = ['<=', '>=', '=']

# tutaj zmieniamy min/max
solution, value, names = solve_lp_simple(C, A, b, signs, sense='max')
print("\n Po nastepujacych iteracjach, wszystkie liczby w CK-ZJ są mniejsze lub równe zero, więc rozwiązanie jest optymalne")
print("\nROZWIĄZANIE:")
for i in range(len(solution)):
    print(f"{names[i]} = {solution[i]:.2f}")
print(f"wartosc funkcji celu przy zadanych warunkach wynosi = {value:.2f}")
