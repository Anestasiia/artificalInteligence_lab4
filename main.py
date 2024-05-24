import random
import matplotlib.pyplot as plt
import numpy as np
import warnings
from matplotlib import MatplotlibDeprecationWarning

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

# створює випадкову мапу
def generate_map():
    n = 26
    distances = [[random.randint(10, 100) for j in range(n)] for i in range(n)]
    with open("map.txt", "w") as f:
        f.write(str(n) + "\n")
        for i in range(n):
            f.write(" ".join(str(x) for x in distances[i]) + "\n")

# зчитує мапу з файлу
def read_map():
    with open("map.txt", "r") as f:
        n = int(f.readline().strip())
        distances = []
        for i in range(n):
            row = [int(x) for x in f.readline().split()]
            distances.append(row)
    return n, distances
# повернення кортежу зі значенням кількості міст та списку списів відстаней

def print_map(n):
    print("Кількість міст: ", n)
    print("Матриця відстаней:")
    for i in range(n):
        print(Distances[i])

# функція для оптимізації маршруту мурашиним алгоритмом
def ant_colony_optimization(distance_matrix, num_ants):
    n = len(distance_matrix)
    ant_locations = random.sample(range(n), num_ants)
    pheromone_matrix = [[PheromoneInitial for _ in range(n)] for _ in range(n)]
# ініціалізація матриці феромонів, де кожен елемент відповідає феромону між містами i та j
    shortest_distance = float('inf')
    shortest_path = []

    for iter in range(IterationNumber):
        for i in range(num_ants):
            current_location = ant_locations[i]
            allowed_locations = [j for j in range(n) if j != current_location]
# створення списку доступних місць для переходу, ігноруючи поточне місцезнаходження
            next_location = None

            if random.random() < q0:
                # обираємо місце з найвищим рівнем феромону
                pheromone_values = [(j, pheromone_matrix[current_location][j]) for j in allowed_locations]
                # список містить (місце, рівень феромону) для всіх доступних місць
                pheromone_values.sort(key=lambda x: x[1], reverse=True)
                next_location = pheromone_values[0][0]
            else:
                # обираємо місце випадковим чином з ймовірністю, залежною від рівня феромону та відстані
                probabilities = [(j, (pheromone_matrix[current_location][j]) ** alpha *
                                  (1 / distance_matrix[current_location][j]) ** beta) for j in allowed_locations]
                # обчислюємо ймовірності для кожного доступного місця за допомогою формули мурашиного алгоритму
                probabilities_sum = sum(p[1] for p in probabilities)
                # ймовірності нормалізуються, щоб утворювати ймовірнісний розподіл
                probabilities = [(p[0], p[1] / probabilities_sum) for p in probabilities]
                probabilities.sort(key=lambda x: x[1], reverse=True)
                random_value = random.random()
                cumulative_probability = 0
                # випадкове число порівнюється з кумулятивною ймовірністю кожного місця
                for p in probabilities:
                    cumulative_probability += p[1]
                    if random_value <= cumulative_probability:
                        next_location = p[0]
                        break
            # остаточний наступний пункт місцезнаходження
            ant_locations[i] = next_location
            # оновлення феромонів на шляхах після кожної ітерації
            for j in range(n):
                if j != current_location:
                    pheromone_matrix[current_location][j] *= (1 - EvaporationRate)
                    pheromone_matrix[current_location][j] += EvaporationRate * PheromoneInitial
            # для кожного шляху, який не next_location, рівень феромонів зменшується на коефіцієнт випаровування
                if j == next_location:
                    pheromone_matrix[current_location][j] += q0 * PheromoneInitial + (1 - q0) * \
                                                             pheromone_matrix[current_location][j]
        # для обраного шляху рівень феромону збільшується на q0 внесок мурахи в поле феромонів
        for i in range(n):
            for j in range(n):
                pheromone_matrix[i][j] *= (1 - EvaporationRate)
        # всі значення феромонів зменшуються на коефіцієнт випаровування
        # побудова кожного маршруту, який пройшли мурахи протягом кожної ітерації
        for start in range(n):
            visited = {start}
            distance = 0
            path = [start]

            while len(visited) < n:
                current_location = path[-1]
                # cтворення списку, що містить відстані від поточної локації до всіх невідвіданих
                distances = [(j, distance_matrix[current_location][j]) for j in allowed_locations if j not in visited]
                if not distances:
                    break
                next_location = min(distances, key=lambda x: x[1])[0]
                # вибирання наступної локаціі, як такої, що має найменшу відстань від поточної
                visited.add(next_location)
                path.append(next_location)
                distance += distance_matrix[current_location][next_location]
            # відстань від поточної локації до наступної додається до загальної
            # add the distance back to the starting point
            distance += distance_matrix[path[-1]][start]
            # перевірка, чи довжина поточного маршруту коротша за найкоротший знайдений маршрут
            if distance < shortest_distance:
                shortest_distance = distance
                shortest_path = path + [start]
        print(f"Iteration {iter + 1}: shortest path length = {shortest_distance}, shortest path = {shortest_path}")
    return shortest_path, shortest_distance

# візуалізація знайденого маршруту на графіку
def plot_map(path, distance):
    path = path[:-1]
    n = len(path)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x_coords = np.cos(theta)
    y_coords = np.sin(theta)
    plt.figure(figsize=(7, 6))
    plt.scatter(x_coords, y_coords, color='red', marker='o')
    plt.title('Cities')
    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.box(True)
    plt.plot(x_coords, y_coords, color='green', alpha=1, linewidth=0.4)
    x_endpoint = np.array([x_coords[-1], x_coords[0]])
    y_endpoint = np.array([y_coords[-1], y_coords[0]])
    plt.plot(x_endpoint, y_endpoint, color='green', alpha=1, linewidth=0.4)
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        plt.text(x, y, str(path[i]) + '(' + str(i + 1) + ')', ha='center', va='bottom', fontsize=10)
    for i in range(n):
        for j in range(i + 1, n):
            plt.plot([x_coords[i], x_coords[j]], [y_coords[i], y_coords[j]], color='gray', alpha=0.3, linewidth=0.1)
    plt.show()


generate_map()
N, Distances = read_map()
alpha = 1
# вага відстані
beta = 5
# вага концентрації феромонів
EvaporationRate = 0.5
# коефіцієнт випаровування феромону
q0 = 0.5
# внесок мурахи в поле феромонів
PheromoneInitial = 1.0
IterationNumber = 10
shortestPath, shortestDistance = ant_colony_optimization(Distances, N)
plot_map(shortestPath, N)
print(f"\nShortest path length = {shortestDistance}, shortest path = {shortestPath}")
