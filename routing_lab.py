import random
import time
import heapq

# --- Часть 1: Представление графа и генерация ---

def generate_random_graph(num_vertices, max_edges_per_vertex, weight_range=(-10, 100)):
    """
    Генерирует случайный взвешенный ориентированный граф.
    - num_vertices: количество вершин в графе.
    - max_edges_per_vertex: максимальное количество исходящих ребер из одной вершины.
    - weight_range: кортеж (min_weight, max_weight) для весов ребер.
    """
    if num_vertices <= 0:
        return {}

    graph = {i: {} for i in range(num_vertices)}
    for i in range(num_vertices):
        # Генерируем случайное количество исходящих ребер
        num_edges = random.randint(1, max_edges_per_vertex)
        for _ in range(num_edges):
            # Выбираем случайного соседа, который не является текущей вершиной
            j = random.randint(0, num_vertices - 1)
            if i == j:
                continue # Петли не создаем
            
            # Генерируем случайный вес ребра
            weight = random.randint(*weight_range)
            graph[i][j] = weight
            
    return graph

# --- Часть 2: Реализация алгоритмов маршрутизации ---

def dijkstra(graph, start_vertex):
    """
    Реализация алгоритма Дейкстры для поиска кратчайших путей.
    ВНИМАНИЕ: Некорректно работает с рёбрами отрицательного веса.
    - graph: граф в виде списка смежности (словарь словарей).
    - start_vertex: начальная вершина.
    """
    if start_vertex not in graph:
        return None, "Стартовая вершина не найдена в графе."

    # Инициализация расстояний: бесконечность для всех, кроме стартовой
    distances = {vertex: float('inf') for vertex in graph}
    distances[start_vertex] = 0
    
    # Приоритетная очередь для хранения (расстояние, вершина)
    priority_queue = [(0, start_vertex)]

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        # Если мы нашли более короткий путь ранее, пропускаем
        if current_distance > distances[current_vertex]:
            continue

        # Проходим по всем соседям текущей вершины
        for neighbor, weight in graph[current_vertex].items():
            if weight < 0:
                # Алгоритм Дейкстры не предназначен для графов с отрицательными весами
                # Это условие добавлено для демонстрации
                 print(f"Предупреждение: Обнаружен отрицательный вес ({weight}) на ребре {current_vertex}->{neighbor}. Результат Дейкстры может быть некорректным.")
            
            distance = current_distance + weight

            # Если найден более короткий путь до соседа, обновляем его
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
                
    return distances, None

def bellman_ford(graph, start_vertex):
    """
    Реализация алгоритма Беллмана-Форда.
    - graph: граф в виде списка смежности.
    - start_vertex: начальная вершина.
    """
    if start_vertex not in graph:
        return None, "Стартовая вершина не найдена в графе."
        
    num_vertices = len(graph)
    distances = {vertex: float('inf') for vertex in graph}
    distances[start_vertex] = 0
    
    # Создаем список ребер для удобства итерации
    edges = []
    for u in graph:
        for v, weight in graph[u].items():
            edges.append((u, v, weight))

    # Шаг 1: Релаксация ребер (V-1) раз
    for _ in range(num_vertices - 1):
        for u, v, weight in edges:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight

    # Шаг 2: Проверка на наличие циклов отрицательного веса
    for u, v, weight in edges:
        if distances[u] != float('inf') and distances[u] + weight < distances[v]:
            return None, "Граф содержит цикл отрицательного веса"
            
    return distances, None

# --- Часть 3: Метрика сложности МакКейба ---

def calculate_mccabe_complexity(func_code_str):
    """
    Простой анализатор для расчета цикломатической сложности.
    Считает количество точек ветвления (if, for, while, and, or) + 1.
    """
    branch_keywords = ['if ', 'for ', 'while ', ' and ', ' or ', 'elif ', 'except ']
    complexity = 1
    lines = func_code_str.split('\n')
    for line in lines:
        for keyword in branch_keywords:
            if keyword in line:
                complexity += line.count(keyword)
    return complexity


# --- Часть 4: Основной блок и демонстрация ---

if __name__ == "__main__":
    # --- Демонстрация 1: Простой граф для проверки корректности ---
    print("--- ДЕМОНСТРАЦИЯ 1: Простой тестовый граф ---")
    simple_graph_positive = {
        0: {1: 10, 2: 3},
        1: {3: 2},
        2: {1: 4, 3: 8, 4: 2},
        3: {4: 7},
        4: {}
    }
    start_node = 0
    
    print("\nГраф (без отрицательных весов):")
    for node, edges in simple_graph_positive.items():
        print(f"  {node}: {edges}")
    
    print(f"\nЗапуск алгоритма Дейкстры от вершины {start_node}:")
    dijkstra_dist, error = dijkstra(simple_graph_positive, start_node)
    if error:
        print(f"Ошибка: {error}")
    else:
        print(f"  Кратчайшие пути: {dijkstra_dist}")
        
    print(f"\nЗапуск алгоритма Беллмана-Форда от вершины {start_node}:")
    bellman_dist, error = bellman_ford(simple_graph_positive, start_node)
    if error:
        print(f"Ошибка: {error}")
    else:
        print(f"  Кратчайшие пути: {bellman_dist}")
        
    # Демонстрация с отрицательными весами
    simple_graph_negative = {
        0: {1: -1, 2: 4},
        1: {2: 3, 3: 2, 4: 2},
        2: {},
        3: {1: 1, 2: 5},
        4: {3: -3}
    }
    
    print("\n-------------------------------------------")
    print("\nГраф (с отрицательными весами):")
    for node, edges in simple_graph_negative.items():
        print(f"  {node}: {edges}")

    print(f"\nЗапуск алгоритма Дейкстры от вершины {start_node} (ожидается некорректная работа):")
    dijkstra_dist_neg, error = dijkstra(simple_graph_negative, start_node)
    if error:
        print(f"Ошибка: {error}")
    else:
        print(f"  Кратчайшие пути: {dijkstra_dist_neg}")

    print(f"\nЗапуск алгоритма Беллмана-Форда от вершины {start_node}:")
    bellman_dist_neg, error = bellman_ford(simple_graph_negative, start_node)
    if error:
        print(f"Ошибка: {error}")
    else:
        print(f"  Кратчайшие пути: {bellman_dist_neg}")
        
    # --- Демонстрация 2: Тестирование производительности на большом графе ---
    print("\n--- ДЕМОНСТРАЦИЯ 2: Тест производительности (100+ вершин) ---")
    
    NUM_VERTICES = 150
    MAX_EDGES = 10
    
    # Для Дейкстры используем только положительные веса
    large_graph_pos_weights = generate_random_graph(NUM_VERTICES, MAX_EDGES, weight_range=(1, 100))
    # Для Беллмана-Форда добавим отрицательные веса
    large_graph_neg_weights = generate_random_graph(NUM_VERTICES, MAX_EDGES, weight_range=(-10, 100))
    
    start_node_large = 0

    print(f"Сгенерирован граф на {NUM_VERTICES} вершин.")
    
    # Тест Дейкстры
    start_time = time.perf_counter()
    _, _ = dijkstra(large_graph_pos_weights, start_node_large)
    end_time = time.perf_counter()
    print(f"Время выполнения алгоритма Дейкстры: {end_time - start_time:.6f} секунд")

    # Тест Беллмана-Форда
    start_time = time.perf_counter()
    _, error = bellman_ford(large_graph_neg_weights, start_node_large)
    end_time = time.perf_counter()
    if error:
        print(f"Алгоритм Беллмана-Форда обнаружил проблему: {error}")
    else:
         print(f"Время выполнения алгоритма Беллмана-Форда: {end_time - start_time:.6f} секунд")
         
    # --- Демонстрация 3: Оценка сложности по метрике МакКейба ---
    print("\n--- ДЕМОНСТРАЦИЯ 3: Оценка структурной сложности (Метрика МакКейба) ---")
    
    # Получаем исходный код функций для анализа
    import inspect
    dijkstra_code = inspect.getsource(dijkstra)
    bellman_ford_code = inspect.getsource(bellman_ford)
    
    mccabe_dijkstra = calculate_mccabe_complexity(dijkstra_code)
    mccabe_bellman = calculate_mccabe_complexity(bellman_ford_code)
    
    print(f"Цикломатическая сложность для `dijkstra`: {mccabe_dijkstra}")
    print(f"Цикломатическая сложность для `bellman_ford`: {mccabe_bellman}")