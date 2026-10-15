# Experiment 1: A* Search Algorithm
#
# Algorithm:
# 1. Represent the graph with edge costs and a heuristic value h(n) for every node.
# 2. Put the start node in the open list with g = 0 and f = g + h.
# 3. Remove the node with the smallest f value from the open list.
# 4. If this node is the goal, return the path and its total cost.
# 5. Otherwise mark it as visited and, for every neighbour, find
#    g = cost so far + edge cost and f = g + h, and add it to the open list.
# 6. Repeat steps 3-5 until the goal is reached or the open list is empty.

import heapq

# 1. Graph (node -> list of (neighbour, cost)) and heuristic values
graph = {
    "A": [("B", 1), ("C", 3)],
    "B": [("D", 3), ("E", 1)],
    "C": [("F", 5)],
    "D": [("G", 2)],
    "E": [("G", 5)],
    "F": [("G", 2)],
    "G": []
}

h = {"A": 5, "B": 4, "C": 4, "D": 2, "E": 3, "F": 2, "G": 0}

def a_star(start, goal):
    # 2. Open list stores (f, g, node, path)
    open_list = [(h[start], 0, start, [start])]
    visited = []

    while open_list:
        # 3. Node with the smallest f value
        f, g, node, path = heapq.heappop(open_list)
        print("Visiting", node, " g =", g, " h =", h[node], " f =", f)

        # 4. Goal check
        if node == goal:
            return path, g

        if node in visited:
            continue
        visited.append(node)

        # 5. Add neighbours to the open list
        for neighbour, cost in graph[node]:
            if neighbour not in visited:
                new_g = g + cost
                new_f = new_g + h[neighbour]
                heapq.heappush(open_list, (new_f, new_g, neighbour, path + [neighbour]))

    return None, 0

# 6. Run A* from A to G
path, cost = a_star("A", "G")
print("\nShortest Path:", " -> ".join(path))
print("Total Cost:", cost)

# ============================================================
# OUTPUT
# ============================================================
# Visiting A  g = 0  h = 5  f = 5
# Visiting B  g = 1  h = 4  f = 5
# Visiting E  g = 2  h = 3  f = 5
# Visiting D  g = 4  h = 2  f = 6
# Visiting G  g = 6  h = 0  f = 6
#
# Shortest Path: A -> B -> D -> G
# Total Cost: 6
