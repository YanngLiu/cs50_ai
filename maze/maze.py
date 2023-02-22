import argparse
import heapq

class Node():
    def __init__(self, state, parent, action):
        self.state = state
        self.parent = parent
        self.action = action

class WeightedNode(Node):
    def __init__(self, weight, state, parent, action):
        super(WeightedNode, self).__init__(state, parent, action)
        self.weight = weight

    def __lt__(self, another) :
        return self.weight < another.weight


class StackFrontier():
    def __init__(self):
        self.frontier = []

    def add(self, node):
        self.frontier.append(node)

    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[-1]
            self.frontier = self.frontier[:-1]
            return node


class QueueFrontier(StackFrontier):
    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[0]
            self.frontier = self.frontier[1:]
            return node


class GreedyBestFirstFrontier(StackFrontier):
    def __init__(self):
        super(GreedyBestFirstFrontier, self).__init__()
        self.heuristicMap = {}
    
    def heuristic(self, state, goal):
        # use manhattan distance from state to goal
        if state not in self.heuristicMap:
            res = abs(state[0] - goal[0]) + abs(state[1] - goal[1])
            self.heuristicMap[state] = res
        return self.heuristicMap[state]

    def add(self, node):
        heapq.heappush(self.frontier, node)

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = heapq.heappop(self.frontier)
            return node


class Maze():
    def __init__(self, filename, strategy):
        self.strategy = strategy
        # Read file and set height and width of maze
        with open(filename) as f:
            contents = f.read()

        # Validate start and goal
        if contents.count("A") != 1:
            raise Exception("maze must have exactly one start point")
        if contents.count("B") != 1:
            raise Exception("maze must have exactly one goal")

        # Determine height and width of maze
        contents = contents.splitlines()
        self.height = len(contents)
        self.width = max(len(line) for line in contents)

        # Keep track of walls
        self.walls = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                try:
                    if contents[i][j] == "A":
                        self.start = (i, j)
                        row.append(False)
                    elif contents[i][j] == "B":
                        self.goal = (i, j)
                        row.append(False)
                    elif contents[i][j] == " ":
                        row.append(False)
                    else:
                        row.append(True)
                except IndexError:
                    row.append(False)
            self.walls.append(row)

        self.solution = None


    def simple(self):
        return self.strategy in {'bfs', 'dfs'}


    def print(self):
        cells = self.solution[1] if self.solution is not None else None
        if self.simple():
            solution = cells
        else:
            solution = None if cells is None else {state for _, state in cells}
        print()
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):
                if col:
                    print(b'\xe2\x96\x88'.decode('utf8'), end="")
                elif (i, j) == self.start:
                    print("A", end="")
                elif (i, j) == self.goal:
                    print("B", end="")
                elif solution is not None and (i, j) in solution:
                    print("*", end="")
                else:
                    print(" ", end="")
            print()
        print()


    def neighbors(self, state):
        row, col = state
        candidates = [
            ("up", (row - 1, col)),
            ("down", (row + 1, col)),
            ("left", (row, col - 1)),
            ("right", (row, col + 1))
        ]

        result = []
        for action, (r, c) in candidates:
            if 0 <= r < self.height and 0 <= c < self.width and not self.walls[r][c]:
                result.append((action, (r, c)))
        return result


    def solve(self):
        """Finds a solution to maze, if one exists."""

        # Keep track of number of states explored
        self.num_explored = 0

        # Initialize frontier to just the starting position
        match self.strategy:
            case 'dfs':
                start = Node(state=self.start, parent=None, action=None)
                frontier = StackFrontier()
            case 'bfs':
                start = Node(state=self.start, parent=None, action=None)
                frontier = QueueFrontier()
            case 'gbfs':
                start = WeightedNode(weight=0, state=self.start, parent=None, action=None)
                frontier = GreedyBestFirstFrontier()
        frontier.add(start)

        # Initialize an empty explored set
        self.explored = {}

        # Keep looping until solution found
        while True:

            # If nothing left in frontier, then no path
            if frontier.empty():
                raise Exception("no solution")

            # Choose a node from the frontier
            node = frontier.remove()
            self.num_explored += 1

            # If node is the goal, then we have a solution
            if node.state == self.goal:
                actions = []
                cells = []
                while node.parent is not None:
                    actions.append(node.action)
                    if self.simple():
                        cells.append(node.state)
                    else:
                        cells.append((node.weight, node.state,))
                    node = node.parent
                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells, self.explored)
                return

            # Mark node as explored
            self.explored[node.state] = -1 if self.simple() else node.weight

            # Add neighbors to frontier
            for action, state in self.neighbors(node.state):
                if not frontier.contains_state(state) and state not in self.explored:
                    if self.simple():
                        child = Node(state=state, parent=node, action=action)
                    else:
                        child = WeightedNode(weight=frontier.heuristic(state, self.goal), state=state, parent=node, action=action)
                    frontier.add(child)


    def output_image(self, filename, show_solution=True, show_explored=False):
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 50
        cell_border = 2
        weight_border = 7

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.width * cell_size, self.height * cell_size),
            "black"
        )
        draw = ImageDraw.Draw(img)

        cells = self.solution[1] if self.solution is not None else None
        if self.simple():
            solution = cells
        else:
            solution = {state for _, state in cells}
            weights = self.solution[2] # {state: weight for weight, state in cells}
        # or cp /workspaces/.codespaces/shared/editors/jetbrains/JetBrainsRider-2022.3.2/jbr/lib/fonts/SourceCodePro-Regular.ttf .
        font = ImageFont.truetype(font="OpenSans-Regular.ttf", size=25)
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):
                weight = -1

                # Walls
                if col:
                    fill = (40, 40, 40)
                # Start
                elif (i, j) == self.start:
                    fill = (255, 0, 0)
                # Goal
                elif (i, j) == self.goal:
                    fill = (0, 171, 28)
                # Solution
                elif solution is not None and show_solution and (i, j) in solution:
                    fill = (220, 235, 113)
                    if not self.simple():
                        weight = weights[(i, j,)]
                # Explored
                elif solution is not None and show_explored and (i, j) in self.explored:
                    fill = (212, 97, 85)
                    if not self.simple():
                        weight = weights[(i, j,)]
                # Empty cell
                else:
                    fill = (237, 240, 252)

                # Draw cell
                draw.rectangle(
                    ([(j * cell_size + cell_border, i * cell_size + cell_border),
                      ((j + 1) * cell_size - cell_border, (i + 1) * cell_size - cell_border)]),
                    fill=fill
                )
                if not self.simple() and weight > 0:
                    draw.text((j * cell_size + cell_border + weight_border, i * cell_size + cell_border + weight_border),str(weight),(0,0,128),font=font)

        img.save(filename)



def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [MAZE_FILE] [STRATEGY]...",
        description=" Harvard CS50 introductrion to AI with python - search."
    )
    parser.add_argument("-m","--maze_file", type=str, required=True)
    parser.add_argument("-s", "--strategy", type=str, default="dfs")
    parser.add_argument("-e", "--show_explored", type=bool, default=False)
    return parser

parser = init_argparse()
args = parser.parse_args()
maze_file = args.maze_file
strategy = args.strategy
show_explored = args.show_explored

m = Maze(maze_file, strategy)
print("Maze:")
m.print()
print("Solving...")
m.solve()
print("States Explored:", m.num_explored)
print("Solution:")
m.print()
m.output_image("maze.png", show_explored=show_explored)
