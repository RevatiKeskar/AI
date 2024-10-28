import numpy as np
import pygame
import random
import heapq
import time

# Initialize Pygame
pygame.init()

# Screen settings
SCREEN_WIDTH, SCREEN_HEIGHT = 400, 600
GRID_SIZE = 4
TILE_SIZE = SCREEN_WIDTH // GRID_SIZE
MARGIN = 5

# Colors
BACKGROUND_COLOR = (255, 255, 255)  # Dark grey
TILE_COLOR = (173, 216, 230)     # Bright blue
TEXT_COLOR = (50, 50, 50)     # White
BUTTON_COLOR = (0, 204, 0)        # Green
BUTTON_HOVER_COLOR = (0, 255, 0)  # Lighter green
MOVE_COUNT_COLOR = (0, 0, 0)  # Yellow
SHADOW_COLOR = (200, 200, 200) 

# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("15 Puzzle Solver")

# Font settings
font = pygame.font.Font(None, 25)
font1 = pygame.font.Font(None, 40)

# Initial puzzle configuration
initial_state = np.array([[1, 2, 3, 4],
                          [6, 0, 7, 12],
                          [5, 9, 8, 15],
                          [13, 14, 10, 11]])

class PuzzleSolver:
    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.goal_state = np.array([[1, 2, 3, 4],
                                    [5, 6, 7, 8],
                                    [9, 10, 11, 12],
                                    [13, 14, 15, 0]])
        self.moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Down, Up, Right, Left

    def solve(self, heuristic):
        open_list = []
        closed_list = set()
        g_score = {self.tuple_state(self.initial_state): 0}
        f_score = {self.tuple_state(self.initial_state): self.heuristic(self.initial_state, heuristic)}
        came_from = {}

        heapq.heappush(open_list, (f_score[self.tuple_state(self.initial_state)], self.tuple_state(self.initial_state)))

        while open_list:
            _, current_tuple = heapq.heappop(open_list)
            current = np.array(current_tuple)
            if np.array_equal(current, self.goal_state):
                return self.reconstruct_path(came_from, current)  # Return the path to solve the puzzle
            
            closed_list.add(current_tuple)
            
            for neighbor, _ in self.get_neighbors(current):
                neighbor_tuple = self.tuple_state(neighbor)
                tentative_g_score = g_score[current_tuple] + 1  # Each move costs 1
                
                if neighbor_tuple in closed_list:
                    continue
                
                if tentative_g_score < g_score.get(neighbor_tuple, float('inf')):
                    came_from[neighbor_tuple] = current_tuple
                    g_score[neighbor_tuple] = tentative_g_score
                    f_score[neighbor_tuple] = tentative_g_score + self.heuristic(neighbor, heuristic)
                    heapq.heappush(open_list, (f_score[neighbor_tuple], neighbor_tuple))
                    
        return None  # No solution found

    def heuristic(self, state, heuristic_type):
        if heuristic_type == 'manhattan':
            return self.manhattan_distance(state)
        elif heuristic_type == 'linear_conflict':
            return self.linear_conflict(state)
        elif heuristic_type == 'misplaced':
            return self.misplaced_tiles(state)
        else:
            return 0

    def manhattan_distance(self, state):
        distance = 0
        for i in range(4):
            for j in range(4):
                if state[i, j] != 0:
                    x, y = divmod(state[i, j] - 1, 4)
                    distance += abs(x - i) + abs(y - j)
        return distance

    def linear_conflict(self, state):
        distance = self.manhattan_distance(state)  # Start with the Manhattan distance
        linear_conflicts = 0

        # Check rows for linear conflicts
        for row in range(4):
            max_value = -1
            for col in range(4):
                tile_value = state[row, col]
                if tile_value != 0 and (tile_value - 1) // 4 == row:  # Tile is in its goal row
                    if tile_value > max_value:
                      max_value = tile_value
                    else:
                        linear_conflicts += 2  # Each linear conflict adds two moves

        # Check columns for linear conflicts
        for col in range(4):
            max_value = -1
            for row in range(4):
                tile_value = state[row, col]
                if tile_value != 0 and (tile_value - 1) % 4 == col:  # Tile is in its goal column
                    if tile_value > max_value:
                     max_value = tile_value
                    else:
                        linear_conflicts += 2  # Each linear conflict adds two moves

        return distance + linear_conflicts


    def misplaced_tiles(self, state):
        return sum(1 for i in range(4) for j in range(4) if state[i, j] != 0 and state[i, j] != self.goal_state[i, j])

    def get_neighbors(self, state):
        neighbors = []
        zero_position = np.argwhere(state == 0)[0]
        for move in self.moves:
            new_position = zero_position + move
            if 0 <= new_position[0] < 4 and 0 <= new_position[1] < 4:
                new_state = state.copy()
                new_state[zero_position[0], zero_position[1]], new_state[new_position[0], new_position[1]] = \
                    new_state[new_position[0], new_position[1]], new_state[zero_position[0], zero_position[1]]
                neighbors.append((new_state, 1))  # Each move has a cost of 1
        return neighbors

    def tuple_state(self, state):
        return tuple(map(tuple, state))

    def reconstruct_path(self, came_from, current):
        """Reconstruct the path from the start to the current state."""
        path = [current]
        while self.tuple_state(current) in came_from:
            current = came_from[self.tuple_state(current)]
            path.append(current)
        path.reverse()
        return path

def draw_puzzle(state):
    """Draw the puzzle on the screen with a 3D effect."""
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            tile_value = state[i, j]
            if tile_value != 0:  # Skip the empty space
                rect = pygame.Rect(j * TILE_SIZE + MARGIN, i * TILE_SIZE + MARGIN, 
                                   TILE_SIZE - 2 * MARGIN, TILE_SIZE - 2 * MARGIN)

                # Draw the shadow (3D effect)
                shadow_rect = rect.move(5, 5)
                pygame.draw.rect(screen, SHADOW_COLOR, shadow_rect)
                
                # Draw the tile
                pygame.draw.rect(screen, TILE_COLOR, rect)
                text = font1.render(str(tile_value), True, TEXT_COLOR)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)

def draw_buttons():
    """Draw Shuffle and Solve buttons."""
    shuffle_button = pygame.Rect(50, SCREEN_HEIGHT - 80, 100, 50)
    solve_button = pygame.Rect(250, SCREEN_HEIGHT - 80, 100, 50)
    
    pygame.draw.rect(screen, BUTTON_COLOR, shuffle_button)
    pygame.draw.rect(screen, BUTTON_COLOR, solve_button)
    
    shuffle_text = font.render("Shuffle", True, TEXT_COLOR)
    solve_text = font.render("Solve", True, TEXT_COLOR)
    
    screen.blit(shuffle_text, shuffle_text.get_rect(center=shuffle_button.center))
    screen.blit(solve_text, solve_text.get_rect(center=solve_button.center))
    
    return shuffle_button, solve_button

def shuffle_puzzle(state):
    """Shuffle the puzzle randomly."""
    flattened = state.flatten()
    np.random.shuffle(flattened)
    while not is_solvable(flattened) or np.array_equal(flattened.reshape((4, 4)), initial_state):
        np.random.shuffle(flattened)
    return flattened.reshape((4, 4))

def is_solvable(puzzle):
    """Check if the shuffled puzzle is solvable."""
    inversion_count = 0
    puzzle_list = puzzle[puzzle != 0]
    for i in range(len(puzzle_list)):
        for j in range(i + 1, len(puzzle_list)):
            if puzzle_list[i] > puzzle_list[j]:
                inversion_count += 1
    return inversion_count % 2 == 0

def animate_solution(path):
    """Animate the solution step by step."""
    for state in path:
        screen.fill(BACKGROUND_COLOR)
        draw_puzzle(np.array(state))
        pygame.display.flip()
        time.sleep(0.5)  # Delay to visualize each move

def display_move_counts(manhattan_moves, linear_conflict_moves, misplaced_moves, times):
    """Display the number of moves and solution times for each heuristic."""
    manhattan_text = font.render(f"Manhattan: {manhattan_moves} moves, {times['manhattan']:.2f}s", True, MOVE_COUNT_COLOR)
    linear_conflict_text = font.render(f"Linear Conflict: {linear_conflict_moves} moves, {times['linear_conflict']:.2f}s", True, MOVE_COUNT_COLOR)
    misplaced_text = font.render(f"Misplaced Tiles: {misplaced_moves} moves, {times['misplaced']:.2f}s", True, MOVE_COUNT_COLOR)
    
    screen.blit(manhattan_text, (20, SCREEN_HEIGHT - 160))
    screen.blit(linear_conflict_text, (20, SCREEN_HEIGHT - 130))
    screen.blit(misplaced_text, (20, SCREEN_HEIGHT - 100))

def main():
    """Main game loop."""
    puzzle_solver = PuzzleSolver(initial_state)
    state = initial_state.copy()
    solved = False
    move_counts = {}
    times = {}
    
    while True:
        screen.fill(BACKGROUND_COLOR)
        draw_puzzle(state)
        shuffle_button, solve_button = draw_buttons()
        
        if solved:
            display_move_counts(move_counts['manhattan'], move_counts['linear_conflict'], move_counts['misplaced'], times)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if shuffle_button.collidepoint(event.pos):
                    state = shuffle_puzzle(initial_state)
                    solved = False
                elif solve_button.collidepoint(event.pos):
                    start_time = time.time()
                    manhattan_path = puzzle_solver.solve('manhattan')
                    end_time = time.time()
                    times['manhattan'] = end_time - start_time
                    move_counts['manhattan'] = len(manhattan_path) - 1

                    start_time = time.time()
                    linear_conflict_path = puzzle_solver.solve('linear_conflict')
                    end_time = time.time()
                    times['linear_conflict'] = end_time - start_time
                    move_counts['linear_conflict'] = len(linear_conflict_path) - 1

                    start_time = time.time()
                    misplaced_path = puzzle_solver.solve('misplaced')
                    end_time = time.time()
                    times['misplaced'] = end_time - start_time
                    move_counts['misplaced'] = len(misplaced_path) - 1
                    
                    # Animate the solution for each heuristic (you can change this to show only one)
                    animate_solution(manhattan_path)
                    animate_solution(linear_conflict_path)
                    animate_solution(misplaced_path)

                    solved = True
        
        pygame.display.flip()

if __name__ == "__main__":
    main()
