import pygame
import random
import os
from enum import Enum
from collections import namedtuple
import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(plot_scores, plot_mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(plot_scores)
    plt.plot(plot_mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(plot_scores)-1, plot_scores[-1], str(plot_scores[-1]))
    plt.text(len(plot_mean_scores)-1, plot_mean_scores[-1], str(plot_mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

# Get the directory of the current Python script
current_directory = os.path.dirname(__file__)

# Path to the 'arial.ttf' file in the same directory as the script
font_path = os.path.join(current_directory, 'arial.ttf')

pygame.init()
font = pygame.font.Font(font_path, 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 400

class SnakeGame:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        
    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
    # places a food block at a random position on the screen
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
    
    # returns True if game over (collision with snake or wall), else False
    def is_collision(self):
        # hits boundary
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            return True
        # hits itself
        if self.head in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
    
    # heuristic function        
    def _manhattan_distance(self, start, goal): 
        return abs(start.x - goal.x) + abs(start.y - goal.y)
    
    # A* algorithm (modified to include obstacles) (ChatGPT implementation)
    def _a_star(self):
        open_list = [self.head]
        came_from = {}
        g_score = {point: float('inf') for point in self.snake}
        g_score[self.head] = 0
        f_score = {point: float('inf') for point in self.snake}
        f_score[self.head] = self._manhattan_distance(self.head, self.food)

        while open_list:
            current = min(open_list, key=lambda point: f_score[point])

            if current == self.food:
                path = []
                while current in came_from:
                    path.insert(0, current)
                    current = came_from[current]
                return path

            open_list.remove(current)

            for neighbor in [
                Point(current.x + BLOCK_SIZE, current.y),
                Point(current.x - BLOCK_SIZE, current.y),
                Point(current.x, current.y + BLOCK_SIZE),
                Point(current.x, current.y - BLOCK_SIZE),
            ]:
                if (
                    0 <= neighbor.x < self.w
                    and 0 <= neighbor.y < self.h
                    and neighbor not in self.snake[1:]  # Exclude snake body from obstacles
                ):
                    if neighbor not in g_score:
                        g_score[neighbor] = float('inf')

                    tentative_g_score = g_score[current] + 1

                    if tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = (
                            g_score[neighbor] + self._manhattan_distance(neighbor, self.food)
                        )
                        if neighbor not in open_list:
                            open_list.append(neighbor)

        return []
    
    def play_step(self):
        action = Direction.RIGHT  # Default action if no path found

        path = self._a_star() # Use the a* algorithm to find the path to the food
        if path:
            next_point = path[0]
            if next_point.x < self.head.x:
                action = Direction.LEFT
            elif next_point.x > self.head.x:
                action = Direction.RIGHT
            elif next_point.y < self.head.y:
                action = Direction.UP
            elif next_point.y > self.head.y:
                action = Direction.DOWN

        self._move(action)  # Move the snake based on the determined action (from a*)
        self.snake.insert(0, self.head)  # Update the snake's position

        game_over = self.is_collision()  # Check if the snake has collided with itself or the wall
        if game_over:
            return True, self.score

        if self.head == self.food: # Check if the snake has eaten the food
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()  # Remove the tail segment if no food is eaten

        self._update_ui()  # Update the game display
        self.clock.tick(SPEED)  # Control game speed
        return False, self.score  # Return game status and score

def play_game():
    # stuff for plotting
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0 # best score
    num_games_played = 0
    
    game = SnakeGame() # initialize the game
    
    while True:
        game_over, score = game.play_step() # play_step plays the game... returns game_over and score
        if game_over:
            # train long memory, plot result
            game.reset()
            num_games_played += 1
            
            # updating best score
            if score > record:
                record = score

            print('Game', num_games_played, 'Score', score, 'Record', record)
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / num_games_played
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            
            if num_games_played == 10:
                plt.savefig('A-Star_10iterations.png')
            elif num_games_played == 100:
                plt.savefig('A-Star_100iterations.png')
            elif num_games_played == 1000:
                plt.savefig('A-Star_1000iterations.png')
            elif num_games_played == 10000:
                plt.savefig('A-Star_10000iterations.png')
            
            if num_games_played >= 10000: # stop after 10000 games
                break

if __name__ == '__main__':
    play_game()
    pygame.quit()