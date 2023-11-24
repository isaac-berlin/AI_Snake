import pygame, time, random

snake_speed = 15

X_Y = 1000, 1000 # Window size
CELL_SIZE = 50 # Size of each cell in the snake



# Pygame init stuff
pygame.init()
pygame.display.set_caption('Snake')
game_window = pygame.display.set_mode((X_Y))
fps = pygame.time.Clock()
