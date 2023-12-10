import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from collections import deque
from game import SnakeGame, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000

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


class Agent:
    
    def __init__(self):
        self.n_games = 0 # number of games played
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate (how much we care about future reward) TODO: try different discount rates
        self.memory = deque(maxlen=MAX_MEMORY)
        
        self.model = Linear_QNet(11, 256, 3) # 11 - size of inpput layer (state with 11 features), 256 - size of hidden layer, 3 - size of output layer (3 possible moves), TODO: try different hidden layer sizes
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma) # TODO: try different learning rates
    
    def get_state(self, game):
        # points around the head
        head = game.snake[0]
        point_l = Point(head.x-BLOCK_SIZE, head.y) # left of head
        point_r = Point(head.x+BLOCK_SIZE, head.y) # right of head
        point_u = Point(head.x, head.y-BLOCK_SIZE) # up of head
        point_d = Point(head.x, head.y+BLOCK_SIZE) # down of head
        
        # directions
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]
        return np.array(state, dtype=int)        
        
        
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples of SARS
        else:
            mini_sample = self.memory
            
        states, actions, rewards, next_states, dones = zip(*mini_sample) # unzips data from tuples into individual lists
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        self.epsilon = 80 - self.n_games # decrease randomness as we train TODO: Edit the epsilon value to train better/worse
        final_move = [0, 0, 0] # [straight, right, left]
        if random.randint(0, 200) < self.epsilon: # choose random action
            move = random.randint(0, 2)
            final_move[move] = 1
        else: # get action from model
            state0 = torch.tensor(state, dtype=torch.float) # convert our current state to a tensor
            prediction = self.model(state0) # get the prediction from the model (runs the forward method in model.py)
            move = torch.argmax(prediction).item() # convert the prediction to a viable move
            final_move[move] = 1 # set the move to 1 to execute it
            
        return final_move

def train():
    # stuff for plotting
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0 # best score
    
    agent = Agent()
    game = SnakeGame()
    
    while True:
        # get old state
        state_old = agent.get_state(game)
        
        # get move
        final_move = agent.get_action(state_old)
        
        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # remember
        agent.remember(state_old, final_move, reward, agent.get_state(game), done)
        
        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            # updating best score
            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record', record)
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            
            if agent.n_games == 10:
                plt.savefig('DQN_10iterations.png')
            elif agent.n_games == 100:
                plt.savefig('DQN_100iterations.png')
            elif agent.n_games == 1000:
                plt.savefig('DQN_1000iterations.png')
            elif agent.n_games == 10000:
                plt.savefig('DQN_10000iterations.png')    

    
if __name__ == '__main__':
    train()
    