import random
import time
import numpy as np
import matplotlib.pyplot as plt

height = 7
width = 10
wind_values = []
for i in range(width):
    wind_values.append([])
    for j in range(height):
        if (i > 2 and i < 6) or i == 8:
            wind_values[i].append(1)
        elif i == 6 or i == 7:
            wind_values[i].append(2)
        else:
            wind_values[i].append(0)
goal = (7,3)
start = (0,3)

class Grid:
    def __init__(self, state=start, stochastic=False):
        self.state      = state
        self.in_play    = True
        self.stochastic = stochastic
        self.reward     = 0
    
    def wind(self, state):
        if not self.stochastic:
            return wind_values[state[0]][state[1]]
        while True:
            wind = wind_values[state[0]][state[1]] + random.randint(-1,1)
            if wind > -1:
                return wind
    
    def bound(self):
        if self.state[0] < 0:
            self.state  = (0, self.state[1])
            return True
        elif self.state[0] > width-1:
            self.state  = (width-1, self.state[1])
            return True
        elif self.state[1] < 0:
            self.state  = (self.state[0], 0)
            return True
        elif self.state[1] > height-1:
            self.state  = (self.state[0], height-1)
            return True
        return False
    
    def take_action(self, action):
        wind = self.wind(self.state)
        previous_state = self.state
        self.state = (self.state[0] + action[0], self.state[1] + action[1])
        bounded = self.bound()
        self.state = (self.state[0], self.state[1] - wind)
        self.bound()
        if bounded or previous_state == self.state:
            self.reward = -2
        elif self.state == goal:
            self.reward = 100
            self.in_play = False
        else:
            self.reward = -1

    def show_grid(self):
        grid = np.zeros((height,width))
        grid[self.state[0]][self.state[1]] = 1
        for i in range(len(grid)):
            print('|',end='')
            for j in range(len(grid[0])):
                if grid[i][j] == 0:
                    print(' - ',end='|')
                elif grid[i][j] == 1:
                    print(' X ',end='|')
            print('\n')

class Agent:
    def __init__(self, discount, alpha, epsilon, algorithm="qlearning", stochastic=False):
        self.discount   = discount
        self.alpha      = alpha
        self.epsilon    = epsilon
        self.actions    = [(-1,0),(0,-1),(1,0),(0,1)]
        self.algorithm  = algorithm
        self.stochastic = stochastic
        self.Grid       = Grid()
        self.states     = []
        for x in range(width):
            for y in range(height):
                self.states.append((x,y))
        self.q          = {}
        for state in self.states:
            self.q[state]   = {(-1,0): 0.0, (0,-1): 0.0, (1,0): 0.0, (0,1): 0.0}

    def best_action(self):
        best    = max(self.q[self.Grid.state].values())
        action  = [a for a, v in self.q[self.Grid.state].items() if v == best]
        return random.choice(action)

    def choose_action(self):
        if random.random() > self.epsilon:
            return self.best_action()
        else:
            return random.choice(self.actions)

    def qlearning(self, num_episodes):
        episode = 0
        while episode < num_episodes:
            state = random.choice(self.states)
            while state == goal:
                state = random.choice(self.states)
            self.Grid = Grid(state, self.stochastic)
            while self.Grid.in_play:
                action = self.choose_action()
                self.Grid.take_action(action)
                next_action = self.best_action()
                self.q[state][action] = self.q[state][action] + self.alpha * (self.Grid.reward + (self.discount * self.q[self.Grid.state][next_action]) - self.q[state][action])
                state = self.Grid.state
            episode += 1

    def sarsa(self, num_episodes):
        episode = 0
        while episode < num_episodes:
            state = random.choice(self.states)
            while state == goal:
                state = random.choice(self.states)
            self.Grid = Grid(state, self.stochastic)
            action = self.choose_action()
            self.epsilon = 1/(episode+1) # Not used for stochastic grid
            while self.Grid.in_play:
                self.Grid.take_action(action)
                next_action = self.choose_action()
                self.q[state][action] = self.q[state][action] + self.alpha * (self.Grid.reward + (self.discount * self.q[self.Grid.state][next_action]) - self.q[state][action])
                action = next_action
                state = self.Grid.state
            episode += 1

    def train(self, num_episodes):
        if self.algorithm == "qlearning":
            print('training qlearning')
            return self.qlearning(num_episodes)
        elif self.algorithm == "sarsa":
            print('training sarsa')
            return self.sarsa(num_episodes)

    def get_move(self, action):
        if action == (1, 0):
            letter = 'R'
        elif action == (-1, 0):
            letter = 'L'
        elif action == (0, 1):
            letter = 'D'
        elif action == (0, -1):
            letter = 'U'
        return letter

    def show_policy(self):
        grid = []
        for i in range(7):
            grid.append([])
            for _ in range(10):
                grid[i].append([])
        for state in self.states:
            best    = max(self.q[state].values())
            act     = [self.get_move(a) for a, v in self.q[state].items() if v == best]
            grid[state[1]][state[0]] = act
        grid[3][7] = ['G']
        sb = ""
        for row in grid:
            sb += str(row) + "\n"
        print(sb)

    def evaluate(self, num_episodes):
        episode = 0
        steps = []
        while episode < num_episodes:
            self.Grid = Grid(start, self.stochastic)
            t = 0
            while self.Grid.in_play:
                t += 1
                self.Grid.take_action(self.best_action())
                if self.Grid.reward == 100:
                    steps.append(t)
                if t > 499:
                    steps.append(t)
                    break
            episode += 1
        return sum(steps)/len(steps)

def testing():
    random.seed(10)
    # example of function call
    discount = 0.9
    epsilon = 0.3
    alpha = 0.5
    algorithm = "qlearning" # "sarsa"
    stochastic = False # True
    agent = Agent(discount, alpha, epsilon, algorithm, stochastic)

    total_episodes = 9000
    agent.train(total_episodes)
    agent.show_policy()

    total_trials = 10
    agent.evaluate(total_trials)

    # goals = []
    # times = []
    # print('qlearning deterministic:')
    # for i in range(10):
    #     print('trial',i+1)
    #     qlearning_deterministic = Agent(discount, alpha, epsilon, "qlearning")
    #     start = time.time()
    #     qlearning_deterministic.train(9000)
    #     end = time.time()
    #     print('training complete, final policy:')
    #     qlearning_deterministic.show_policy()
    #     print('time required:', round(end-start,1), 'seconds')
    #     times.append(round(end-start,1))
    #     print('beginning evaluation')
    #     goals.append(qlearning_deterministic.evaluate(1))
    #     print()
    # print('qlearning deterministic complete\nresults:')
    # print('avg steps to goal:',sum(goals)/10)
    # print('avg time for training:',sum(times)/10)
    # print()

    # goals = []
    # times = []
    # print('sarsa deterministic:')
    # for i in range(10):
    #     print('trial',i+1)
    #     sarsa_deterministic = Agent(discount, alpha, epsilon, "sarsa")
    #     start = time.time()
    #     sarsa_deterministic.train(9000)
    #     end = time.time()
    #     print('training complete, final policy:')
    #     sarsa_deterministic.show_policy()
    #     print('time required:', round(end-start,1), 'seconds')
    #     times.append(round(end-start,1))
    #     print('beginning evaluation')
    #     goals.append(sarsa_deterministic.evaluate(1))
    #     print()
    # print('sarsa deterministic complete\nresults:')
    # print('avg steps to goal:',sum(goals)/10)
    # print('avg time for training:',sum(times)/10)
    # print()

    # episode_count = []
    # print('qlearning stochastic:')
    # for i in range(10):
    #     print('trial', i+1)
    #     qlearning_stochastic = Agent(discount, alpha, epsilon, "qlearning", True)
    #     start = time.time()
    #     episode_count.append(qlearning_stochastic.train(300))
    #     end = time.time()
    #     print('training complete, final policy:')
    #     qlearning_stochastic.show_policy()
    #     print('time required:', round(end-start,1), 'seconds')
    #     print('beginning evaluation')
    #     # goals.append(qlearning_stochastic.evaluate(10))
    #     print()
    # print('qlearning stochastic complete\nresults:')
    # print('avg number of episodes required:',sum(episode_count)/10)
    # print()
    
    # epsilon = 0.1

    # episode_count = []
    # print('sarsa stochastic:')
    # for i in range(10):
    #     print('trial', i+1)
    #     sarsa_deterministic = Agent(discount, alpha, epsilon, "sarsa", True)
    #     start = time.time()
    #     episode_count.append(sarsa_deterministic.train(300))
    #     print('avg number of episodes required:',sum(episode_count)/len(episode_count))
    #     end = time.time()
    #     print('training complete, final policy:')
    #     sarsa_deterministic.show_policy()
    #     print('time required:', round(end-start,1), 'seconds')
    #     print('beginning evaluation')
    #     # goals.append(qlearning_stochastic.evaluate(10))
    #     print()
    # print('sarsa deterministic complete\nresults:')
    # print('avg number of episodes required:',sum(episode_count)/10)
    # print()