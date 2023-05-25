This code uses a Grid class and an Agent class
The Grid class is used by the Agent to simulate the states, actions, and return rewards to the Agent for training. 
The Grid class also determines the effect of the wind on the Agent independently.
The Agent class is responsible for training qlearning algorithm or sarsa algorithm

GLOBAL VARIABLES:
height, height of board int
width, width of board int
wind_values, 2 dimensional list containing default wind values at each state, accessed as wind_values[x coordinate][y coordinate]
goal, tuple of goal state (x coordinate, y coordinate) as (7, 3)
start, tuple of start state (x coordinate, y coordinate) as (0, 3)

GRID:
Instantiated with parameters: 
	start, tuple defaulted to global start
	stochasitc, boolean representing if the wind is stochastic
Attributes:
	state, representing current state as tuple of (x coordinate, y coordinate)
	in_play, representing whether the game is still in play (still moves to play and no winner) as boolean
	stochastic, representing whether the wind values are determined by their grid values as boolean
	reward, representing the current reward for the state as integer
Methods:
	wind, takes a state 'state'
		if attribute stochastic is False returns the wind value from global variable wind_values at current state 
		if attribute stochastic is True returns:
			the wind value from global variables wind_values at current state with probability 1/3 
			the wind value from global variables wind_values at current state + 1 with probability 1/3
			the wind value from global variables wind_values at current state - 1 with probability 1/3, 
			if the returned value will never be negative (wind will always push upwards or none)
	bound
		checks if current state is out of the grid bounds
		if state is out of bounds, push state back in bounds and return True
		if state is not out of bounds, return False
	take_action, takes tuple 'action' of (x change, y change)
		gets wind value for current state
		adds value of action to current state (state x + action x, state y + action y)
		subtract value of wind from y coordinate (subtraction will push up in the grid)
		bound the state
		if the state was out of bounds or the action resulted in the state not changing set reward to -2
		else if the state is now the goal set reward to 100 and in_play to False
		else if the state action results in a default move (not out of bounds or reaching goal) set reward to -1
	show_grid
		display grid used for testing, not used in final code
		

Agent:
Instantiated with parameters: 
	floats: discount, alpha, epsilon, 
	string representing the algorithm "qlearning" or "sarsa" defaults to "qlearning"
	boolean representing if the Grid wind is stochastic to be passed into attribute representing Grid.
Attributes:
	instantiated parameters with same names
	actions, representing the possible actions as a list of tuples where each tuple is an action represented by the change as (x change, y change) (eg. (-1, 0) is left)
	Grid, representing an Object Grid
	states, representing all possible states as list of tuples of (x coordinate, y coordinate)
	q, representing the state action table as a dictionary of states with values as dictionaries of all actions with values of floats representing the current state action value (instantiated as 0.0)
Methods:
	best_action
		returns a random choice of actions at current Grid.state with the maximum value
	choose_action
		returns random action epsilon % of the time
		returns best action from method best_action 1 - epsilon % of the time
	qlearning, takes number of episodes for training num_episodes
		for each episode determine starting state as random state that is not the goal state
		instantiate Grid with starting state
		while the Grid is still in play
		choose an action using choose_action method and take action on Grid
		determine the best action using best_action method for new state
		update q value for state action pair using bellman equation with value of Q(current state (after action),best action determined in previous step) as Q(S', a)
	sarsa, takes number of episodes for training num_episodes
		for each episode determine starting state as random state that is not the goal state
		instantiate Grid with starting state
		choose first action using choose_action method
		while the Grid is still in play
		take action on Grid
		choose next action using choose_action method
		update q value for state action pair using bellman equation with value of Q(current state (after action),next chosen action from previous step) as Q(S', A')
		set action to next action
	train
		runs respective algorithm, qlearning if parameter algorithm is "qlearning" and sarsa if parameter algorithm is "sarsa"
	get_move, takes action
		returns letter representing action parameter (eg. "L" if action is (-1, 0))
	show_policy
		prints a grid where each location is a list of optimal actions at that location (state) and the goal state is "G"
	evaluate, takes parameter representing the number of evaluation episodes num_episodes
		for each episode instantiate Grid with start state
		take actions retrieved from best_action for current q table
		if the number of actions exceeds 5000 assume looped actions and break printing episode number and failure message
		if the goal is reached print episode number and success message with number of steps required to reach goal