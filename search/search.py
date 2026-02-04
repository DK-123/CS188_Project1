# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    
    "*** YOUR CODE HERE ***"
    tree = util.Stack()

    start = problem.getStartState()
    visited = set()

    if problem.isGoalState(start):
        return []
    
    tree.push((start, []))


    while not tree.isEmpty():

        node = tree.pop()
        state = node[0]
        actions = node[1]

        if state not in visited:
            visited.add(state)

            if problem.getSuccessors(state):
                children = problem.getSuccessors(state)

                for child in reversed(children):
                    new_state = child[0]
                    action = child[1]
                    cost = child[2]

                    if new_state not in visited:
                        new_node = (new_state, actions + [action])
                        tree.push(new_node)
                        # visited.add(child)

        if problem.isGoalState(state):
            return actions
             

    return []


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

    '''
    Notes:
    - for BFS we use a queue (FIFO - first in first out) because we are searching level by level
    - visit ALL nodes on each level before moving on to the next level
    - need to keep track of 2 things: search nodes we visited AND search nodes that are in the queue that need to be visited 
    - first we push the start (root) into the queue
    - once we pop from the queue, we need to store the visited nodes -> use a set to avoid duplicates?
    - push children nodes into the queue
    - *every time we pop and mark a node as visited, we push children nodes of that node into the queue
    -  a search node must contain not only a state but also the information necessary to reconstruct the path (plan)
       which gets to that state
    '''
    
    # first we need to initialize the start State and its actions and the queue
    queue = util.Queue()
    startState = problem.getStartState()
    actions = []
    first_search_node = (startState, actions)
    queue.push(first_search_node) 
    visited_states = set() 

    # keep going until our queue is empty
    while not queue.isEmpty():
        current_node = queue.pop()
        current_state = current_node[0]
        current_actions = current_node[1]

        if problem.isGoalState(current_state):
            return current_actions
        
        visited_states.add(current_state)
        
        #successors are basically the children nodes and they contain 3 things: successor state, action, stepCost
        successors = problem.getSuccessors(current_state) 

        for next_state, action, stepCost in successors:
            if next_state not in visited_states:
                visited_states.add(next_state)
                new_actions = current_actions + [action]
                next_node = (next_state, new_actions)
                queue.push(next_node)

    return actions

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    pq = util.PriorityQueue()
    startState = problem.getStartState()
    actions = []

    first_node = (startState, actions)
    totCost = 0
    pq.push((first_node, actions, totCost), 0) 
    visited_states = set()


    while not pq.isEmpty():
        state, actions, totCost = pq.pop()

        if state not in visited_states:
            visited_states.add(state)
        else:
            continue

        if problem.isGoalState(state):
            return actions
        
        if problem.getSuccessors(state):
            successors = problem.getSuccessors(state)
        
        for next_state, action, stepCost in successors:
            if next_state not in visited_states:
                visited_states.add(next_state)
                new_actions = actions + [action]
                pq.push((next_state, new_actions, totCost + stepCost), totCost + stepCost)
    return actions

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

