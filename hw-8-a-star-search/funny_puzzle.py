import heapq

def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    distance = 0
    for i in range(1,8):
        from_index = from_state.index(i)
        to_index = to_state.index(i)
        x = from_index // 3
        y = from_index % 3
        w = to_index // 3
        z = to_index % 3
        x_dist = abs(x - w)
        y_dist = abs(y - z)
        distance += x_dist + y_dist
    return distance

def print_succ(state):
    """
    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    """
    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """

    # for each tile, need to check if adjacent tiles are 0
    # if they are, you can move them
    # however, very specific rules to check if 2 tiles are adjacent
    # get adjacent tiles

    succ_states = []

    for i, val in enumerate(state):
        #print(i)
        if(state[i] != 0):
            if(i != 2 and i != 5 and i != 8):
                if(state[i+1] == 0):
                    succ_states.append(create_state(state, i, i+1))
                    #print(new_state)
            if(i < 6):
                if(state[i+3] == 0):
                    succ_states.append(create_state(state, i, i+3))
            if(i != 0 and i != 3 and i != 6):
                if(state[i-1] == 0):
                    succ_states.append(create_state(state, i, i-1))
            if(i > 2):
                if(state[i-3] == 0):
                    succ_states.append(create_state(state, i, i-3))   
    return sorted(succ_states)
    
def create_state(state, i, new_i):
    new_state = state.copy()
    new_state[i] = state[new_i]
    new_state[new_i] = state[i]
    return new_state

def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """
    
    # g is moves
    index_counter = 0

    # open list
    pq = []
    dict = {}
    g = 0
    h = get_manhattan_distance(state, goal_state)
    f = g + h

    heapq.heappush(pq, (f, state, (g, h, index_counter, index_counter-1)))
    dict[index_counter] = ((state, h, g), index_counter-1)
    # closed list
    visited = {tuple(state)}
    max_length = 0
    
    state_info_list = []
    #state_info_list = [(state, get_manhattan_distance(state), moves)]
    
    while pq:
        p = heapq.heappop(pq)
        
        f = p[0]
        current_state = p[1]      
        g = p[2][0]
        h = p[2][1]
        current_index = p[2][2]
        parent_index = p[2][3]

        if current_state == goal_state:
            #h = 0
            break
        succ_states = get_succ(current_state)
        for succ_state in succ_states:
            if tuple(succ_state) not in visited:
                index_counter += 1
                moves = g + 1
                h = get_manhattan_distance(succ_state, goal_state)
                dict[index_counter] = ((succ_state, h, moves), current_index)
                heapq.heappush(pq, (moves + h, succ_state, (moves, h, index_counter, current_index)))
                visited.add(tuple(succ_state))
        max_length = max(max_length, len(pq))
        
    state_info_list.append((current_state, h, g))

    while(current_index != -1):
        state_info_list.append(dict[current_index][0])
        current_index = dict[current_index][1]

    # breaks to here
    for state_info in reversed(state_info_list[1:]):
        current_state = state_info[0]
        h = state_info[1]
        move = state_info[2]
        print(current_state, "h={}".format(h), "moves: {}".format(move))
    print("Max queue length: {}".format(max_length))
    
"""
if __name__ == "__main__":

    #print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))

    #print(get_manhattan_distance([2,5,1,4,3,6,7,0,0], [1, 2, 3, 4, 5, 6, 7, 0, 0]))

    #print_succ([2,5,1,4,0,6,7,0,3])
    
    #solve([1,2,3,4,5,6,0,7,0])

    solve([2,5,1,4,0,6,7,0,3])
"""
