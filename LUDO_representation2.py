from collections import deque
from copy import deepcopy
import numpy as np
from keras.models import load_model

from helpers import *
from DQN import *


'''
G jail -> 0
Y jail -> 1
B jail -> 2
R jail -> 3

board positions -> 4 - 55

G home -> 56
Y home-> 57
B home -> 58
R home -> 59
'''


g = {}
y = {}
b = {}
r = {}

safe_pos = [13, 26, 39, 52]

in_game = {
    'g': 5,
    'y': 18,
    'b': 31,
    'r': 44
}

final_pos = {
    'g': 55,
    'y': 16,
    'b': 29,
    'r': 42
}

homes = {
    'g': 56,
    'y': 57,
    'b': 58,
    'r': 59
}

jails = {
    'g': 0,
    'y': 1,
    'b': 2,
    'r': 3
}

turns = {
    0: 'green',
    1: 'yellow',
    2: 'blue',
    3: 'red'
}

blocks = []


def reset_game():
    global g, y, b, r
    g[0] = 0
    g[1] = 0
    g[2] = 0
    g[3] = 0

    y[0] = 1
    y[1] = 1
    y[2] = 1
    y[3] = 1

    b[0] = 2
    b[1] = 2
    b[2] = 2
    b[3] = 2

    r[0] = 3
    r[1] = 3
    r[2] = 3
    r[3] = 3


def game_ended():
    if g[0] == 56 and g[1] == 56 and g[2] == 56 and g[3] == 56:
        return 'g'
    if y[0] == 57 and y[1] == 57 and y[2] == 57 and y[3] == 57:
        return 'y'
    if b[0] == 58 and b[1] == 58 and b[2] == 58 and b[3] == 58:
        return 'b'
    if r[0] == 59 and r[1] == 59 and r[2] == 59 and r[3] == 59:
        return 'r'
    return False


def check_positions_of_others(turn):
    global g, y, b, r, safe_pos, in_game, final_pos, homes, turns

    g_pos = [position for _, position in g.items()]
    y_pos = [position for _, position in y.items()]
    b_pos = [position for _, position in b.items()]
    r_pos = [position for _, position in r.items()]

    others_positions = []
    if turns[turn] == 'green':
        others_positions.extend(y_pos)
        others_positions.extend(b_pos)
        others_positions.extend(r_pos)
    elif turns[turn] == 'yellow':
        others_positions.extend(g_pos)
        others_positions.extend(b_pos)
        others_positions.extend(r_pos)
    elif turns[turn] == 'blue':
        others_positions.extend(g_pos)
        others_positions.extend(y_pos)
        others_positions.extend(r_pos)
    elif turns[turn] == 'red':
        others_positions.extend(g_pos)
        others_positions.extend(y_pos)
        others_positions.extend(b_pos)

    return others_positions


def check_who_takes_position(new_pos):
    global g, y, b, r, safe_pos, in_game, final_pos, homes, turns

    pieces = []
    for piece, position in g.items():
        if position == new_pos:
            pieces.append(('g', piece))

    for piece, position in y.items():
        if position == new_pos:
            pieces.append(('y', piece))

    for piece, position in b.items():
        if position == new_pos:
            pieces.append(('b', piece))

    for piece, position in r.items():
        if position == new_pos:
            pieces.append(('r', piece))

    return pieces


def check_possible_moves(dice, turn):
    global g, y, b, r, safe_pos, in_game, final_pos, homes, jails, turns, blocks

    possible_moves = []

    other_positions = check_positions_of_others(turn)

    if turns[turn] == 'green':
        for piece, position in g.items():
            if position == jails['g'] and dice == 6 and in_game['g'] not in blocks:   # green jail
                move = [piece, [position, in_game['g']], ['get in']]
            elif position >= in_game['g'] and position <= final_pos['g']:
                if position + dice > final_pos['g']:    # going to green home
                    move = [piece, [position, homes['g']], ['go home']]
                else:   # normal move in the board
                    pos_between = list(range(position + 1, position + dice + 1))    # check if there is a block in between
                    for pos_b in pos_between:
                        if pos_b in blocks:
                            move = None
                            break
                        move = [piece, [position, position + dice], []]
            else:
                move = None

            if move is not None:
                new_pos = move[1][1]

                # check for block
                count = 0
                for piece_2, pos_2 in g.items():
                    if pos_2 != jails['g'] and pos_2 != homes['g'] and pos_2 == new_pos:
                        move[2].extend(['block made'])
                    if pos_2 != jails['g'] and pos_2 != homes['g'] and pos_2 == position:
                        count += 1
                if count >= 2:
                    move[2].extend(['block destroyed'])

                # check for safe positions
                if new_pos in safe_pos:
                    move[2].extend(['to safe position'])
                if position in safe_pos:
                    move[2].extend(['off safe position'])

                # check for knocking others out
                if new_pos in other_positions and new_pos not in safe_pos:
                    indices = [i for i, x in enumerate(other_positions) if x == new_pos]
                    if len(indices) == 1:
                        move[2].extend(['knock out'])

                possible_moves.append(move)

    elif turns[turn] == 'yellow':
        for piece, position in y.items():
            if position == jails['y'] and dice == 6 and in_game['y'] not in blocks:   # yellow jail
                move = [piece, [position, in_game['y']], ['get in']]
            elif position >= in_game['y'] and position <= 55 or position >= 4 and position <= final_pos['y']:
                if position <= final_pos['y'] and position + dice > final_pos['y']:    # going to yellow home
                    move = [piece, [position, homes['y']], ['go home']]
                else:   # normal move in the board
                    if position + dice > 55:
                        pos_between = list(range(position + 1, 56)) + list(range(4, position + dice - 52 + 1))  # check if there is a block in between
                    else:
                        pos_between = list(range(position + 1, position + dice + 1))    # check if there is a block in between
                    for pos_b in pos_between:
                        if pos_b in blocks:
                            move = None
                            break

                        if position + dice > 55:
                            move = [piece, [position, position + dice - 52], []]
                        else:
                            move = [piece, [position, position + dice], []]
            else:
                move = None

            if move is not None:
                new_pos = move[1][1]

                # check for block
                count = 0
                for piece_2, pos_2 in y.items():
                    if pos_2 != jails['y'] and pos_2 != homes['y'] and pos_2 == new_pos:
                        move[2].extend(['block made'])
                    if pos_2 != jails['y'] and pos_2 != homes['y'] and pos_2 == position:
                        count += 1
                if count >= 2:
                    move[2].extend(['block destroyed'])

                # check for safe positions
                if new_pos in safe_pos:
                    move[2].extend(['to safe position'])
                if position in safe_pos:
                    move[2].extend(['off safe position'])

                # check for knocking others out
                if new_pos in other_positions and new_pos not in safe_pos:
                    indices = [i for i, x in enumerate(other_positions) if x == new_pos]
                    if len(indices) == 1:
                        move[2].extend(['knock out'])

                possible_moves.append(move)

    elif turns[turn] == 'blue':
        for piece, position in b.items():
            if position == jails['b'] and dice == 6 and in_game['b'] not in blocks:   # blue jail
                move = [piece, [position, in_game['b']], ['get in']]
            elif position >= in_game['b'] and position <= 55 or position >= 4 and position <= final_pos['b']:
                if position <= final_pos['b'] and position + dice > final_pos['b']:    # going to blue home
                    move = [piece, [position, homes['b']], ['go home']]
                else:   # normal move in the board
                    if position + dice > 55:
                        pos_between = list(range(position + 1, 56)) + list(range(4, position + dice - 52 + 1))  # check if there is a block in between
                    else:
                        pos_between = list(range(position + 1, position + dice + 1))    # check if there is a block in between
                    for pos_b in pos_between:
                        if pos_b in blocks:
                            move = None
                            break

                        if position + dice > 55:
                            move = [piece, [position, position + dice - 52], []]
                        else:
                            move = [piece, [position, position + dice], []]
            else:
                move = None

            if move is not None:
                new_pos = move[1][1]

                # check for block
                count = 0
                for piece_2, pos_2 in b.items():
                    if pos_2 != jails['b'] and pos_2 != homes['b'] and pos_2 == new_pos:
                        move[2].extend(['block made'])
                    if pos_2 != jails['b'] and pos_2 != homes['b'] and pos_2 == position:
                        count += 1
                if count >= 2:
                    move[2].extend(['block destroyed'])

                # check for safe positions
                if new_pos in safe_pos:
                    move[2].extend(['to safe position'])
                if position in safe_pos:
                    move[2].extend(['off safe position'])

                # check for knocking others out
                if new_pos in other_positions and new_pos not in safe_pos:
                    indices = [i for i, x in enumerate(other_positions) if x == new_pos]
                    if len(indices) == 1:
                        move[2].extend(['knock out'])

                possible_moves.append(move)

    elif turns[turn] == 'red':
        for piece, position in r.items():
            if position == jails['r'] and dice == 6 and in_game['r'] not in blocks:   # red jail
                move = [piece, [position, in_game['r']], ['get in']]
            elif position >= in_game['r'] and position <= 55 or position >= 4 and position <= final_pos['r']:
                if position <= final_pos['r'] and position + dice > final_pos['r']:    # going to red home
                    move = [piece, [position, homes['r']], ['go home']]
                else:   # normal move in the board
                    if position + dice > 55:
                        pos_between = list(range(position + 1, 56)) + list(range(4, position + dice - 52 + 1))  # check if there is a block in between
                    else:
                        pos_between = list(range(position + 1, position + dice + 1))    # check if there is a block in between
                    for pos_b in pos_between:
                        if pos_b in blocks:
                            move = None
                            break

                        if position + dice > 55:
                            move = [piece, [position, position + dice - 52], []]
                        else:
                            move = [piece, [position, position + dice], []]
            else:
                move = None

            if move is not None:
                new_pos = move[1][1]

                # check for block
                count = 0
                for piece_2, pos_2 in r.items():
                    if pos_2 != jails['r'] and pos_2 != homes['r'] and pos_2 == new_pos:
                        move[2].extend(['block made'])
                    if pos_2 != jails['r'] and pos_2 != homes['r'] and pos_2 == position:
                        count += 1
                if count >= 2:
                    move[2].extend(['block destroyed'])

                # check for safe positions
                if new_pos in safe_pos:
                    move[2].extend(['to safe position'])
                if position in safe_pos:
                    move[2].extend(['off safe position'])

                # check for knocking others out
                if new_pos in other_positions and new_pos not in safe_pos:
                    indices = [(i, x) for i, x in enumerate(other_positions) if x == new_pos]
                    if len(indices) == 1:
                        move[2].extend(['knock out'])

                possible_moves.append(move)

    return possible_moves


def events2reward(events):
    # go home, knock out, get in, block made, to safe position, [], off safe position, block destroyed
    # 1        0.5        0.4     0.3         0.2               0   -0.2               -0.3
    if not events:  # empty list
        return 0

    mapping = {
        'go home': 1,
        'knock out': 0.5,
        'get in': 0.4,
        'block made': 0.3,
        'to safe position': 0.2,
        'off safe position': -0.2,
        'block destroyed': -0.3
        # 'get in': 0,
        # 'block made': 0,
        # 'to safe position': 0,
        # 'off safe position': 0,
        # 'block destroyed': 0
    }
    return sum([mapping[event] for event in events])


def actions2one_hot(actions):
    if not actions:     # normal move
        return np.array([0, 0, 0, 0, 0, 1, 0, 0])

    mapping = {
        'go home': 0,
        'knock out': 1,
        'get in': 2,
        'block made': 3,
        'to safe position': 4,
        '': 5,     # normal move
        'off safe position': 6,
        'block destroyed': 7,
    }
    vec = np.zeros((8,))
    for act in actions:
        vec[mapping[act]] = 1
    return vec


def state2input_neurons_without_action():
    global g, y, b, r

    g_vec = np.zeros((60,))
    y_vec = np.zeros((60,))
    b_vec = np.zeros((60,))
    r_vec = np.zeros((60,))

    for _, pos in g.items():
        g_vec[pos] += 0.25
    for _, pos in y.items():
        y_vec[pos] += 0.25
    for _, pos in b.items():
        b_vec[pos] += 0.25
    for _, pos in r.items():
        r_vec[pos] += 0.25

    return np.append(np.append(g_vec, y_vec), np.append(b_vec, r_vec))


def state_and_actions2input_neurons(actions):
    global g, y, b, r

    g_vec = np.zeros((60, ))
    y_vec = np.zeros((60, ))
    b_vec = np.zeros((60, ))
    r_vec = np.zeros((60, ))

    for _, pos in g.items():
        g_vec[pos] += 0.25
    for _, pos in y.items():
        y_vec[pos] += 0.25
    for _, pos in b.items():
        b_vec[pos] += 0.25
    for _, pos in r.items():
        r_vec[pos] += 0.25

    state_vec = np.append(np.append(g_vec, y_vec), np.append(b_vec, r_vec))
    actions_vec = actions2one_hot(actions)
    input_neurons = np.append(state_vec, actions_vec)
    return input_neurons


def predict(model, moves_events):
    global g, y, b, r

    q_values = []
    state_vec = state2input_neurons_without_action()
    for piece_num, from_to, actions in moves_events:
        old_g = deepcopy(g)
        old_y = deepcopy(y)
        old_b = deepcopy(b)
        old_r = deepcopy(r)

        new_pos = from_to[1]
        is_knock_out = 'knock out' in actions

        if is_knock_out:
            pieces = check_who_takes_position(new_pos)
            color, piece_num = pieces[0]

            if color == 'g':
                g[piece_num] = jails['g']
            elif color == 'y':
                y[piece_num] = jails['y']
            elif color == 'b':
                b[piece_num] = jails['b']
            elif color == 'r':
                r[piece_num] = jails['r']

            assert len(pieces) == 1, 'Two many pieces for knock out!'

        g[piece_num] = new_pos

        new_state_vec = state2input_neurons_without_action()

        input_neurons = np.append(state_vec, new_state_vec)
        input_neurons = np.expand_dims(input_neurons, axis=0)
        pred = model.predict(input_neurons)[0][0]
        q_values.append(pred)

        g = deepcopy(old_g)
        y = deepcopy(old_y)
        b = deepcopy(old_b)
        r = deepcopy(old_r)

    return q_values.index(max(q_values))


def play(alg, use_model=False):
    global g, y, b, r, safe_pos, in_game, final_pos, homes, jails, turns, blocks

    num_games = 10000
    # win_ratios = []
    wins = {
        'g': 0,
        'y': 0,
        'b': 0,
        'r': 0
    }

    # DQN
    if not use_model:
        model = build_model()
        memory = deque(maxlen=100000)
        epsilon, decay_value, gamma, batch_size = set_parameters(num_games)

    for num_game in range(num_games):
        reset_game()
        turn = 0  # green
        steps = 0
        blocks = []

        while True:
            dice = roll_dice()

            possible_moves = check_possible_moves(dice, turn)
            # print(turns[turn], possible_moves)

            if len(possible_moves) == 0:
                steps += 1
                turn = (turn + 1) % 4
                continue

            if alg == 'random':
                # make the random move
                action = random_index(len(possible_moves))
                move = possible_moves[action]
                # print(turns[turn], move)

            elif alg == 'heuristic':
                # from best to worse (with rewards)
                # go home, knock out, get in, block made, to safe position, [], off safe position, block destroyed
                # 1        0.5        0.4     0.3         0.2               0   -0.2               -0.3

                if turns[turn] == 'green':
                    moves_events = list(map(lambda el: el[2], possible_moves))
                    moves_rewards = [events2reward(events) for events in moves_events]
                    action = moves_rewards.index(max(moves_rewards))
                    move = possible_moves[action]
                else:
                    # make the random move
                    action = random_index(len(possible_moves))
                    move = possible_moves[action]

            elif alg == 'dqn':
                if turns[turn] == 'green':
                    if not use_model:
                        if random_0_1() < epsilon:
                            # make random move
                            action = random_index(len(possible_moves))
                            move = possible_moves[action]
                        else:   # q learning
                            # predict the move
                            # moves_events = list(map(lambda el: el[2], possible_moves))
                            max_index = predict(model, possible_moves)
                            move = possible_moves[max_index]

                            # add options to memory
                            piece_n, from_to, actions = move

                            g_prev = deepcopy(g)
                            y_prev = deepcopy(y)
                            b_prev = deepcopy(b)
                            r_prev = deepcopy(r)

                            state = state2input_neurons_without_action()
                            reward = events2reward(actions)

                            # make the move
                            new_pos = from_to[1]
                            is_knock_out = 'knock out' in actions

                            if is_knock_out:
                                pieces = check_who_takes_position(new_pos)
                                color, piece_num = pieces[0]

                                if color == 'g':
                                    g[piece_num] = jails['g']
                                elif color == 'y':
                                    y[piece_num] = jails['y']
                                elif color == 'b':
                                    b[piece_num] = jails['b']
                                elif color == 'r':
                                    r[piece_num] = jails['r']

                                assert len(pieces) == 1, 'Two many pieces for knock out!'

                            g[piece_n] = new_pos

                            next_state = state2input_neurons_without_action()

                            # check if game ended
                            done = True if game_ended() else False

                            # get next all possible actions
                            next_possible_actions = []
                            for dice_num in range(1, 7):
                                next_possible_moves = check_possible_moves(dice_num, 0)
                                for new_piece_num, new_from_to, new_actions in next_possible_moves:
                                    old_g = deepcopy(g)
                                    old_y = deepcopy(y)
                                    old_b = deepcopy(b)
                                    old_r = deepcopy(r)

                                    new_pos_2 = new_from_to[1]
                                    is_knock_out_2 = 'knock out' in new_actions

                                    if is_knock_out_2:
                                        pieces_2 = check_who_takes_position(new_pos_2)
                                        color_2, piece_num_2 = pieces_2[0]

                                        if color_2 == 'g':
                                            g[new_piece_num] = jails['g']
                                        elif color_2 == 'y':
                                            y[new_piece_num] = jails['y']
                                        elif color_2 == 'b':
                                            b[new_piece_num] = jails['b']
                                        elif color_2 == 'r':
                                            r[new_piece_num] = jails['r']

                                        assert len(pieces_2) == 1, 'Two many pieces for knock out!'

                                    g[new_piece_num] = new_pos_2

                                    next_possible_actions.append(state2input_neurons_without_action())

                                    g = deepcopy(old_g)
                                    y = deepcopy(old_y)
                                    b = deepcopy(old_b)
                                    r = deepcopy(old_r)

                            # remember experience
                            memory = remember_experience(memory, np.append(state, next_state), reward, next_state, next_possible_actions, done)

                            # unmake the move
                            g = deepcopy(g_prev)
                            y = deepcopy(y_prev)
                            b = deepcopy(b_prev)
                            r = deepcopy(r_prev)

                    else:   # use trained model
                        # moves_events = list(map(lambda el: el[2], possible_moves))
                        max_index = predict(use_model, possible_moves)
                        move = possible_moves[max_index]

                # elif turns[turn] == 'yellow':   # heuristic
                #     moves_events = list(map(lambda el: el[2], possible_moves))
                #     moves_rewards = [events2reward(events) for events in moves_events]
                #     action = moves_rewards.index(max(moves_rewards))
                #     move = possible_moves[action]

                else:   # not a green player
                    # make the random move
                    action = random_index(len(possible_moves))
                    move = possible_moves[action]

            old_pos = move[1][0]
            new_pos = move[1][1]
            is_knock_out = 'knock out' in move[2]
            is_block_made = 'block made' in move[2]
            is_block_destroyed = 'block destroyed' in move[2]

            if is_block_made:
                blocks.append(new_pos)
            if is_block_destroyed:
                blocks.remove(old_pos)

            if is_knock_out:
                pieces = check_who_takes_position(new_pos)
                color, piece_num = pieces[0]

                if color == 'g':
                    g[piece_num] = jails['g']
                elif color == 'y':
                    y[piece_num] = jails['y']
                elif color == 'b':
                    b[piece_num] = jails['b']
                elif color == 'r':
                    r[piece_num] = jails['r']

                assert len(pieces) == 1, 'Two many pieces for knock out!'

            if turns[turn] == 'green':
                g[move[0]] = new_pos
            elif turns[turn] == 'yellow':
                y[move[0]] = new_pos
            elif turns[turn] == 'blue':
                b[move[0]] = new_pos
            elif turns[turn] == 'red':
                r[move[0]] = new_pos

            # increment turn and steps
            turn = (turn + 1) % 4
            steps += 1

            # print(steps)
            # print('g', g)
            # print('y', y)
            # print('b', b)
            # print('r', r)
            # print()

            if game_ended():
                wins[game_ended()] += 1
                break

        ### after game
        if not use_model:
            # decrease epsilon
            epsilon -= decay_value

            # train network
            model = replay_experience(model, memory, gamma, batch_size)

            # if num_game % 100 == 0:
            #     win_ratio = wins['g'] / (num_game + 1)
            #     print('ratio', win_ratio)
            #     win_ratios.append(win_ratio)

        print(num_game, steps)

    if not use_model:
        model.save('model_' + str(num_games) + 'games.h5')

    # print(win_ratios)

    return wins


if __name__ == '__main__':
    alg = 'dqn'
    use_model = load_model('model_10000games_rep2.h5')

    wins = play(alg, use_model=use_model)

    print('WINS: ', wins)

    plot_wins(wins, alg)


    # todo
    # other state representation


