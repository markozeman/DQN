from random import randint, uniform
import matplotlib
matplotlib.use('Qt5Agg')
import pylab as plt


def roll_dice():
    return randint(1, 6)


def random_index(l):
    # from 0 to l - 1
    return randint(0, l - 1)


def random_0_1():
    return uniform(0, 1)


def wins_to_percent(wins):
    total_games = wins['g'] + wins['y'] + wins['b'] + wins['r']
    wins['g'] = (wins['g'] / total_games) * 100
    wins['y'] = (wins['y'] / total_games) * 100
    wins['b'] = (wins['b'] / total_games) * 100
    wins['r'] = (wins['r'] / total_games) * 100
    return total_games, wins


def plot_wins(wins, alg):
    total_games, wins = wins_to_percent(wins)
    bar = plt.bar(['green', 'yellow', 'blue', 'red'], [wins['g'], wins['y'], wins['b'], wins['r']])
    bar[0].set_color('g')
    bar[1].set_color('y')
    bar[2].set_color('b')
    bar[3].set_color('r')

    plt.xlabel('player')
    plt.ylabel('% of games won')
    plt.title('Percentage of games won by each player (%d games played)' % (total_games))

    name = 'figures/%s_%dgames' % (alg, total_games)
    plt.savefig(name)

    plt.show()


def plot_training_improvement():
    # 1000  iterations, every 100 this win ratio
    data = [0.26732673267326734, 0.2885572139303483, 0.2823920265780731, 0.28428927680798005, 0.3033932135728543,
            0.3227953410981697, 0.34950071326676174, 0.36704119850187267, 0.39289678135405104, 0.4045954045954046]
    x = list(range(100, 1001, 100))

    plt.plot(x, data)
    plt.xlabel('number of games')
    plt.ylabel('% of games won')
    plt.title('Percentage of games won by AI guided player during training')

    plt.savefig('figures/training_1000games')
    plt.show()

