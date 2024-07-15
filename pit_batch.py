import Arena
import matplotlib.pyplot as plt
from MCTS import MCTS
from hex.HexGame import HexGame, display
from hex.HexPlayers import *
from hex.pytorch.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = HexGame(6)

# all players
rp = RandomPlayer(g)
hp = HumanHexPlayer(g)
abp1 = AlphaBetaPlayer(g, maxDepth=1)
abp2 = AlphaBetaPlayer(g, maxDepth=2)
abp3 = AlphaBetaPlayer(g, maxDepth=3)
abps = [None, abp1, abp2, abp3]

res = {'random': {}, 'abp1': {}, 'abp2': {}, 'abp3': {}}

num = 10 

full_cps = [1, 2, 3, 5, 7, 8, 9, 11, 13, 17, 21, 24, 28, 29, 30, 31, 33, 36, 38,\
			39, 40, 41, 42, 44, 48, 50 ,57, 59, 60, 61, 63, 67, 68, 69, 71, 72, 73,\
			74, 78, 79, 85, 89, 91, 95, 99]
cur_cps = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 50]

for cp in cur_cps:#修改上面cur_cps值来选择要对战的检查点
	n1 = NNet(g)
	n1.load_checkpoint('./pretrained_models/hex/','checkpoint_{}.pth.tar'.format(cp))
	args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
	mcts = MCTS(g, n1, args1)	
	azp = lambda x, player: np.argmax(mcts.getActionProb(x, player, temp=0))

	arena = Arena.Arena(azp, rp.play, g, display=display)
	print('=========== playing check point {} vs {} ==========='.format(cp, 'random'))
	az_won, rp_won, draws = arena.playGames(num, verbose=True)
	print((az_won, rp_won, draws))
	total_turn = arena.total_turn
	print('sim count MCTS all', mcts.sim_count, 'avg game', mcts.sim_count/num, 'avg turn', mcts.sim_count/total_turn)
	res['random'][cp] = (az_won, num)

	for depth in [1, 3]:	#[1,2]
		player = abps[depth]
		player.sim_count = 0
		mcts.sim_count = 0

		arena = Arena.Arena(azp, player.play, g, display=display, mcts=mcts, ab=player)
		print('=========== playing check point {} vs abp d{} ==========='.format(cp, depth))
		az_won, rp_won, draws = arena.playGames(num, verbose=True)
		print((az_won, rp_won, draws))
		total_turn = arena.total_turn
		print('sim count MCTS all', mcts.sim_count, 'avg game', mcts.sim_count/num, 'avg turn', mcts.sim_count/total_turn)
		print('sim count alpha beta', player.sim_count, 'avg game', player.sim_count/num, 'avg turn', player.sim_count/total_turn)
		res['abp{}'.format(depth)][cp] = (az_won, num)

	print('current res')
	print(res)

print('final res')
print(res)

# Function to extract win rates from results dictionary 输出学习曲线到图片
def extract_win_rates(res, players):
    win_rates = {player: [] for player in players}
    checkpoints = sorted(res['random'].keys())

    for cp in checkpoints:
        for player in players:
            if player in res:
                az_won, total_games = res[player][cp]
                win_rate = az_won / total_games
                win_rates[player].append(win_rate)
            else:
                win_rates[player].append(0)  # No data for this player at this checkpoint

    return checkpoints, win_rates

# Players to plot
players = ['random', 'abp1', 'abp2']

# Extract win rates
checkpoints, win_rates = extract_win_rates(res, players)

# Plotting the learning curves
plt.figure(figsize=(10, 6))
for player in players:
    plt.plot(checkpoints, win_rates[player], label=player)

plt.xlabel('Checkpoint')
plt.ylabel('Win Rate')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)

# Save the plot as a file
plt.savefig('./logs/learning/learning_curve.png')