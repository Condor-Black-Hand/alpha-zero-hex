import Arena
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

g = HexGame(11)#棋盘大小

# all players
rp = RandomPlayer(g).play
hp = HumanHexPlayer(g).play
abp = AlphaBetaPlayer(g, maxDepth=3)
abpp = abp.play     #有点问题？

# nnet players
n1 = NNet(g)
n1.load_checkpoint('./pretrained_models/hex/','checkpoint_311.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.5})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x, player: np.argmax(mcts1.getActionProb(x, player, temp=0))

n2 = NNet(g)
n2.load_checkpoint('./pretrained_models/hex/','checkpoint_396.pth.tar')
args2 = dotdict({'numMCTSSims': 50, 'cpuct':1.5})
mcts2 = MCTS(g, n2, args2)
n2p = lambda x, player: np.argmax(mcts2.getActionProb(x, player, temp=0))

arena = Arena.Arena(n1p, n2p, g, display=display, mcts=mcts1, ab=abp)       #如果是我们先手就把n1p放在前面，hp放在后面，后手就把hp放在前面，n1p放在后面
num = 2    #设置为1，换手时记得修改上面的顺序，设置为1不能避免输出错误棋步
#输入队伍名，传给playGames，再传给playGame
teamA = input("Enter name for Team A: ")
teamB = input("Enter name for Team B: ")

print(arena.playGames(num, verbose=True, teamA=teamA, teamB=teamB, mode=1))
total_turn = arena.total_turn
#print('sim count MCTS all', mcts1.sim_count, 'avg game', mcts1.sim_count/num, 'avg turn', mcts1.sim_count/total_turn)
#print('sim count alpha beta', abp.sim_count, 'avg game', abp.sim_count/num, 'avg turn', abp.sim_count/total_turn)