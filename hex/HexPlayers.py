import numpy as np
from hex.HexGame import display

"""
the play method of Player give original board ,player and return action in canonical board
this is correct for play from mcts too
"""

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, player):
        canonicalBoard = self.game.getCanonicalForm(board, player)
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(canonicalBoard, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanHexPlayer():
    def __init__(self, game):
        self.game = game

    def getCanonicalPosition(self, pos, player):
        if player == 1:
            return pos

        n = self.game.n
        x, y = pos
        board = np.zeros(shape=(n, n))
        board[x][y] = 1
        board = self.game.getCanonicalForm(board, -1)
        loc = np.where(board == -1)
        x, y = loc[0][0], loc[1][0]
        return (x, y)

    def char_to_index(self, char):
        return ord(char.lower()) - ord('a')  # ASCII值相减

    def parse_input(self, a):
        try:
            if a[0].isdigit():  # 第一个输入如果是数字（行号），就把数字给y（行）；第二个输入是字母（列号），转换成数字给x（列）
                y = int(a[0])
                x = self.char_to_index(a[1])
            else:
                x = self.char_to_index(a[0])
                y = int(a[1])
        except ValueError:
            raise ValueError("Invalid input format. Please use letter-number format like 'a 1' or '1 a'.")
        return x, y - 1  # 做一个对应显示的列号的处理

    def play(self, board, player):
        # display(board)
        canonicalBoard = self.game.getCanonicalForm(board, player)
        valid = self.game.getValidMoves(canonicalBoard, 1)
        # for i in range(len(valid)):
        #     if not valid[i]:
        #         print(int(i/self.game.n), int(i%self.game.n))
        while True:
            a = input("Enter your move (e.g., 'a 1' or '1 a'): ")
            try:
                x, y = self.parse_input(a.split())
                x, y = self.getCanonicalPosition((x, y), player)
                a = self.game.n * x + y
                if valid[a]:
                    break
                else:
                    print('Invalid move, please try again.')
            except (IndexError, ValueError):
                print('Invalid input format. Please use letter-number format like "a 1" or "1 a".')

        return a


class AlphaBetaPlayer():
    def __init__(self, game, maxDepth):
        self.game = game
        self.maxDepth = maxDepth
        self.sim_count = 0

    def alphaBeta(self, board, depth, alpha, beta, maximizingPlayer, player):
        """ 
        board here is in canonical foarm for the root player who call play of AlphaBeta
        """

        # print('depth {} alpha {} beta {} maximizingPlayer {} player {}'.format(depth, alpha, beta, maximizingPlayer, player))
        # display(board)

        self.sim_count += 1

        if depth == 0 or self.game.getGameEnded(board, 1) != 0:
            return self.game.getScore(board), None

        valids = self.game.getValidMoves(board, player)
            
        if maximizingPlayer:
            value = (float('-inf'), None)
            for action in range(self.game.getActionSize()):
                if valids[action]==0:
                    continue
                nextBoard, _ = self.game.getNextState(board, player, action)
                childValue, childAction = self.alphaBeta(nextBoard, depth - 1, alpha, beta, False, -player)
                value = max(value, (childValue, action), key=lambda t: t[0])
                alpha = max(alpha, value[0])
                if alpha >= beta:
                    break
            return value                                           
        else:
            value = (float('inf'), None)
            for action in range(self.game.getActionSize()):
                if valids[action]==0:
                    continue
                nextBoard, _ = self.game.getNextState(board, player, action)
                childValue, childAction = self.alphaBeta(nextBoard, depth - 1, alpha, beta, True, -player)
                value = min(value, (childValue, action), key=lambda t: t[0])
                beta = min(beta, value[0])
                if alpha >= beta:
                    break
            return value

    def play(self, board, player):
        canonicalBoard = self.game.getCanonicalForm(board, player)
        score, action = self.alphaBeta(canonicalBoard, self.maxDepth, float('-inf'), float('inf'), True, 1)
        return action
        