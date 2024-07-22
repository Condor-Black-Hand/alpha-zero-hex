import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time
import datetime


def index_to_letter(index):
    return chr(ord('a') + index)


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None, mcts=None, ab=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

        self.total_turn = 0
        self.total_match = 0    #比赛局数
        self.mcts = mcts
        self.ab = ab

    def playGame(self, verbose=False, teamA=None, teamB=None, mode=0):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        start_time = datetime.datetime.now()  #游戏开始时间
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        moves = []

        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1

            if verbose:
                assert (self.display)

                self.display(board)
                print("Turn ", str(it), "Player ", str(curPlayer))
                # print('connn', curPlayer)
                # self.display(self.game.getCanonicalForm(board, curPlayer))
                # print(board)

                action = players[curPlayer + 1](board, curPlayer)       #和下面的board相比是没有转换过的?

                col_index = int(action / self.game.n)       #action是错的
                row_index = action % self.game.n + 1        #action是错的
                col_letter = index_to_letter(col_index)
                col_letter_record = index_to_letter(col_index - 32)
                print('===============  Action:', row_index, col_letter, ' ===============')  #改了下打印方式
            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                print('invalid action in arena', action)
                assert valids[action] > 0
            board, _ = self.game.getNextState(self.game.getCanonicalForm(board, curPlayer), 1, action)
            board = self.game.getOriginalForm(board, curPlayer)     #棋盘是转换过了的

            moves.append(f"{'R' if curPlayer == 1 else 'B'}({col_letter_record},{row_index})")      #记录每一步棋，你也是错的
            curPlayer = -curPlayer

            #if self.mcts is not None and self.ab is not None:
            #    print('player {} mcts {} {} ab {} {}'.format(-curPlayer, self.mcts.sim_count, self.mcts.sim_count/it, self.ab.sim_count, self.ab.sim_count/it))
            #print("")    #隔一行更好看



        if verbose:
            assert (self.display)
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)

        self.total_turn += it
        self.total_match += 1
        # print('player {} mcts {} {} ab {} {}'.format(-curPlayer, self.mcts.sim_count, self.mcts.sim_count/it, self.ab.sim_count, self.ab.sim_count/it))

        # Determine the winner
        winner = self.game.getGameEnded(board, 1)
        result = '先手胜' if winner == 1 else '后手胜' if winner == -1 else '平局'

        # Create the game record string 棋谱格式
        formatted_start_time = start_time.strftime('%Y-%m-%d %H:%M:%S')
        game_record = '{[HEX]'
        game_record += f"[{teamA}][{teamB}]"
        game_record += f"[{result}]"
        game_record += f"[{formatted_start_time} 宜宾][2024 CCGC]"
        game_record += f"{''.join([';' + move for move in moves])}"
        game_record += '}'

        # Generate a unique filename
        if self.total_match==1:     #都是在前面那个先手，设定的是交换先后手对战两局的棋谱
            filename = f"HEX-第{self.total_match}局-{teamA} vs {teamB}-{result}.txt"
        elif self.total_match==2:
            filename = f"HEX-第{self.total_match}局-{teamB} vs {teamA}-{result}.txt"

        filepath = './logs/match_record' + '/' + filename

        # Save the game record to a file  如果是训练和pit_batch可以不保存，也就是默认为0
        if mode == 1:
            with open(filepath, 'w') as f:
                f.write(game_record)

        return self.game.getGameEnded(board, 1)

    def playGames(self, num, verbose=False, teamA=None, teamB=None, mode=0):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        oneWon = 0
        twoWon = 0
        draws = 0

        if num>1:       #2局以上
            num_t = int(num / 2)
        elif num==1:    #1局
            num_t=num

        for _ in range(num_t):
            gameResult = self.playGame(verbose=verbose, teamA=teamA, teamB=teamB, mode=mode)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps + 1,
                                                                                                       maxeps=maxeps,
                                                                                                       et=eps_time.avg,
                                                                                                       total=bar.elapsed_td,
                                                                                                       eta=bar.eta_td)
            bar.next()

        if num>1:
            self.player1, self.player2 = self.player2, self.player1

            for _ in range(num_t):
                gameResult = self.playGame(verbose=verbose, teamA=teamA, teamB=teamB, mode=mode)
                if gameResult == -1:
                    oneWon += 1
                elif gameResult == 1:
                    twoWon += 1
                else:
                    draws += 1
                # bookkeeping + plot progress
                eps += 1
                eps_time.update(time.time() - end)
                end = time.time()
                bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps + 1,
                                                                                                       maxeps=num,
                                                                                                       et=eps_time.avg,
                                                                                                       total=bar.elapsed_td,
                                                                                                       eta=bar.eta_td)
                bar.next()

            bar.finish()

        return oneWon, twoWon, draws
