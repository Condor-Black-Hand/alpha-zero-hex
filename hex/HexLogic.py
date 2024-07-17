'''
Author: Ali Agha
Date: Jan 19, 2018.

Modified by Su322.

Board class.
Board data:
  1=red, -1=blue, 0=empty
  first dim is column , 2nd is row:
     pieces[1][7] is the hexagon in column 2, row 7
hexagons are stored and manipulated as (x,y) tuples.
x is the column, y is the row.
'''
import collections
from heapq import *


class Board():

    # list of all 6 directions on the board, as (x,y) offsets
    __directions = [(-1,0),(0,-1),(1,-1),(1,0),(0,1),(-1,1)]

    def __init__(self, n):
        "Set up initial board configuration."

        self.n = n
        # Create the empty board array.
        self.pieces = [None]*self.n
        for i in range(self.n):
            self.pieces[i] = [0]*self.n

    # add [][] indexer syntax to the Board
    def __getitem__(self, index):
        return self.pieces[index]

    def get_legal_moves(self, color):
        """Returns all the legal moves. color is not needed. kept just for consistency
        """
        moves = list()  # stores the legal moves.

        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 0:
                    moves.append((x, y))
        return moves

    def has_legal_moves(self, color):
        return len(self.get_legal_moves(color))>0

    def execute_move(self, move, color):
        """Perform the given move on the board; 
        """

        # Add the piece to the empty square.
        x, y = move
        #print(self[x][y], color)
        self.pieces[x][y] = color


    def is_valid_pos(self, x, y):
        if x < 0 or y < 0 or x >= self.n or y >= self.n:
            return False
        return True

    def get_neighbors(self, pos, color, with_weights=False):
        x, y = pos
        neighbors = []
        for x_offset, y_offset in self.__directions:
            nx, ny = (x + x_offset, y + y_offset)
            if self.is_valid_pos(nx, ny):
                if with_weights:
                    if self.pieces[nx][ny] != -color:
                        neighbors.append(((nx, ny), int(self.pieces[nx][ny] == 0)))
                else:
                    if self.pieces[nx][ny] == color:
                        neighbors.append((nx, ny))

        return neighbors

    def is_connected(self, root, color):
        if self.pieces[root[0]][root[1]] != color:
            return False
        visited, queue = set([root]), collections.deque([root])
        while queue:
            pos = queue.popleft()
            for neighbor in self.get_neighbors(pos, color):
                if neighbor not in visited:
                    #修改过了            neighbor[0]是左边的                           neighbor[1]是上边的
                    if (color == 1 and neighbor[1] == self.n-1) or (color == -1 and neighbor[0] == self.n-1):
                        return True
                    visited.add(neighbor)
                    queue.append(neighbor)

        return False

    def count_to_connect(self):
        shortest = (float('inf'), None)  # 初始化最短路径为无穷大和空路径
        for y in range(self.n):  # 遍历顶部行的每一列
            if self.pieces[0][y] == -1:  # 如果该位置由对手占据，跳过
                continue
            cost, path = self.shortest_path((0, y))  # 计算从该位置开始的最短路径及其代价
            if cost < shortest[0]:  # 如果找到更短的路径，更新最短路径
                shortest = (cost, path)

        if shortest[0] == float('inf'):  # 如果找不到任何路径，返回一个大数和空路径
            return (self.n ** 2, shortest[1])

        return shortest  # 返回找到的最短路径及其代价

    def shortest_path(self, source):
        """ 
        dijkstra algorithm based on https://gist.github.com/kachayev/5990802
        return shortest path to other side for player 1
        """
        source_cost = int(self.pieces[source[0]][source[1]] == 0)  # 如果起点为空，代价为 1；否则为 0
        shortest = (float("inf"), None)  # 初始化最短路径为无穷大和空路径
        q, seen, mins = [(source_cost, source, [])], set(), {source: source_cost}  # 初始化优先队列、已访问节点集合和最小代价字典
        while q:
            (cost, v1, path) = heappop(q)  # 从优先队列中弹出代价最小的节点
            if v1 not in seen:
                seen.add(v1)  # 标记节点为已访问
                path = [*path, v1]  # 更新路径
                if v1[0] == self.n - 1 and cost < shortest[0]:  # 如果到达底部且代价更小，更新最短路径
                    shortest = (cost, path)

                for v2, c in self.get_neighbors(v1, 1, with_weights=True):  # 遍历邻居节点
                    if v2 in seen: continue  # 如果邻居节点已访问，跳过
                    prev = mins.get(v2, None)
                    next = cost + c  # 计算从当前节点到邻居节点的代价
                    if prev is None or next < prev:  # 如果找到更短的路径，更新最小代价字典和优先队列
                        mins[v2] = next
                        heappush(q, (next, v2, path))

        return shortest  # 返回找到的最短路径及其代价


