# union set
def makeset(x):
    x.parent=x

def find(x):
    if x.parent ==x:
        return x
    else:
        return find(x)

def union(x,y):
    xroot=find(x)
    yroot=find(y)
    xroot.parent=yroot

# 并查集（684）：https://leetcode.com/problems/redundant-connection/description/
# Input: [[1,2], [2,3], [3,4], [1,4], [1,5]]
# Output: [1,4]
# Explanation: The given undirected graph will be like this:
# 5 - 1 - 2
#     |   |
#     4 - 3
class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        self.p = {x:x for x in range(1,len(edges)+1)}
        for x, y in edges:
            if self.union(x,y): 
                return [x, y]
            
    def find(self,x):
        if self.p[x] != x: 
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, x, y):
        if self.find(x) == self.find(y): 
            return True
        self.p[self.find(x)] = self.find(y)
        
# 200. Number of Islands https://leetcode.com/problems/number-of-islands/
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid or not grid[0]:
            return 0
        uf = UnionFind(grid)
        directions = [(0,1),(0,-1),(-1,0),(1,0)]
        m,n=len(grid),len(grid[0])
        for i in range(m):
            for j in range(n):
                if grid[i][j]=="0":
                    continue
                for d in directions:
                    nr,nc=i+d[0],j+d[1]
                    if nr>=0 and nc>=0 and nr<m and nc<n and grid[nr][nc]=="1":
                        uf.union(i*n+j,nr*n+nc)
        return uf.count
        
class UnionFind(object):
    def __init__(self,grid):
        m,n =len(grid),len(grid[0])
        self.count=0
        self.parent=[-1]*(m*n)
        self.rank=[0]*(m*n)
        for i in range(m):
            for j in range(n):
                if grid[i][j]=="1":
                    self.parent[i*n+j]=i*n+j
                    self.count+=1
    def find(self,i):
        if self.parent[i]!=i:
            self.parent[i]=self.find(self.parent[i])
        return self.parent[i]
    
    def union(self,x,y):
        rootx=self.find(x)
        rooty=self.find(y)
        if rootx!=rooty:
            if self.rank[rootx]>self.rank[rooty]:
                self.parent[rooty]=rootx
            elif self.rank[rootx]<self.rank[rooty]:
                self.parent[rootx]=rooty
            else:
                self.parent[rooty]=rootx
                self.rank[rootx]+=1
            self.count-=1    

# 547. Friend Circles https://leetcode.com/problems/friend-circles/submissions/
class Solution:
    # union find
    def findCircleNum(self, M: List[List[int]]) -> int:
        n = len(M)
        union = [i for i in range(n)]

        def find(x):
            if union[x] != x:
                union[x] = find(union[x])
            return union[x]

        def unite(x, y):
            union[find(y)] = find(x)
            
        for i in range(n):
            for j in range(i+1, n):
                if M[i][j]:
                    unite(i,j)
                    
        for i in range(n):
            union[i] = find(i)
            
        return len(set(union))   
