from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

wrldH = 4
wrldW = 12
STRT = [3, 0]
END = [3, 11]

U = 0
D = 1
L = 2
R = 3
ACTIONS = [U,D,L,R]

# eps = [0.1,0.5]
epsilonS = [0.05,0.1,0.2,0.5]
ucbParams = [0.1]#,0.5,1,5,10]

alpha = 0.5

gamma = 1

class cliffWalkAgent:
    def __init__(self):
        self.sarsaQTable = np.zeros((wrldH, wrldW, 4))
        self.qlQTable = np.zeros((wrldH, wrldW, 4))
        self.t = 0
        self.actCt = np.zeros(4)

    def rst(self):
        self.actCt = np.zeros(4)
        self.t = 0
#         self.sarsaQTable = np.zeros((wrldH,wrldW,4))
#         self.qlTable = np.zeros((wrldH,wrldW,4))

    def step(self, s, a):
        self.t += 1
        self.actCt[a] += 1
        x, y = s
        if a == U:
            nS = [max(x - 1, 0), y]
        elif a == L:
            nS = [x, max(y - 1, 0)]
        elif a == R:
            nS = [x, min(y + 1, wrldW - 1)]
        elif a == D:
            nS = [min(x + 1, wrldH - 1), y]
        else:
            assert False
        r = -1
        if (a == D and x == 2 and 1 <= y <= 10) or (
            a == R and s == STRT):
            r = -100
            nS = STRT
        return nS, r

    def ucb(self, s, onPlcy, eps, ucbParam):
        if np.random.binomial(1, eps) == 1:
            return np.random.choice(ACTIONS)
        if onPlcy:
            v = self.sarsaQTable[s[0], s[1], :]
        else:
            v = self.qlQTable[s[0], s[1], :]

        ucbVs = v + ucbParam * np.sqrt(np.log(self.t + 1) / (self.actCt + 1e-5))
        return np.random.choice([p for p, q in enumerate(ucbVs) if q == np.max(ucbVs)])

    def sarsa(self, eps,ucbParam,stpSz=alpha):
        self.rst()
        s = STRT
        act = self.ucb(s,True,eps,ucbParam)
        rwrds = 0.0
        while s != END:
            nS, rwrd = self.step(s, act)
            nA = self.ucb(nS,True,eps,ucbParam)
            rwrds += rwrd
            trgt = 0.0
            qN = self.sarsaQTable[nS[0], nS[1], :]
            bstAs = np.argwhere(qN == np.max(qN))
            for a in ACTIONS:
                if a in bstAs:
                    trgt += ((1.0 - eps) / len(bstAs) + eps / len(ACTIONS)) * self.sarsaQTable[nS[0], nS[1], a]
                else:
                    trgt += eps / len(ACTIONS) * self.sarsaQTable[nS[0], nS[1], a]
            trgt *= gamma
            self.sarsaQTable[s[0], s[1], act] += stpSz * (rwrd + trgt - self.sarsaQTable[s[0], s[1], act])
            s = nS
            act = nA
        return rwrds

    def qLrng(self, eps,ucbParam,stpSz=alpha):
        self.rst()
        s = STRT
        rwrds = 0.0
        while s != END:
            a = self.ucb(s,False,eps,ucbParam)
            nS, rwrd = self.step(s, a)
            rwrds += rwrd
            self.qlQTable[s[0], s[1], a] += stpSz * (rwrd + gamma * np.max(self.qlQTable[nS[0], nS[1], :])-self.qlQTable[s[0], s[1], a])
            s = nS
        return rwrds

def prntOptPlcy(qTable):
    oPlcy = []
    for i in range(0, wrldH):
        oPlcy.append([])
        for j in range(0, wrldW):
            if [i, j] == END:
                oPlcy[-1].append('G')
                continue
            bstA = np.argmax(qTable[i, j, :])
            if bstA == U:
                oPlcy[-1].append('U')
            elif bstA == D:
                oPlcy[-1].append('D')
            elif bstA == L:
                oPlcy[-1].append('L')
            elif bstA == R:
                oPlcy[-1].append('R')
    for rw in oPlcy:
        print(rw)

def main():
    epsds = 500
    runs = 50
    for ucbParam in ucbParams:
        for eps in epsilonS:
            rwrdsSarsa = np.zeros(epsds)
            rwrdsQLrng = np.zeros(epsds)
            for r in tqdm(range(runs)):
                agnt = cliffWalkAgent()
                for i in range(0,epsds):
                    rwrdsSarsa[i] += agnt.sarsa(eps,ucbParam)
                    rwrdsQLrng[i] += agnt.qLrng(eps,ucbParam)
            rwrdsSarsa /= runs
            rwrdsQLrng /= runs
            plt.plot(rwrdsSarsa, label='Sarsa')
            plt.plot(rwrdsQLrng, label='Q Learning')
            plt.xlabel('Episodes')
            plt.ylabel('Rewards Sum during an episode')
            plt.ylim([-100, 0])
            plt.legend()
            plt.savefig('epsilonPlot_{}_ucb_{}.png'.format(eps, ucbParam))
            plt.close()
            print('Sarsa Optimal Policy:')
            prntOptPlcy(agnt.sarsaQTable)
            print('Q Learning Optimal Policy:')
            prntOptPlcy(agnt.qlQTable)

if __name__ == '__main__':
    main()
