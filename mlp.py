#Fabricio Eduardo Omoto RA: 536520
#Joao Pedro Calandrin RA: 541605

import numpy as np

class Mlp():
    e = 2.7183
    x = []
    mse = None
    padrao = []
    desejado = []
    o1 = []
    y = [0,0,0,0]
    yd_y = [0,0,0,0]
    epoca = 0
    wx0h1 = None
    wx0h2 = None
    wx0o1 = None
    wx1h1 = None
    wx1h2 = None
    wx2h1 = None
    wx2h2 = None
    wh1o1 = None
    wh2o1 = None
    dwx0h1 = 0
    dwx0h2 = 0
    dwx0o1 = 0
    dwx1h1 = 0
    dwx1h2 = 0
    dwx2h1 = 0
    dwx2h2 = 0
    dwh1o1 = 0
    dwh2o1 = 0
    yh1 = None
    yh2 = None
    yo1 = None
    deltao1 = None
    deltah1 = None
    deltah2 = None
    n = 0.2
    l = 0.3

    def __init__(self, padrao, desejado):
        self.gerar_pesos()
        self.padrao = padrao
        self.desejado = desejado

    def gerar_pesos(self):
        self.wx0h1 = np.random.uniform(low=-1, high=1)
        self.wx0h2 = np.random.uniform(low=-1, high=1)
        self.wx0o1 = np.random.uniform(low=-1, high=1)
        self.wx1h1 = np.random.uniform(low=-1, high=1)
        self.wx1h2 = np.random.uniform(low=-1, high=1)
        self.wx2h1 = np.random.uniform(low=-1, high=1)
        self.wx2h2 = np.random.uniform(low=-1, high=1)
        self.wh1o1 = np.random.uniform(low=-1, high=1)
        self.wh2o1 = np.random.uniform(low=-1, high=1)

    def calcular_y(self):
        for i in range(0,4):
            uh1 = self.wx0h1 + self.wx1h1 * self.padrao[i][0] + self.wx2h1 * self.padrao[i][1] #padrao[linha][n da entrada]  ex: padrao[0][0] = x1 da primeira linha
            self.yh1 = 1 / (self.e ** (-uh1) + 1)
            uh2 = self.wx0h2 + self.wx1h2 * self.padrao[i][0] + self.wx2h2 * self.padrao[i][1]
            self.yh2 = 1 / (self.e ** (-uh2) + 1)
            uo1 = self.wx0o1 + self.yh1 * self.wh1o1 + self.yh2 * self.wh2o1
            self.yo1 = 1 / (self.e ** (-uo1) + 1)
            self.y[i] = self.yo1
            self.yd_y[i] = self.desejado[i] - self.y[i]


    def calcular_mse(self):
        soma = 0
        for i in range(0,4):
            soma += self.yd_y[i] ** 2
        self.mse = (1 / 4) * soma
        return self.mse

    def treinar(self):
        while self.calcular_mse() < 1 and self.epoca < 10000:
            self.epoca += 1
            for i in range(0,4):
                #calculo do y
                uh1 = self.wx0h1 + self.wx1h1 * self.padrao[i][0] + self.wx2h1 * self.padrao[i][1]
                self.yh1 = 1 / (self.e ** (-uh1) + 1)
                uh2 = self.wx0h2 + self.wx1h2 * self.padrao[i][0] + self.wx2h2 * self.padrao[i][1]
                self.yh2 = 1 / (self.e ** (-uh2) + 1)
                uo1 = self.wx0o1 + self.yh1 * self.wh1o1 + self.yh2 * self.wh2o1
                self.yo1 = 1 / (self.e ** (-uo1) + 1)
                self.y[i] = self.yo1
                self.yd_y[i] = self.desejado[i] - self.y[i]
                #calculo do delta
                self.deltao1 = self.yd_y[i] * (self.yo1 * (1 - self.yo1))
                self.deltah1 = (self.yh1 * (1 - self.yh1)) * self.wh1o1 * self.deltao1
                self.deltah2 = (self.yh2 * (1 - self.yh2)) * self.wh2o1 * self.deltao1
                #calculo do gradiente
                gx0h1 = self.deltah1
                gx0h2 = self.deltah2
                gx0o1 = self.deltao1
                gx1h1 = self.padrao[i][0] * self.deltah1
                gx1h2 = self.padrao[i][0] * self.deltah2
                gx2h1 = self.padrao[i][1] * self.deltah1
                gx2h2 = self.padrao[i][1] * self.deltah2
                gh1o1 = self.yh1 * self.deltao1
                gh2o1 = self.yh2 * self.deltao1
                #atualizacao dos pesos
                self.dwx0h1 = self.n * gx0h1 + self.l * self.dwx0h1
                self.wx0h1 = self.dwx0h1 + self.wx0h1

                self.dwx0h2 = self.n * gx0h2 + self.l * self.dwx0h2
                self.wx0h2 = self.dwx0h2 + self.wx0h2

                self.dwx0o1 = self.n * gx0o1 + self.l * self.dwx0o1
                self.wx0o1 = self.dwx0o1 + self.wx0o1

                self.dwx1h1 = self.n * gx1h1 + self.l * self.dwx1h1
                self.wx1h1 = self.dwx1h1 + self.wx1h1

                self.dwx1h2 = self.n * gx1h2 + self.l * self.dwx1h2
                self.wx1h2 = self.dwx1h2 + self.wx1h2

                self.dwx2h1 = self.n * gx2h1 + self.l * self.dwx2h1
                self.wx2h1 = self.dwx2h1 + self.wx2h1

                self.dwx2h2 = self.n * gx2h2 + self.l * self.dwx2h2
                self.wx2h2 = self.dwx2h2 + self.wx2h2

                self.dwh1o1 = self.n * gh1o1 + self.l * self.dwh1o1
                self.wh1o1 = self.dwh1o1 + self.wh1o1

                self.dwh2o1 = self.n * gh2o1 + self.l * self.dwh2o1
                self.wh2o1 = self.dwh2o1 + self.wh2o1

#PORTA XOR
padrao = [[0, 0], [0, 1], [1, 0], [1, 1]]
desejado = [0, 1, 1, 0]

m = Mlp(padrao, desejado)
m.calcular_y() #primeiro calculo Y
m.calcular_mse()
m.treinar()
print("MSE: " + str(m.mse))
for i in range(0,4):
    print("x1: " + str(padrao[i][0]) + " x2: " + str(padrao[i][1]) + " y: " + str(m.y[i]))
