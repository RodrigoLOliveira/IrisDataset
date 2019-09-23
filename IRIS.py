import numpy as np
from sklearn import datasets
from pybrain.datasets.classification import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
import matplotlib.pyplot as plt

iris = datasets.load_iris()

entrada, saida = iris.data, iris.target

dataset = ClassificationDataSet(4, 1, nb_classes=3)

#Adicionar as amostras ao dataset
for i in range(len(entrada)):
    dataset.addSample(entrada[i], saida[i])

#Recuperar dados para realizar o treinamento da rede
parteTreino, parteDados = dataset.splitWithProportion(0.6)
print("Quantidade para treinamento da rede : " + str(len(parteTreino)))

#Separando a parte de dados para realização do teste e para a validação da rede
teste, validacao = parteDados.splitWithProportion(0.5)
print("Quantidade para teste da rede : " + str(len(teste)))
print("Quantidade para validação da rede : " + str(len(validacao)))

#Criando a rede
rede = buildNetwork(dataset.indim, 3, dataset.outdim)

#Realizando o treinamento e recuperando os erros
treinamento = BackpropTrainer(rede, dataset=parteTreino, learningrate=0.01, momentum=0.1, verbose=True)
erroTreino, erroVal = treinamento.trainUntilConvergence(dataset = parteTreino, maxEpochs=100)

plt.plot(erroTreino, 'b', erroVal, 'r')
plt.show()