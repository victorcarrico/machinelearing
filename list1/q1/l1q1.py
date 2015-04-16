import csv
import numpy as np
import time
import matplotlib.pyplot as plt

def getTime():
	return int(round(time.time() * 1000))

def getData(filename):#recebe os dados do aquivo scv e randomiza as entradas
	file = open(filename, 'rb')
	data = csv.reader(file, delimiter=',')
	matrix = [row for row in data]
	#for row in matrix:
	#	print (row)
	#print 'matriz direto do CSV'
	#print matrix[0]
	#print matrix[len(matrix)-1]
	if matrix[(len(matrix)-1)] == []:	
		matrix = np.delete(matrix,(len(matrix)-1))
	np.random.shuffle(matrix)
	return matrix

def formatRow(rowIn, rowType):#formata corretamente uma linha com strings para o vetor de entrada['cat','num','ignore']
	#print 'row antes'
	#print rowIn
	ignore = 'ig'
	for y in range(len(rowIn)-1):
		if rowType[y] == 'num':
			rowIn[y] = num(rowIn[y])
		elif rowType[y] == 'cat':
			s = rowIn[y][:]
			rowIn[y] = s
		elif rowType[y] == 'ignore':
			rowIn[y] = ignore
	removeIgnores(rowIn)
	#print 'row depois'
	#print rowIn

def removeIgnores(row):
	for i in range(len(row)):
		if row[i] == 'ig':
			row.pop(i)
			removeIgnores(row)
			break

def rowClassToLast(row, currentClassPosition):
	temp = row[len(row)-1]
	row[len(row)-1] = row[currentClassPosition]
	row[currentClassPosition] = temp

def toNumColumn(row, indexToNum):
	row[indexToNum] = num(row[indexToNum])

def normalizeColumn(matrix, index):
	column = _getColumn(matrix, index)
	a = max(column) - min(column)
	for x in matrix:
		x[index] = (x[index] - min(column))/a


def _getColumn(matrix,index):
	ret = []
	for y in matrix:
		ret.append(y[index])
	return ret

def formatData(matrix, rowType, classPosition): #formata toda a matriz de acordo com o rowType definido
	for row in matrix:
		rowClassToLast(rowType, classPosition)
		rowClassToLast(row, classPosition)
		formatRow(row,rowType)

def num(s):
	try:
		return int(s)
	except ValueError:
		return float(s)

def splitDataSet (a): #divide na proporcao de 70% / 30%
	s = int(len(a)*0.7)
	treinamento = a[:s]
	teste = a[s:]
	return [treinamento,teste]

def getIrisData ():#[0] treinamento, [1] teste
	mat = getData('iris.data')
	formato = ['num','num','num','num','cat']
	formatData(mat, formato,4)
	normalizeColumn(mat,0)
	normalizeColumn(mat,1)
	normalizeColumn(mat,2)
	normalizeColumn(mat,3)
	return splitDataSet(mat)

def getGlassData():
	#https://archive.ics.uci.edu/ml/datasets/Glass+Identification
	#Glass Identification Data Set 
	mat = getData('glass.data')
	formato = ['ignore','num','num','num','num','num','num','num','num','num','cat'] 
	formatData(mat, formato,10)
	normalizeColumn(mat,0)
	normalizeColumn(mat,1)
	normalizeColumn(mat,2)
	normalizeColumn(mat,3)
	normalizeColumn(mat,4)
	normalizeColumn(mat,5)
	normalizeColumn(mat,6)
	normalizeColumn(mat,7)
	normalizeColumn(mat,8)
	return splitDataSet(mat)

def euclideanDist(a, b):
	distance = 0
	for x in range(len(a)-1):
		#somatorio do quadrado das diferencas dos atributos
		distance += (a[x] - b[x])**2
	return (distance**0.5)

def kNN_list(treinoSet, test, k):
	ret = []
	for t in treinoSet:
		d = euclideanDist(t,test)
		ret.append((d,t[len(t)-1]))
	ret = sorted(ret)
	ret = ret[:k]	
	return ret

def kNN_iris (treino, test, k, w):# is peso
	lst = kNN_list(treino, test, k)
	qtd_Iris_virginica = 0
	qtd_Iris_setosa = 0
	qtd_Iris_versicolor = 0 
	z=0.0000000000000001
	for i in lst:
		if i[1] == 'Iris-setosa':
			qtd_Iris_setosa += 1/((i[0])**w+z)
		elif i[1] == 'Iris-versicolor':
			qtd_Iris_versicolor += 1/((i[0])**w+z)
		elif i[1] == 'Iris-virginica':	
			qtd_Iris_virginica += 1/((i[0])**w+z)
	out = []
	out.append((qtd_Iris_virginica,'Iris-virginica'))
	out.append((qtd_Iris_versicolor,'Iris-versicolor'))
	out.append((qtd_Iris_setosa,'Iris-setosa'))
	out = sorted(out)
	return out[-1][1]

def kNN_glass (treino, test, k, w):# is peso
	lst = kNN_list(treino, test, k)
	qtd = [0.0]*7 
	z=0.0000000000000001
	for i in lst:
		if i[1] == '1':
			qtd[0] += 1/((i[0])**w+z)
		elif i[1] == '2':
			qtd[1]+= 1/((i[0])**w+z)
		elif i[1] == '3':	
			qtd[2]+= 1/((i[0])**w+z)
		elif i[1] == '4':	
			qtd[3] += 1/((i[0])**w+z)
		elif i[1] == '5':	
			qtd[4] += 1/((i[0])**w+z)
		elif i[1] == '6':	
			qtd[5]+= 1/((i[0])**w+z)
		elif i[1] == '7':	
			qtd[6] += 1/((i[0])**w+z)	
	out = []
	out.append((qtd[0],'1'))
	out.append((qtd[1],'2'))
	out.append((qtd[2],'3'))
	out.append((qtd[3],'4'))
	out.append((qtd[4],'5'))
	out.append((qtd[5],'6'))
	out.append((qtd[6],'7'))
	out = sorted(out)
	return out[-1][1]

def accuracyRate_iris(tr1,te1,k,w):
	acertos = 0
	#print kNN_iris(tr1,te1[0],5,0)
	for t in te1:
		if t[-1] == kNN_iris(tr1,t,k,w):
			acertos += 1	
	return ((float(acertos)/(len(te1)))*100)

def accuracyRate_glass(tr1,te1,k,w):
	acertos = 0
	#print kNN_iris(tr1,te1[0],5,0)
	for t in te1:
		if t[-1] == kNN_glass(tr1,t,k,w):
			acertos += 1	
	return ((float(acertos)/(len(te1)))*100)




def main():	
	print '================AP.MAQ. L1Q1================='
	kVector = [1,2,3,5,7,9,11,13,15]
	a1 = getIrisData()
	tr1 = a1[0]
	te1 = a1[1]

	a3 = getGlassData()
	tr3 = a3[0]
	te3 = a3[1]

	

	plt.xlabel('Valor de k')
	plt.ylabel('Taxa de Acerto (pct)')
	
	print 'Iris Database '
	for w in [0,1]:
		aV1 = []
		tV1 = []
		if w == 0:
			print ' k-NN sem peso:'
			plt.title('k-NN sem peso - Iris Database')
		elif w==1:
			print ' k-NN com peso:'
			plt.title('k-NN com peso - Iris Database')
		for k in kVector:
			ti = getTime()
			a = accuracyRate_iris(tr1,te1,k,w)
			tf = getTime()
			t = float(tf-ti)/1000
			print '   k = %d, acerto: %f %% , em %f seg' % (k, a, t)
			aV1.append(a)
			tV1.append(t)
		plt.plot(kVector, aV1, 'ro-')
		plt.show()

	print '--------------'
	
	print 'Glass Database '
	for w in [0,1]:
		aV1 = []
		tV1 = []
		if w == 0:
			print ' k-NN sem peso:'
			plt.title('k-NN sem peso - Glass Database ')
		elif w==1:
			print ' k-NN com peso:'
			plt.title('k-NN com peso - Glass Database ')
		for k in kVector:
			ti = getTime()
			a = accuracyRate_glass(tr3,te3,k,w)
			tf = getTime()
			t = float(tf-ti)/1000
			print '   k = %d, acerto: %f %% , em %f seg' % (k, a, t)
			aV1.append(a)
			tV1.append(t)
		plt.plot(kVector, aV1, 'ro-')
		plt.show()
main()
