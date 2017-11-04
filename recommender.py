import heapq
import numpy
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as linalg
import random
import math

class CollaborativeFiltering():

	def __init__(self,matrix):
		self.__matrix=matrix
		self.__m=self.__matrix.shape[0]
		self.__n=self.__matrix.shape[1]
		self.__means=numpy.ndarray((self.__m,1))
	
	def setMeans(self):
		for i in range(self.__m):
			self.__means[i]=self.__matrix[i].mean()
	
	def similarity(self,i,j):
		#print(i,j)
		ans=0

		t1=self.__matrix[i]-self.__means[i]
		t2=self.__matrix[j]-self.__means[j]
		
		ans=numpy.dot(t1,t2)
		
		m1=math.sqrt(numpy.dot(t1,t1))
		m2=math.sqrt(numpy.dot(t2,t2))

		try:
			return ans/(m1*m2)
		except:
			return 0


	def predictRating(self,i,j):
		#print(i,j)
		k=5
		a=[]
		b=[]
		avg=self.__means[i][0]
		for l in range(self.__m):
			if self.__matrix[l][j]!=0:
				
				a.append((-1*self.similarity(i,l),l))
		
		heapq.heapify(a)
		try:
			p=len(a)
			for l in range(min(k,p)):
				b.append(heapq.heappop(a))
			
			ans=0
			dr=0
			for l in range(len(b)):
				ans=ans+((-1*b[l][0])*self.__matrix[b[l][1]][j])
				dr=dr-b[l][0]
			
			if dr==0:
				#print(i,j)
				return avg
			
			ans=ans/dr
			return ans
		
		except IndexError:
			return avg


	def predict(self):
		m=self.__m//4
		n=self.__n//4
		rows=set()
		cols=set()
		
		#Randomly select m/4 rows and n/4 columns. Thus 75% of data is for training and 25% of data is testing
		for i in range(m):
			rows.add(random.randrange(self.__m))
		for i in range(n):
			cols.add(random.randrange(self.__n))

		rows=list(rows)
		cols=list(cols)

		#print(rows,cols)
		#input()

		#Matrix which stores the calculated results
		m1=self.__matrix.copy()

		for i in rows:
			for j in cols:
				self.__matrix[i][j]=0

		self.setMeans()

		#RMSE and Spearman
		'''
		diff=0
		n=0
		for i in rows:
			for j in cols:
				if m1[i][j]!=0:
					#m1[i][j]=self.predictRating(i,j)
					diff=diff+((self.predictRating(i,j)-m1[i][j])**2)
					n=n+1
		
		rmse=math.sqrt(diff/n)
		print(diff)
		spearman=1-((6*diff)/(n*((n**2)-1)))
		print(rmse)
		print(spearman)
		'''

		#'''
		#Precision for Top 10 movies of 10 random users
		users=[]
		n_users=250	
		k=10
		threshold=0.5
		nr=0
		dr=0
		for i in range(n_users):
			users.append(random.randrange(self.__m))

		for i in users:
			ratings=[]
			for j in range(self.__n):
				if m1[i][j]!=0:
					ratings.append((-1*m1[i][j],j))

			heapq.heapify(ratings)
			for j in range(min(len(ratings),k)):
				t=heapq.heappop(ratings)
				r=self.predictRating(i,t[1])
				dr=dr+1
				#print(r,-1*t[0])
				if abs(r-(-1*t[0])) < threshold/2:
					nr=nr+1

		precision=nr/dr
		print(precision)
		#'''
		self.__matrix=m1