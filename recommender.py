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

class CollaborativeFilteringBaseline():

	def __init__(self,matrix):
		self.__matrix=matrix.transpose()
		self.__m=self.__matrix.shape[0]
		self.__n=self.__matrix.shape[1]
		self.__means=numpy.ndarray((self.__m,1))
		self.__rowmeans=numpy.ndarray((self.__m,1))
		self.__colmeans=numpy.ndarray((self.__n,1))
	
	def setMeans(self):
		for i in range(self.__m):
			self.__rowmeans[i]=self.__matrix[i].mean()
		matrix=self.__matrix.transpose()
		for i in range(self.__n):
			self.__colmeans[i]=matrix[i].mean()
		self.__u=self.__rowmeans.mean()

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
			#print(ans,m1,m2)
			#print(self.__matrix[i],self.__means[i],self.__matrix[j],self.__means[j])
			#print(self.__matrix[i]-self.__means[i],self.__matrix[j]-self.__means[j])
			#input()
			return 0

	def predictRating(self,i,j):
		#print(i,j)
		k=5
		a=[]
		b=[]
		avg=self.__u+(self.__rowmeans[i][0]-self.__u)+(self.__colmeans[j][0]-self.__u)
		#print(self.__u,self.__rowmeans[i][0],self.__colmeans[j][0])
		#print(avg)
		#input()
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
				ans=ans+((-1*b[l][0])*(self.__matrix[b[l][1]][j]-(self.__u+(self.__rowmeans[i][0]-self.__u)+(self.__colmeans[j][0]-self.__u))))
				dr=dr-b[l][0]
			
			if dr==0:
				#print(i,j)
				return avg
			
			ans=ans/dr
			ans=ans+avg
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
			users.append(random.randrange(self.__n))

		for i in users:
			ratings=[]
			for j in range(self.__m):
				if m1[j][i]!=0:
					ratings.append((-1*m1[j][i],j))

			heapq.heapify(ratings)
			for j in range(min(len(ratings),k)):
				t=heapq.heappop(ratings)
				r=self.predictRating(t[1],i)
				dr=dr+1
				#print(r,-1*t[0])
				if abs(r-(-1*t[0])) < threshold/2:
					nr=nr+1

		precision=nr/dr
		print(precision)
		#'''
		self.__matrix=m1

class SVDRecommenderSystem():

	def __init__(self,matrix):
		self.__matrix=matrix
		self.__m=self.__matrix.shape[0]
		self.__n=self.__matrix.shape[1]

	def svd(self,m1):
		for ttt in range(1):
			pass
			#Method 1
			'''
			p=numpy.linalg.eig(m1.transpose().dot(m1))
			eigvals=p[0]
			v=p[1].transpose()
			sigma=numpy.zeros((self.__m,self.__n))	
			eigvals=[(eigvals[i],i) for i in range(len(eigvals))]
			eigvals=sorted(eigvals,reverse=True)
			print(sigma)
			print(v)
			
			for i in range(self.__n-1):
				v[i,:],v[eigvals[i][1],:]=v[eigvals[i][1],:],v[i,:].copy()
			
			for i in range(1,self.__n):
				s=numpy.zeros((1,self.__n))
				for j in range(i):
					s=s+((v[j,:].dot(v[i,:]))*v[j,:])
					#print(s)
					#print(s.shape)
					#input()

				#print(s.shape)
				#print(v[i,:])
				#print(s)
				#print(v[i,:]-s)
				#input()
				v[i,:]=v[i,:]-s
				v[i,:]/=math.sqrt(v[i,:].dot(v[i,:]))
			'''	
			'''
			for i in range(self.__m):
				sigma[i][i]=eigvals[i][0]**0.5
			u=numpy.zeros((self.__m,self.__m))
			for i in range(self.__m):
				print(m1)
				print(m1.dot((v[i,:].transpose())))
				u[:,i]=((m1.dot(v[i,:].transpose()))/eigvals[i][0])
				input()
			'''
			
			#Method 2
			'''
			sigma=numpy.zeros((self.__m,self.__n))	
			p1=numpy.linalg.eig(m1.transpose().dot(m1))
			eigvals1=p1[0]
			v=p1[1].transpose()
			p=numpy.linalg.eig(m1.dot(m1.transpose()))
			eigvals=p[0]
			u=p[1]

			eigvals=[(eigvals[i],i) for i in range(len(eigvals))]
			eigvals=sorted(eigvals,reverse=True)

			for i in range(self.__m-1):
				u[:,i],u[:,eigvals[i][1]]=u[:,eigvals[i][1]],u[:,i].copy()
			for i in range(1,self.__m):
				s=numpy.zeros((1,self.__m))
				for j in range(i):
					s=s+((u[:,j].dot(u[:,i]))*u[:,j])

				u[:,i]=u[:,i]-s
				u[:,i]/=math.sqrt(u[:,i].dot(u[:,i]))


			eigvals1=[(p1[0][i],i) for i in range(len(p1[0]))]
			eigvals1=sorted(eigvals1,reverse=True)

			for i in range(self.__n-1):
				v[i,:],v[eigvals1[i][1],:]=v[eigvals1[i][1],:],v[i,:].copy()

			for i in range(1,self.__n):
				s=numpy.zeros((1,self.__n))
				for j in range(i):
					s=s+((v[j,:].dot(v[i,:]))*v[j,:])
				v[i,:]=v[i,:]-s
				v[i,:]/=math.sqrt(v[i,:].dot(v[i,:]))

			if self.__m <= self.__n:
				for i in range(self.__m):
					t=(m1.dot(v[i,:]))/math.sqrt(eigvals[i][0])
					if ((t+u[:,i])<(10**-10)*numpy.ones(t.shape)).all():
						u[:,i]=u[:,i]*-1

				i=0
				for e in eigvals:
					sigma[i][i]=e[0]**0.5
					i=i+1
			else:
				for i in range(self.__n):
					t=(m1.transpose().dot(u[:,i]))/math.sqrt(eigvals1[i][0])
					t=t.transpose()
					if ((t+v[i,:])<(10**-10)*numpy.ones(t.shape)).all():
						v[i,:]=v[i,:]*-1

				i=0
				for e in eigvals1:
					sigma[i][i]=e[0]**0.5
					i=i+1
			
			'''

		#Method 3
		if self.__m<self.__n:
			p=numpy.linalg.eigh(m1.transpose().dot(m1))
			v=p[1].transpose()
			eigvals=p[0]
			eigvals=[(eigvals[i],i) for i in range(len(eigvals))]
			eigvals=sorted(eigvals,reverse=True)
			order=[e[1] for e in eigvals]
			
			v=v[order,:]

			sigma=numpy.zeros((self.__m,self.__n))
		
			for i in range(self.__m):
				sigma[i][i]=math.sqrt(eigvals[i][0])

			t=m1.dot(v.transpose())
			
			u=numpy.zeros((self.__m,self.__m))
			for i in range(self.__m):
				u[:,i]=t[:,i]/sigma[i][i]
				u[:,i]/=math.sqrt(u[:,i].dot(u[:,i]))
		else:
			p=numpy.linalg.eigh(m1.dot(m1.transpose()))#,k=self.__n)
			u=p[1]
			eigvals=p[0]
			eigvals=[(eigvals[i],i) for i in range(len(eigvals))]
			eigvals=sorted(eigvals,reverse=True)
			order=[e[1] for e in eigvals]
			
			u=u[:,order]

			sigma=numpy.zeros((self.__m,self.__n))
		
			for i in range(self.__n):
				sigma[i][i]=math.sqrt(eigvals[i][0])

			t=u.transpose().dot(m1)
			
			v=numpy.zeros((self.__n,self.__n))
			for i in range(self.__n):
				v[i,:]=t[i,:]/sigma[i][i]
				u[i,:]/=math.sqrt(u[i,:].dot(u[i,:]))

		return (u,sigma,v)
		
	def precision(self,m,n_users=250,k=10):
		#Precision for Top 10 movies of 10 random users
		users=[]
		threshold=0.5
		nr=0
		dr=0
		for i in range(n_users):
			users.append(random.randrange(self.__n))

		for i in users:
			ratings=[]
			for j in range(self.__m):
				if self.__matrix[j][i]!=0:
					ratings.append((-1*m[j][i],j))

			heapq.heapify(ratings)
			for j in range(min(len(ratings),k)):
				t=heapq.heappop(ratings)
				r=m[t[1]][i]
				dr=dr+1
				#print(r,-1*t[0])
				if abs(r-(-1*t[0])) < threshold/2:
					nr=nr+1

		precision=nr/dr
		print("Precision in top {} for {} users: {}".format(k,n_users,precision))
	

	def rmse(self,m,rows,cols):
		#RMSE and Spearman
		#'''
		#print(rows)
		#print(cols)
		diff=0
		n=0
		for i in rows:
			for j in cols:
				if self.__matrix[i][j]!=0:
					#m1[i][j]=self.predictRating(i,j)
					#print(m[i][j],m1[i][j])
					diff=diff+((m[i][j]-self.__matrix[i][j])**2)
					n=n+1
		
		
		rmse=math.sqrt(diff/n)
		spearman=1-((6*diff)/(n*((n**2)-1)))
		print("RMSE: {}".format(rmse))
		print("Spearman Rank Correlation: {}".format(spearman))
		#'''
	
	def reduceenergy(self,sigma,percent=0.9):
		s=0
		for i in range(min(self.__m,self.__n)):
			s+=(sigma[i][i]**2)
		s1=s
		for i in range(min(self.__m,self.__n)-1,-1,-1):
			s-=(sigma[i][i]**2)
			if s/s1 <= 0.9:
				print(i)
				break				
		return i

	def predict(self):
		#'''
		m=self.__m//2
		n=self.__n//2
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

		#Matrix with test data removed
		m1=self.__matrix.copy()

		for i in rows:
			for j in cols:
				m1[i][j]=0
		#'''
		#m1=csr_matrix(m1)
		svd=self.svd(m1)
		
		u=svd[0]
		sigma=svd[1]
		v=svd[2]
		m=u.dot(sigma.dot(v))
		
		print("SVD with 100% energy")
		self.rmse(m,rows,cols)
		self.precision(m)

		r=self.reduceenergy(svd[1])
		print(u.shape[0],v.shape[1])
		print(r)	
		u=u[:,:r]
		sigma=sigma[:r,:r]
		v=v[:r,:]
		
		m=u.dot(sigma.dot(v))
		return r
