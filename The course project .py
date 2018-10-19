
# coding: utf-8

# In[ ]:


#First Session Code : Intro to Python

import math,random, matplotlib.pyplot as plt,scipy.stats
from scipy import stats as ss

heights = [180, 169, 161, 179, 175, 175,
           184, 176, 179, 164, 168, 172]

def biggest(y):
    maxi = y[0]
    for x in y:
        if x > maxi:
            maxi = x
        else:
            maxi = maxi
    return maxi
biggest ([3,4,5,6,6,78])

def smallest(y):
    mini = y[0]
    for x in y:
        if x < mini:
            mini = x
        else:
            mini = mini
    return mini

smallest([3,4,5,6,6,78])

def summation(y):
    tot = 0
    for i in range(len(y)):
        tot += y[i]
    return tot
summation([1,2,3,4,5])

def product(y):
    tot = 1
    for i in range(len(y)):
        tot *= y[i]
    return tot
product([1,2,3,4,5])

def factorial(n):
    #here we use recursion for calculation
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)
factorial (7)

def search(x,y):
    count = 0
    while count < len(y):
        if y[count] == x:
            return True
        else: 
            count += 1
            
search(5,[1,2,3,4,5,5])
###############################################################################################
#Second Session Code : Histograms

def bins(Sample):
    n = len(Sample)
    return 1 + int(1.44 * math.log(n))
bins(heights)

def bounds(Sample):
    k = bins(Sample)
    mini = Sample[0]
    #define width of each bin
    height = 1.0 *(biggest(Sample) - smallest(Sample))/ k
    res = []
    for i in range(1,k+1):
        xlow = smallest(Sample) + (i-1)* height
        xhigh = smallest(Sample) + i * height
        res.append((xlow,xhigh))
    return res
bounds(heights)

def Frequencies(Sample):
    k = bins(Sample)
    bs = bounds(Sample)
    res = [0 for _ in range(k)]
    for i in Sample:
        for j in range(len(bs)):
            if i >= bs[j][0] and i <= bs[j][1]:
                res[j] += 1            
    return res
Frequencies(heights)

###############################################################################################
#Session 3,4 Code: Statistics

def Expectation(x):
    n = len(x)
    tot = 0
    for i in range (n):
        tot = tot + (1/n)*x[i];
    return tot
Expectation(heights)

def median(y):
    #here we test at first whether the list length is even or odd
    y.sort()
    n = len(y)
    if n % 2 == 1:
        return y[int((n+1)/2)]
    elif n % 2 == 0:
        return .5 * (y[int(n/2)]+y[int((n+2)/2)])
median(heights)

def mode(t):
    occurence = 0
    res = []
    for n in t:
        if t.count(n)>occurence:
            occurence = t.count(n)
            res.append(occurence)
        occurence = 0
    resunique = set(res)
    for j in t:
        if t.count(j) == max(resunique):
            return j
mode(heights)

def cdf(t,x):
    t.sort()
    prob = 0
    for j in t:
        if j <= x:
            prob = prob + (1/len(t))
        elif j> x:
            prob = prob
    return prob
cdf(heights,178)

def quantile(lst,prob):
    lst.sort()
    listt = list(set(lst))
    freq = dict()
    for n in listt:
        freq[n] = cdf(lst,n)
    for j in range(len(listt)):
        if prob == freq[listt[j]]:
            return listt[j]
        elif prob > freq[listt[j]] and prob < freq[listt[j+1]]:
            X_L = listt[j]
            X_H = listt[j+1]
            P_L = cdf(lst,listt[j])
            P_H = cdf(lst,listt[j+1])
            D = prob - P_L
            B = P_H - P_L
            C = X_H - X_L
            A = (D/B)*C
            X_star = X_L + A
            return X_star
quantile(heights,0.73)
def standarddeviation(lst):
    lst.sort()
    X = Expectation(lst)
    N = len(lst)
    stdev = 0
    for i in range(N):
        stdev += (1/N) * (lst[i] - X) * (lst[i] - X)
    return math.sqrt(stdev)
standarddeviation(heights)

def skewness(lst):
    m = median(lst)
    d = mode(lst)
    sigma = standarddeviation(lst)
    w = Expectation(lst)
    A_d = (w-d)/sigma
    A_m = (w-m)/sigma
    return A_d, A_m
skewness(heights)

###############################################################################################
#Session 5,6 Code: Chi-Squared test
def norm(m,s):
    '''sum_r = 0
    for i in range(12):
        sum_r += random.random()
    sum_r -= 5'''
    return m + s * (sum([random.random() for _ in range(12)]))
xs = [norm(10,2) for _ in range(100)]
xs
plt.hist(xs,bins = 16)
plt.show()

def ChiTest(Sample):
    #we will hypothesize natural distribution
    k = bins(Sample)
    m = bounds(Sample)
    n = Frequencies(Sample)
    SD = standarddeviation(Sample)
    mean = Expectation(Sample)
    #now we calculate the theoretical frequencies of the sample
    firstcdf = []
    secondcdf = []
    df = k-3
    p = 0.95
    chiinverse = scipy.stats.chi2.ppf(p,df)
    for i in range (k):
        cdf_min = scipy.stats.norm.cdf(m[i][0],mean,SD)
        cdf_max = scipy.stats.norm.cdf(m[i][1],mean,SD)
        firstcdf.append((cdf_min,cdf_max))
        secondcdf.append(k * (cdf_max - cdf_min))
        #now it's time to find test value
    lis = []
    for i in range (len(n)):
        term = ((secondcdf[i] - n[i])**2)/secondcdf[i]
        lis.append(term)
    chivalue = sum(lis)
    if chivalue <= chiinverse:
        return True
    else:
        return False        
ChiTest (xs)

###############################################################################################
#kolmogorov smirnov test
def Freqs(Sample):
    Sample.sort()
    uniq = list(set(Sample))
    res = [0 for _ in range(len(uniq))]
    for i in range(len(uniq)):
        res[i] = Sample.count(uniq[i])
    return res
Freqs(heights)

def Ranks(Sample):
    rk = 0 
    Sample.sort()
    res = []
    i = 0
    uniq = list(set(Sample))
    freq = Freqs(Sample)
    while i < len(uniq):
        if freq[i] == 1:
            res.append(rk + 1)
        else:
            term = 0
            templis = []
            for i in range(freq[i]):
                templis.append(rk + i)
                term += templis[i]
            res.append(1.0 * (term / freq[i]))
        rk += freq[i]
        i += 1
    return res

Ranks(heights)

def Kromtest(Sample):
    uniques = list(set(Sample))
    N = len( Sample)
    rs = Ranks(Sample)
    r = Expectation(Sample)
    SD = standarddeviation(Sample)
    difs = []
    for i in range (len(uniques)):
        FX = scipy.stats.norm.cdf(uniques[i],r,SD)
        D_minus = FX - (rs[i]- 1)/ N
        D_plus =( rs[i]/ N ) - FX
        difs.append((D_minus, D_plus))
    D = max(max(difs))
    K = math.sqrt( (-.5*math.log((alpha/2))))
    return D <= K
Kromtest(heights)

###############################################################################################
#Regression Analysis
A = [[1, 2],
     [3, 4],
     [5, 6]]
B = [[1, 2],
     [3, 4]]
def Transpose(matrix):
    res = []
    newlist = []
    for i in range(len(matrix[0])):
        for j in range (len(matrix)):
            newlist.append(matrix[j][i])
        res.append(newlist)
        newlist = []
    return res

def dotprdct(A,B):
    C = [[0 for x in range(len(A))] for y in range(len(B[0]))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k]*B[k][j]
    return C

def det2 (A):
    return A[0][0] * A[1][1] - A[1][0] * A[0][1]

def minor3(A, i, j):
    m = []
    for ii in range(3):
        if ii != i:
            row = []
            for jj in range(3):
                if jj != j:
                    row.append(A[ii][jj])
            m.append(row)
    return det2(m)

def compl3(A,i,j):
    return (-1.0) ** (i + j) * minor3(A, i, j)

det3 = lambda A: sum([A[1][i] * compl3(A, 1, i) for i in range(3)])

def adj3(A):
    res = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(compl3(A, j, i))
        res.append(row)
    return res

def inv3(A):
    d = 1.0 * det3(A)
    a = adj3(A)
    res = []
    for i in range(3):
        for j in range(3):
            a[i][j] = a[i][j] / d
    return a

X = [[1, 2, 1],
     [1, -1, 1],
     [0, 3, 1],
     [0, 1, 1],
     [1, 0, 1]]
y = [[2],
     [1],
     [5],
     [4],
     [2]]

import numpy as np
X = np.matrix(X)
y = np.matrix(y)
X.T

from sklearn.datasets import load_iris
iris = load_iris()

X = [list(x) for x in iris.data[:, 1:3]]
for x in X:
    x.append(1.0)
y = iris.data[:, 3]
X, y = np.matrix(X), np.matrix(y).T

from matplotlib import pyplot

def R2(X, y):
    # take the sample size
    n = len(y)
    # calculate the linear regression coefficients
    b = X.T.dot(X).I.dot(X.T.dot(y))
    # calculate theoretical values of the dependant variable
    yt = [b[0, 0] * X[i, 0] + b[1, 0] * X[i, 1] + b[2, 0]
          for i in range(n)]
    # calculate residuals
    es = [yt[i] - y[i, 0] for i in range(n)]
    
    # show the distribution of residuals
    pyplot.hist(es, bins=8)
    pyplot.show()
    
    return 1.0 - np.array(es).var() / y.var()

R2(X, y)
  


# In[3]:


heights = [180, 169, 161, 179, 175, 175,
           184, 176, 179, 164, 168, 172]

def Kromtest(Sample):
    uniques = list(set(Sample))
    N = len( Sample)
    rs = Ranks(Sample)
    r = Expectation(Sample)
    SD = standarddeviation(Sample)
    difs = []
    for i in range (len(uniques)):
        FX = scipy.stats.norm.cdf(uniques[i],r,SD)
        D_minus = FX - (rs[i]- 1)/ N
        D_plus =( rs[i]/ N ) - FX
        difs.append((D_minus, D_plus))
    D = max(max(difs))
    K = math.sqrt( (-.5*math.log((alpha/2))))
    return D <= K
Kromtest(heights)

