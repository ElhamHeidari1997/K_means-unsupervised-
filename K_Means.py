import sklearn
import numpy as np
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits()
data=scale(digits.data)

y= digits.target
k=10
samples, features = data.shape

clf= KMeans(n_clusters=k,init="random" ,n_init=10 )
clf.fit(data)
