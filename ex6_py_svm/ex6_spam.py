import scipy as sp
import re
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm
from nltk.stem.porter import PorterStemmer
from functools import reduce

# load vocabulary list
vocab_list = sp.genfromtxt('vocab.txt', dtype='str', usecols=(1,))
n = vocab_list.size

# Load sample
f = open('emailSample1.txt')
str = f.read()

def process(content):
  content = content.lower()
  content = re.sub('<[^<>]+>', ' ', content)
  content = re.sub('[0-9]+', 'number', content)
  content = re.sub('(http|https)://[^\s]*', 'httpaddr', content)
  content = re.sub('[^\s]+@[^\s]+', 'emailaddr', content)
  content = re.sub('[$]+', 'dollar', content)
  content = re.sub('[^\s|^\w]', '', content)
  content = list(filter(None, content.split()))

  # Stem
  stemmer = PorterStemmer()
  def find_vocab_index(e):
    i = sp.where(vocab_list == e)[0]
    return i[0] if len(i) > 0 else None

  word_indices = sp.array(list(filter(None, map(find_vocab_index, map(stemmer.stem, content)))))
  return word_indices

# Preprocess
word_indices = process(str)

def update_feature(acc, curr):
  acc[curr] = 1
  return acc

# Map feature vector
email_feature = reduce(update_feature, word_indices, sp.zeros(n))

# Load training data
data = loadmat('spamTrain.mat')
X = sp.matrix(data['X'])
Y = sp.matrix(data['y'])

# Load test data
data = loadmat('spamTest.mat')
X_test = sp.matrix(data['Xtest'])
Y_test = sp.matrix(data['ytest'])

# Train
C = 0.1
model = svm.LinearSVC(C=C)
model.fit(X, sp.array(Y).ravel())
p = model.predict(X_test)
a = model.score(X_test, Y_test) * 100
print("Test Accuracy: {a}%".format(**locals()))

# Sort weights
weights = model.coef_.ravel()

print('Top predictors of spam:')
for i in sp.argsort(weights)[-10:][::-1].tolist():
  word = vocab_list[i]
  weight = sp.around(weights[i], 2)
  print("{word}: {weight}".format(**locals()))

# Predict a sample email
f = open('spamSample1.txt')
str = f.read()
wi = process(str)
x = reduce(update_feature, wi, sp.zeros(n))
p = model.predict(sp.array([x]))[0]

print("Processed Spam Classification on spamSample1.txt: {p}".format(**locals()))
print('(1 indicates spam, 0 indicates not spam)')
