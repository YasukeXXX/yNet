import numpy as np
class layer(object):
	def __init__(self,num_in,num_out):
		self.input_size = num_in
		self.output_size = num_out
		self.W = np.random.rand(num_out,num_in)-0.5
		self.b = np.random.rand(num_out)-0.5
		self.delta = np.zeros(num_out)
		self.output = np.zeros(num_out)
		self.actfunc = f(self.output)
	def forward(self,x):
		self.actfunc.x = np.dot(self.W,x)+self.b
		self.output = self.actfunc.func()
		return self.output
	def backward(self,delta):
		self.delta = delta*self.actfunc.dfunc()
class f(object):
	def __init__(self,x):
		self.x = x
	def __call__(self):
		return self.x
	def func(self):
		return np.tanh(self.x)
	def dfunc(self):
		return 1-self.func()**2
class softmax(f):
	def __init__(self,x):
		super(softmax,self).__init__(x)
	def func(self):
		return np.exp(self.x)/np.sum(np.exp(self.x))
class sigmoid(f):
	def func(self):
		return 1/(1+np.exp(-self.x))
	def dfunc(self):
		return (1-self.func())*self.func()

class Loss(object):
	def __init__(self,y,t):
		self.y = y
		self.t = t
	def SE(self):
		return ((self.t - self.y)**2)/2
	def dSE(self):
		return -(self.t-self.y)
	def CrossEntropy(self):
		C = -np.mean(np.sum((self.t*np.log(self.y)+(1-self.t)*np.log(1-self.y)),axis=1),axis=0)
		return C
	def dCrossEntropy(self):
		return np.mean(self.y-self.t,axis=0)
class nn(object):
	def __init__(self,shape):
		self.shape = shape
		self.num_layers = len(shape)-1
		self.layers = [0 for i in range(len(shape)-1)]
		for i in range(len(shape)-1):
			self.layers[i] = layer(shape[i],shape[i+1])
		self.eta = 0.1
		self.output = np.zeros(shape[-1])
	def forward(self,x):
		self.input = x
		for i in range(self.num_layers):
			x = self.layers[i].forward(x)
		self.output = x
		return x
	def backward(self,target):
		self.loss = Loss(self.output,target)
		dloss = self.loss.dSE()
		df = self.layers[-1].actfunc.dfunc()
		self.layers[-1].delta = dloss*df
		for i in reversed(range(self.num_layers-1)):
			self.layers[i].backward(np.dot(self.layers[i+1].delta,self.layers[i+1].W))
	def update(self):
		self.layers[0].W -= self.eta*np.outer(self.layers[0].delta,self.input)
		self.layers[0].b -= self.eta*self.layers[0].delta
		for i in range(1,self.num_layers):
			self.layers[i].W -= self.eta*np.outer(self.layers[i].delta,self.layers[i-1].output)
			self.layers[i].b -= self.eta*self.layers[i].delta
	def learn(self,data,batch_size=10):
		x,t = data
		self.forward(x)
		self.backward(t)
		self.update()
	def save(self,filename):
		import shelve
		data = shelve.open(filename)
		data['layers'] = self.layers
		data['shape'] = self.shape
		data['eta'] = self.eta
		data.close()
	def load(self,filename):
		import shelve
		data = shelve.open(filename)
		self.layers = data['layers']
		self.shape = data['shape']
		self.num_layers = len(self.layers)
		self.eta = data['eta']
		data.close()