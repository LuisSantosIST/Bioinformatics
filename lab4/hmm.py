import numpy as np

class HMM(object):

	def __init__(self, X, Z, P, O, inicial_dist=None):
		self.X = np.array(X)
		self.Z = np.array(Z)
		self.P = np.array(P)
		self.O = np.array(O)

		if inicial_dist:
			self.inicial_dist = np.array(inicial_dist)
		else:
			self.inicial_dist = np.array([1/len(X)]*len(X))

	def emisson_distribution(self, e):
		for i in range(0, len(self.Z)):
			if e == self.Z[i]:
				return self.O[:,i]
		return None

	def viterbi(self, seq):
		# initialization
		m = np.diag(self.emisson_distribution(seq[0])).dot(self.inicial_dist)
		I = [[]]*(len(seq)-1)

		# cycle over the sequence
		for i in range(1, len(seq)):
			I[i-1] = np.argmax(self.P.T.dot(np.diag(m)), axis=1)
			m = np.diag(self.emisson_distribution(seq[i])).dot(np.amax(self.P.T.dot(np.diag(m)), axis=1))

		# backtrack
		# current variable will be initialized as the most probable final state and it will backtrack until the first state.
		current = np.argmax(m) 
		states = [current,]
		for i in range(len(I)-1, -1, -1):
			current = I[i][current]
			states.append(current)

		states.reverse()
		return states

	def forward(self, seq):
		a = np.diag(self.emisson_distribution(seq[0])).dot(self.inicial_dist)
		for i in range(1, len(seq)):
			a = (np.diag(self.emisson_distribution(seq[i])).dot(self.P.T)).dot(a)
		return a

	def norm(self, vec):
		return vec/np.sum(vec)

def run():
	P = [[0.6,  0.4,  0.0],
	     [0.25, 0.5,  0.25],
	     [0.25, 0.25, 0.5]]

	O = [[0.4, 0.3, 0.0, 0.3], 
	     [0.1, 0.1, 0.4, 0.4], 
	     [0.4, 0.3, 0.3, 0.0]]

	X = ["S1", "S2", "S3"]
	Z = ['A', 'T', 'C' ,'G']

	obsevation = "CATGCGGGTTATAAC"

	# question 2.B 
	print ("\nX = CATGCGGGTTATAAC")
	print ("\nquestion 2.B - most probable sequence of states that generates sequence X")
	hmm = HMM(X, Z, P, O)
	viterbi = hmm.viterbi(obsevation)
	path = "Viterbi result path: "
	for i in range(0, len(viterbi)-1):
		path += X[viterbi[i]] + " - "
	path += X[viterbi[i+1]]
	print (path)

	# question 2.D
	print ("\nquestion 2.D - compute the probability P(X)")
	forward = hmm.forward(obsevation)
	prob_observation = np.sum(forward)/len(forward)
	print ("using forward P(X) = " + str(prob_observation))
	print ("forward result normalized:")
	print(hmm.norm(forward))

if __name__ == '__main__':
    run()

		

