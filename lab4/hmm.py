import numpy as np

class HMM(object):

	 """This class defines an Hidden Markov Model.
	 An HMM is defined as a tuple (X, Z, P, O) where:
	 X - is the set of possible states.
	 Z - the set of possible observation.
	 P - is the transition probability matrix.
	 O - is the obsertation/emission probability matrix. """

	def __init__(self, X, Z, P, O, inicial_dist=None):
		"""Create a HMM. If no initial distribution is given it will assume that it is equal for each state."""
		self.X = np.array(X)
		self.Z = np.array(Z)
		self.P = np.array(P)
		self.O = np.array(O)

		if inicial_dist:
			self.inicial_dist = np.array(inicial_dist)
		else:
			self.inicial_dist = np.array([1/len(X)]*len(X))

	def emission_distribution(self, e):
		"""Maps an observation/emission string into the correct distribution for that emission. """
		for i in range(0, len(self.Z)):
			if e == self.Z[i]:
				return self.O[:,i]
		return None

	def viterbi(self, seq):
		""" Viterbi Algorithm. Returns a list of states index."""
		# initialization
		m = np.diag(self.emission_distribution(seq[0])).dot(self.inicial_dist)
		I = [[]]*(len(seq)-1)

		# cycle over the sequence
		for i in range(1, len(seq)):
			I[i-1] = np.argmax(self.P.T.dot(np.diag(m)), axis=1)
			m = np.diag(self.emission_distribution(seq[i])).dot(np.amax(self.P.T.dot(np.diag(m)), axis=1))

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
		"""" Forward algorithm. Returns a vector with the probability of each state given an observed sequence. """
		a = np.diag(self.emission_distribution(seq[0])).dot(self.inicial_dist)
		for i in range(1, len(seq)):
			a = (np.diag(self.emission_distribution(seq[i])).dot(self.P.T)).dot(a)
		return a

	def norm(self, vec):
		""" Auxiliar function to normalize a vector. Usefull por exemple to normalize the forward result."""
		return vec/np.sum(vec)

""" main module function used to solve the LAB. """
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

		

