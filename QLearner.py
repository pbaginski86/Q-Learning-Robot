"""  		   	  			    		  		  		    	 		 		   		 		  
Template for implementing QLearner  (c) 2015 Tucker Balch  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		   	  			    		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		   	  			    		  		  		    	 		 		   		 		  
All Rights Reserved  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		   	  			    		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		   	  			    		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		   	  			    		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		   	  			    		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		   	  			    		  		  		    	 		 		   		 		  
or edited.  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		   	  			    		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		   	  			    		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		   	  			    		  		  		    	 		 		   		 		  
GT honor code violation.  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
-----do not edit anything above this line---
Student Name:   (replace with your name)
GT User ID: 
GT ID:  (replace with your GT ID)
"""

import numpy as np
import random as rand


class QLearner(object):

    def __init__(self, num_states=100, num_actions=4, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99,
                 dyna=0, verbose=False):
        # For starters, I am just going to initialize everything that I find in the init function
        self.verbose = verbose  # Do we print or do we not? I'm leaving the print statements that
        # were already in here
        self.num_states = num_states  # As per instructions and init function - check later if right
        self.num_actions = num_actions  # as per the above
        self.alpha = alpha  # As per instructions and init function - check later if right
        self.gamma = gamma  # As per instructions and init function - check later if right
        self.rar = rar  # As per instructions and init function - check later if right
        self.radr = radr  # As per instructions and init function - check later if right
        self.dyna = dyna  # As per instructions and init function - check later if right
        self.state = 0  # As per youtube video
        self.robot_move = 0  # As per youtube video
        self.Q_table = np.zeros(shape=(num_states, num_actions))  # Lecture specifies to initialize Q
        # table as zeros in the shape of actions and states, or a states X actions matrix
        self.transition_count = np.full(shape=(num_states, num_actions, num_states), fill_value=0.000001)
        # from lecture 03 - 07 - 3 (s,a,s') "introducing a new table t count or t c"
        self.transition = np.zeros(shape=(num_states, num_actions, num_states))  # from lecture 03 - 07 - 7
        self.reward = np.zeros(shape=(num_states, num_actions))  # from lecture 03 - 07 - 7 and 03 - 07 - 5]
        self.num_a = self.num_actions - 1

    def author(self):
        return ""

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        # As per the youtube lecture, I am first checking if I need to pick a random action
        self.state = s  # I wasn't sure if this should be here or lower - it doesn't seem to differ
        if rand.random() <= self.rar:  # From lecture video on youtube at 38min22secs
            robot_move = int(self.num_a * rand.random())  # matches the zip file content
        else:  # This is finding the maximum value of Q as per the udacity lectures
            robot_move = np.argmax(self.Q_table[s, :])  # as per lecture video questions
        self.robot_move = robot_move  # setting a to be the action again
        return robot_move  # Return the current action.

    def query(self, s_prime, r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        self.Q_table[self.state, self.robot_move] = float((float(1) - self.alpha)) * \
            self.Q_table[self.state, self.robot_move] + \
            self.alpha * (r + self.gamma *
                        self.Q_table[s_prime, np.argmax(
                                self.Q_table[s_prime, :])])
        # This is as per the udacity lecture video - exact copy of update rule
        if rand.random() <= self.rar:  # as per youtube video, chosing random action
            robot_move = int(self.num_a * rand.random())  # as per youtube video chosing random
        else:
            robot_move = np.argmax(self.Q_table[s_prime, :])  # if not random, then chose action as per update rule

        if self.dyna is not 0:
            # Implementing as per udacity lecture 3-07-5 Learning R 1min35sec
            # First step is to update T as per lecture
            # Second step is to update R as per lecture
            # Third step is to create a loop that iterates over dyna
            # we are chosing s to be random and a to be random in the loop first
            # s prime is to be inferred from T
            # we update r as a query of R at s,a
            # This is where we are constructing the model, first part of DYNA
            # The first one is the update of R as per the Learning R lecture video
            self.reward[self.state, self.robot_move] = float(float(1) - self.alpha) * \
                                                  self.reward[self.state, self.robot_move] + (self.alpha * r)
            # The second one is the update of the transition counting table
            self.transition_count[self.state, self.robot_move, s_prime] = self.transition_count[self.state,
                                                                                        self.robot_move, s_prime] + 1
            # The last one is the update of the transition matrix as per the Quiz in Udacity
            self.transition = self.transition_count / self.transition_count.sum(axis=2, keepdims=True)  # Not sure here
            # Here we are doing the artificial construction of states/Q as described in the video
            for i in range(self.dyna):  # This is the iterator for self dyna, it's what prof calls hallucinate
                s_v2 = int(self.num_states * rand.random())  # Initialize randomly, this was in file
                a_v2 = int(self.num_actions * rand.random())  # Initialize randomly, this was in file
                # as https://eli.thegreenplace.net/2018/slow-and-fast-methods-for-generating-random-integers-in-python/
                # Comment in piazza post reveals: np.argmax(np.random.multinomial(...arguments...))
                # Implementing this as per the piazza post @1824 comment by John Austin Griffith
                s_prime_v2 = np.argmax(np.random.multinomial(1, self.transition[s_v2, a_v2, :]))
                r = self.reward[s_v2, a_v2]  # As per the lecture
                # In this last part, we are updating Q
                self.Q_table[s_v2, a_v2] = float((float(1) - self.alpha)) * \
                                                            self.Q_table[s_v2, a_v2] + \
                                                            self.alpha * (r + self.gamma *
                                                                          self.Q_table[s_prime_v2, np.argmax(
                                                                              self.Q_table[s_prime_v2, :])])

        self.state = s_prime
        self.robot_move = robot_move
        self.rar = self.rar * self.radr  # As per the assignment instructions on quantsoftware

        return robot_move
