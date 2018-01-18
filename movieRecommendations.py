#!/usr/bin/env python
"""
movie_recommendations.py
Original author: Felix Sun (6.008 TA, Fall 2015)
Modified by:
- Danielle Pace (6.008 TA, Fall 2016),
- George H. Chen (6.008/6.008.1x instructor, Fall 2016)

Please read the project instructions beforehand! Your code should go in the
blocks denoted by "YOUR CODE GOES HERE" -- you should not need to modify any
other code!
"""

import matplotlib.pyplot as plt
import movie_data_helper
import numpy as np
import scipy
import scipy.misc
from sys import exit


def compute_posterior(prior, likelihood, y):
    """
    Use Bayes' rule for random variables to compute the posterior distribution
    of a hidden variable X, given N i.i.d. observations Y_0, Y_1, ..., Y_{N-1}.

    Hidden random variable X is assumed to take on a value in {0, 1, ..., M-1}.

    Each random variable Y_i takes on a value in {0, 1, ..., K-1}.

    Inputs
    ------
    - prior: a length M vector stored as a 1D NumPy array; prior[m] gives the
        (unconditional) probability that X = m
    - likelihood: a K row by M column matrix stored as a 2D NumPy array;
        likelihood[k, m] gives the probability that Y = k given X = m
    - y: a length-N vector stored as a 1D NumPy array; y[n] gives the observed
        value for random variable Y_n

    Output
    ------
    - posterior: a length M vector stored as a 1D NumPy array: posterior[m]
        gives the probability that X = m given
        Y_0 = y_0, ..., Y_{N-1} = y_{n-1}
    """

    # -------------------------------------------------------------------------
    # ERROR CHECKS -- DO NOT MODIFY
    #

    # check that prior probabilities sum to 1
    if np.abs(1 - np.sum(prior)) > 1e-06:
        exit('In compute_posterior: The prior probabilities need to sum to 1')

    # check that likelihood is specified as a 2D array
    if len(likelihood.shape) != 2:
        exit('In compute_posterior: The likelihood needs to be specified as ' +
             'a 2D array')

    K, M = likelihood.shape

    # make sure likelihood and prior agree on number of hidden states
    if len(prior) != M:
        exit('In compute_posterior: Mismatch in number of hidden states ' +
             'according to the prior and the likelihood.')

    # make sure the conditional distribution given each hidden state value sums
    # to 1
    for m in range(M):
        if np.abs(1 - np.sum(likelihood[:, m])) > 1e-06:
            exit('In compute_posterior: P(Y | X = %d) does not sum to 1' % m)

    #
    # END OF ERROR CHECKS
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE FOR PART (b)
    #
    # Place your code to compute the log of the posterior here: store it in a
    # NumPy array called `log_answer`. If you exponentiate really small
    # numbers, the result is likely to underflow (i.e., it will be so small
    # that the computer will just make it 0 rather than storing the right
    # value). You need to go to log-domain. Hint: this next line is a good
    # first step.
    
    #print ('Debut COMPUTE_POSTERIOR')
    
    log_prior = np.log(prior)
    
    #utilisation de y
    mydict = {}
    num = 0
    den = 0
    #print ('visualisation de y ', y)
    for i in y:
        if i not in mydict:
            mydict[i]=1
        else:
            mydict[i]+=1
    for i in range(M):
        if i not in mydict:
            mydict[i]=0
    mylist = []
    for i in mydict.keys():
        mylist.append(mydict[i])
    myarray = np.array(mylist)
    #print ('myarray = combien de fois on a chaque rating ', myarray)
    
    #transformation de la likelihood par y
    n = np.shape(myarray)[0]
    #print ('n ', n)
    #print ('likelihood initiale ', likelihood)
    likelihood_time_myarray = np.zeros((n,n))
    for i in range(n):
        for j in range (n):
            likelihood_time_myarray[i][j] += np.log(likelihood[i][j])*myarray[i]
            #print ('likelihood[i][j] ', likelihood[i][j])
            #print ('my array [i] ', myarray[i])
            #print ('mise à la puissance ', np.exp(np.log(likelihood[i][j])*myarray[i]))
    #print ('likelihood_time_myarray ', likelihood_time_myarray)
    
    
    
    #log_likelihood = np.log(likelihood_time_myarray)
    #print ('log_likelihood ', log_likelihood)
    
    num += log_prior + likelihood_time_myarray.sum(axis=0)
    #print ('num ', num)
    
    den = scipy.misc.logsumexp(likelihood_time_myarray.sum(axis=0) + log_prior)
    #print ('den ', den)        
    
    log_answer= num - den
    #print ('log_answer ', log_answer)
    
    #
    # END OF YOUR CODE FOR PART (b)
    # -------------------------------------------------------------------------

    # do not exponentiate before this step
    posterior = np.exp(log_answer)
    #print ('sum_posterior ' , posterior.sum(axis=0))
    
    #print ('FIN COMPUTE_POSTERIOR')
    return posterior

#print ('test compute_posterior ', compute_posterior(np.array([ 0.5, 0.5]), np.array([[ 0.45, 0.6 ], [ 0.55, 0.4 ]]), np.array([0, 0, 1, 1, 1, 0, 1, 1])), 'expected value ', '[0.6746346188187191, 0.3253653811812809]')
#print ('test compute_posterior ', compute_posterior(np.array([ 0.5, 0.5]), np.array([[ 0.45, 0.6 ], [ 0.55, 0.4 ]]), np.array([0, 0, 0, 1,0,0,1,1,1,0,1,1,1,1,1,1, 1, 1, 1])), 'expected value ', '[0.910873103747323, 0.0891268962527537]')


def compute_movie_rating_likelihood(M):
    """
    Compute the rating likelihood probability distribution of Y given X where
    Y is an individual rating (takes on a value in {0, 1, ..., M-1}), and X
    is the hidden true/inherent rating of a movie (also takes on a value in
    {0, 1, ..., M-1}).

    Please refer to the instructions of the project to see what the
    likelihood for ratings should be.

    Output
    ------
    - likelihood: an M row by M column matrix stored as a 2D NumPy array;
        likelihood[k, m] gives the probability that Y = k given X = m
    """

    # define the size to begin with
    
    
    likelihood = np.zeros((M, M))
    
    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE FOR PART (c)
    #
    # Remember to normalize the likelihood, so that each column is a
    # probability distribution.
    #
    #print ('DEBUT COMPUTE_LIKELIHOOD')
    #print ('likelihood ', likelihood)
    
    X=np.arange(0,M)
    #print ('X ', X)
    
    Y=np.arange(0,M)
    #print ('Y ', Y)    
    
    for i in range (0,M):
        for j in range (0,M):
            if i == j:
                likelihood[i][j]=2
            elif i != j:
                likelihood[i][j]=1/np.abs(Y[i]-X[j])
    
    #print ('before normalization',likelihood)
    likelihood=likelihood/likelihood.sum(axis=0)
    
    
    #print ('FIN COMPUTE_LIKELIHOOD')
    #
    # END OF YOUR CODE FOR PART (c)
    # -------------------------------------------------------------------------

    return likelihood

#pouet = compute_movie_rating_likelihood(11)
#print ('test compute_movie_rating_likelihood(M) ', pouet  )
#test normalisation
#print ('test normalisation de la likelihood ', pouet.sum(axis=0))

def infer_true_movie_ratings(num_observations=-1):
    """
    For every movie, computes the posterior distribution and MAP estimate of
    the movie's true/inherent rating given the movie's observed ratings.

    Input
    -----
    - num_observations: integer that specifies how many available ratings to
        use per movie (the default value of -1 indicates that all available
        ratings will be used).

    Output
    ------
    - posteriors: a 2D array consisting of the posterior distributions where
        the number of rows is the number of movies, and the number of columns
        is M, i.e., the number of possible ratings (remember ratings are
        0, 1, ..., M-1); posteriors[i] gives a length M vector that is the
        posterior distribution of the true/inherent rating of the i-th movie
        given ratings for the i-th movie (where for each movie, the number of
        observations used is precisely what is specified by the input variable
        `num_observations`)
    - MAP_ratings: a 1D array with length given by the number of movies;
        MAP_ratings[i] gives the true/inherent rating with the highest
        posterior probability in the distribution `posteriors[i]`
    """
    
    #print ('DEBUT INFER TRUE MOVIE RATING')
    
    M = 11  # all of our ratings are between 0 and 10
    prior = np.array([1.0 / M] * M)  # uniform distribution
    likelihood = compute_movie_rating_likelihood(M)
    
    #print ('tronche de la likelihood', likelihood)

    # get the list of all movie IDs to process
    movie_id_list = movie_data_helper.get_movie_id_list()
    num_movies = len(movie_id_list)

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE FOR PART (d)
    #
    # Your code should iterate through the movies. For each movie, your code
    # should:
    #   1. Get all the observed ratings for the movie. You can artificially
    #      limit the number of available ratings used by truncating the ratings
    #      vector according to num_observations.
    #   2. Use the ratings you retrieved and the function compute_posterior to
    #      obtain the posterior of the true/inherent rating of the movie
    #      given the observed ratings
    #   3. Find the rating for each movie that maximizes the posterior

    # These are the output variables - it's your job to fill them.
    posteriors = np.zeros((num_movies, M))
    #print ('initialisation posteriors ', posteriors)
    MAP_ratings = np.zeros(num_movies)
    #print ('initialisation MAP_ratings ', MAP_ratings)

    for i in range(num_movies):
        mylist=[]
        if num_observations == -1:
            ratings_movie_i = movie_data_helper.get_ratings(i)
            posteriors[i] += compute_posterior(prior, likelihood, ratings_movie_i)
        else : 
            for j in range(num_observations):
                mylist.append(movie_data_helper.get_ratings(i)[j])
            ratings_movie_i = np.array(mylist)
            posteriors[i] += compute_posterior(prior, likelihood, ratings_movie_i)
        MAP_ratings[i] += np.argmax(posteriors[i],axis=0)
    
    #print ('FIN INFER TRUE MOVIE RATING')
    #
    # END OF YOUR CODE FOR PART (d)
    # -------------------------------------------------------------------------

    return posteriors, MAP_ratings


#print ('test INFERTRUE MOVIE RATING ', infer_true_movie_ratings(10))
#print ('MEGA TEST ' , movie_data_helper.get_ratings(10))

def compute_entropy(distribution):
    """
    Given a distribution, computes the Shannon entropy of the distribution in
    bits.

    Input
    -----
    - distribution: a 1D array of probabilities that sum to 1

    Output:
    - entropy: the Shannon entropy of the input distribution in bits
    """

    # -------------------------------------------------------------------------
    # ERROR CHECK -- DO NOT MODIFY
    #
    if np.abs(1 - np.sum(distribution)) > 1e-6:
        exit('In compute_entropy: distribution should sum to 1.')
    #
    # END OF ERROR CHECK
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE FOR PART (f)
    #
    # Be sure to:
    # - use log base 2
    # - enforce 0log0 = 0
    entropy = 0
    for i in range(np.shape(distribution)[0]):
        if distribution[i] < 10e-8:
            entropy += 0
        else:
            entropy += distribution[i] * np.log(1 / distribution[i]) / np.log(2)
    
    #
    # END OF YOUR CODE FOR PART (f)
    # -------------------------------------------------------------------------

    return entropy


def compute_true_movie_rating_posterior_entropies(num_observations):
    """
    For every movie, computes the Shannon entropy (in bits) of the posterior
    distribution of the true/inherent rating of the movie given observed
    ratings.

    Input
    -----
    - num_observations: integer that specifies how many available ratings to
        use per movie (the default value of -1 indicates that all available
        ratings will be used)

    Output
    ------
    - posterior_entropies: a 1D array; posterior_entropies[i] gives the Shannon
        entropy (in bits) of the posterior distribution of the true/inherent
        rating of the i-th movie given observed ratings (with number of
        observed ratings given by the input `num_observations`)
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE FOR PART (g)
    #
    # Make use of the compute_entropy function you coded in part (f).
    
    movie_id_list = movie_data_helper.get_movie_id_list()
    num_movies = len(movie_id_list)
    posterior_entropies = np.zeros((num_movies))
    calculation = infer_true_movie_ratings(num_observations)
    
    for i in range(num_movies):
        posterior_entropies[i] += compute_entropy(calculation[0][i])
    
    #
    # END OF YOUR CODE FOR PART (g)
    # -------------------------------------------------------------------------

    return posterior_entropies

#print ('TEST compute_true_movie_rating_posterior_entropies(num_observations) ' ,compute_true_movie_rating_posterior_entropies(10))

def main():

    # -------------------------------------------------------------------------
    # ERROR CHECKS
    #
    # Here are some error checks that you can use to test your code.

    print("Posterior calculation (few observations)")
    prior = np.array([0.6, 0.4])
    likelihood = np.array([
        [0.7, 0.98],
        [0.3, 0.02],
    ])
    y = [0]*2 + [1]*1
    print("My answer:")
    print(compute_posterior(prior, likelihood, y))
    print("Expected answer:")
    print(np.array([[0.91986917, 0.08013083]]))

    print("---")
    print("Entropy of fair coin flip")
    distribution = np.array([0.5, 0.5])
    print("My answer:")
    print(compute_entropy(distribution))
    print("Expected answer:")
    print(1.0)

    print("Entropy of coin flip where P(heads) = 0.25 and P(tails) = 0.75")
    distribution = np.array([0.25, 0.75])
    print("My answer:")
    print(compute_entropy(distribution))
    print("Expected answer:")
    print(0.811278124459)

    print("Entropy of coin flip where P(heads) = 0.75 and P(tails) = 0.25")
    distribution = np.array([0.75, 0.25])
    print("My answer:")
    print(compute_entropy(distribution))
    print("Expected answer:")
    print(0.811278124459)

    #
    # END OF ERROR CHECKS
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE FOR TESTING THE FUNCTIONS YOU HAVE WRITTEN,
    # for example, to answer the questions in part (e) and part (h)
    #
    # Place your code that calls the relevant functions here.  Make sure it's
    # easy for us graders to run your code. You may want to define multiple
    # functions for each of the parts of this problem, and call them here.

    #
    # END OF YOUR CODE FOR TESTING
    # -------------------------------------------------------------------------


if __name__ == '__main__':
    main()
