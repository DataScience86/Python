# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:21:49 2018

@author: Mohit Sharma
@title: Solutions To Internal Quizes
"""

#%%
"""
Script - Defining_Functions 
Question1 - Write a function to check if a function is prime function or not

"""
def check_prime(n):
    if n <= 1:
        return False
    for i in range(2, n):
        if n % i == 0:
            return True
    else:
        return False
    
#%%
"""
Script - Defining_Functions 
Question3 - Write a Program to display the Fibonacci sequence of n-1th term.
where n is provided by the user.

"""

def fabonacci_series(nterms = 20):
    term1 = 0
    term2 = 1
    count = 0

    # uncomment to take input from the user
    #nterms = int(input("How many terms? "))
    
    # check if the number of terms is valid
    if nterms <= 0:
       print("Please enter a positive integer")
    elif nterms == 1:
       print("Fibonacci sequence upto",nterms,":")
       print(term1)
    else:
       print("Fibonacci sequence upto",nterms,":")
       while count < nterms:
           print(term1,end=' ', flush = True)
           nth_term = term1 + term2
           # update values
           term1 = term2
           term2 = nth_term
           count += 1

