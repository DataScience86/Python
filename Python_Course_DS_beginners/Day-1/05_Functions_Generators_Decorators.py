# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:11:18 2018

@author: Mohit Sharma
@title: Defining Function
"""

def function_name(argument):
    print(argument)
    
    
# Defining a squaring function

def square(n):
    print(n**2)

square()
square( n = 2)

# lets check what happens if I do not provide a value of n
square()

# We can also set a default value while defining the function.

def square(n = 1):
    print(n*n)
    
square(3)


#%%
# Question1

# Write a function to check if a function is prime function or not
# Fill in the below blanks to complete the function

def check_prime(n):
    if n <= 1:
        return ______
    for i in range(2, n):
        if ___ % ____ == ____:
            return ______
    else:
        return ______

# Question2

# Write a function to check if the sum of the number is greater
# than 15 or not.

# Question3

# Write a Program to display the Fibonacci sequence of n-1th term.
# where n is provided by the user.

#%%
# Lets discuss in more details how this function thing works

def main():
    myname()

def myname():
    print('Mohit Sharma')
    
if __name__ == '__main__' : 
    myname()

# Every function returns an output. By default it is none

def main():
    name = myname()
    print(name)

def myname():
    print('Mohit Sharma')
    
if __name__ == '__main__' : main()

# How to correct that
def main():
    name = myname()
    print(name)

def myname():
    return' Mohit Sharma'
    
if __name__ == '__main__' : main()

#%%
# Things worth noting about default and non-default arguments
# Non default arguments should come before default arguments

def main():
    addition(2, 5)
    
def addition(a, b):
    print(a + b)
    
if __name__ == '__main__' : main()


# if not all the arguments are given - throughs an error
def main():
    addition(2)
    
def addition(a, b):
    print(a + b)
    
if __name__ == '__main__' : main()


# Givin non default argumnets after the default one - Throughs an error
def main():
    addition(2)
    
def addition(a = 1, b):
    print(a + b)
    
if __name__ == '__main__' : main()

# Something very interesting Call by value
# Givin non default argumnets after the default one - Throughs an error
def main():
    x = 23
    addition(a = 2, b = x)
    print(f'in main function value of x is {x}')
    
    
def addition(a = 1, b = 1):
    a = 11
    print(a + b)
    print(a)
    
if __name__ == '__main__' : main()

"""
this is what they call in the Python documentation call 
by value, and in most call-by-value languages, when you 
pass a variable to a function, the function operates on 
a copy of the variable, so the value is passed, but not 
the object itself, and this is an important distinction 
because if the object were actually passed, then if I 
were to change it here, it would also change there in 
main, and you notice that it doesn't act like that in Python.

This gets even more complicated as the above is only applicable to
immutable objects and not mutable ones. 

"""
# An example

def main():
    x = [23]
    y = x
    y[0] = 3
    print(id(x))
    print(id(y))
    
    #addition(2, b = x)
    #print(f'in main function value of x is {x}')
    
    
def addition(a, b):
    print(id(a))
    a = 11
    print(id(a))
    print(a + b)
    print(a)
    
if __name__ == '__main__' : main()

"""
What I've done here is I've created a list, 
I've assigned it to x, I've assigned that x = y, 
and I changed the element of the list only in y. I 
did not change it in x, and the value of both of them 
was changed. So when you assign a mutable, you're actually 
assigning a reference to the mutable, and I have the side effect 
that when I change an element of that list in one place, it gets 
changed in both places because it's really just one object, and 
functions work exactly the same way.

So if I run this, you notice that we have this list, 
and the list is being printed in kitten as the list with 
five, and in main as the list with five. It's a one-element 
list, but if down here in kitten I change a sub zero equals three, 
it's changed in both places, so this is not call by value at all. 
This is actually, strictly, call by reference. What's being 
passed is a reference to the object, and you can change 
the object in the caller from the function.

So this is important to understand: an integer is not mutable, 
so it cannot change, so when you assign a new value to an integer, 
you're actually assigning an entirely different object to the name. 
The original integer is not changed, the name simply refers to a new 
object.

"""

def sum(*a
#%% 
# Python functions allow variable length, argumnet lits
# Argument list
def main():
    
    name_list(1,2, 3)
    
def name_list(*args):
    if len(args):
        for i in args:
            print(i)
    else:
        print('That Is All')
        
if __name__ == '__main__' : 
    main()

# lets see how we can use it

#Python functions allow variable length, argumnet lits
def main():
    num_list =[]
    n = int(input("Tell How mnay numers : "))
    for i in range(n):
        num = int(input(f'Enter {i} number: '))
        num_list.append(num)
    
    addition(num_list)
    
def addition(*args):
    j = 0
    if len(args):
        for i in args:
            j = j + i
            print(j)
    else:
        print('That Is All')
        
if __name__ == '__main__' : main()

# Calling the above by using an variable

def main():
    x = [1,2,3]
    addition(*x)
    
def addition(*args):
    j = 0
    if len(args):
        for i in args:
            j = j + i
            print(j)
    else:
        print('That Is All')
        
if __name__ == '__main__' : main()


# like above we can pass dictionaries as well unlike tuples in above 
# Example

def main():
    key_value(key1 = '1', key2 = '2', key3 = '3')
    
def key_value(**kwargs):
    
    if len(kwargs):
        for i in kwargs:
            print('value {}, is {}'.format(i, kwargs[i]))
    else:
        print('That Is All')
        
if __name__ == '__main__' : main()

# We can also pass the dictionary in the similar fasion as we did 
# for tuples above ,in the form of a variable


def main():
    dict1 = dict(key1 = '1', key2 = '2', key3 = '3')
    key_value(**dict1)
    
def key_value(**kwargs):
    
    if len(kwargs):
        for i in kwargs:
            print('value {}, is {}'.format(i, kwargs[i]))
    else:
        print('That Is All')
        
if __name__ == '__main__' : main()

#%%

# Returnig a value - Also rememer there is no difference between
# a function and procedure in python

def main():
    final = addition(2, 3)
    print(type(final))
    
def addition(a, b):
    print(a + b)
    
if __name__ == '__main__' : main()

""" 
If we do not return a value from a function, it returns none.
So it is important thta you return a value from a function.

"""

def main():
    final = addition(2, 3)
    print(type(final))
    
def addition(a, b):
    print(a + b)
    return a + b
    
if __name__ == '__main__' : main()

# One can return anything, either dictionary, list or tuple.

#%%

"""
Generators !

A generator is a special class of function that serves as 
an iterator instead of returning a single value the generator 
returns a stream of values.

"""

def main():
    for i in inclusive_range(25): # Can pass upto 3 arguments
        print(i, end = ' ')
    print()

def inclusive_range(*args):
    numargs = len(args)
    start = 0
    step = 1
    
    # initialize parameters
    if numargs < 1:
        raise TypeError(f'expected at least 1 argument, got {numargs}')
    elif numargs == 1:
        stop = args[0]
    elif numargs == 2:
        (start, stop) = args
    elif numargs == 3:
        (start, stop, step) = args
    else: raise TypeError(f'expected at most 3 arguments, got {numargs}')

    # generator
    i = start
    while i <= stop:
        yield i
        i += step

if __name__ == '__main__': main()

#%%

"""
Decorator !

A decorator is a special function which works as a wrapper function.

"""

def make_pretty(func):
    def inner():
        print("I got decorated")
        func()
    return inner

def ordinary():
    print("I am ordinary")

# In the example shown above, make_pretty() is a decorator. 
# In the assignment step.
pretty = make_pretty(ordinary)
pretty()

# The function ordinary() got decorated and the returned 
# function was given the name pretty.

""" 

To do the above task IE of decorating the function. Python has a 
special syntax to simplify this.

"""

@make_pretty
def ordinary():
    print("I am ordinary")
    
# The above statement is equivalent to the below statement

def ordinary():
    print("I am ordinary")
ordinary = make_pretty(ordinary)


# Example of decorating functions with Parameters

def divide(a, b):
    return a/b

divide(21, 3)


# However if I divide 21 by 0 we get an error
# lets check for this by using a decorator

def smart_divide(func):
   def inner(a,b):
      print("You are going to divide",a,"by",b)
      if b == 0:
         print("Sorry! cannot divide")
         return

      return func(a,b)
   return inner

@smart_divide
def divide(a,b):
    return a/b

result = smart_divide(divide)
result(2, 2)
