# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 17:51:05 2018

@author: Mohit Sharma
@title: Python datatypes

"""

var = None
print('variable value is: {} and its type is {}'.format(var, type(var)))


#%%
# Little bit more about format()

num1 = 10
num2 = 3

print('First Number is: {} and Second Number is {}'.format(num1, num2))

print('First Number is: {1} and Second Number is {0}'.format(num1, num2))

print('First Number is: {1:>04} and Second Number is {0:<04}'.format(num1, num2))

#%%
"""
Python 3 has two basic numeric data types integer and floating point. You may
hear about other data types derived from these such as complex type. In fact,
one can create his own data type as well. But for now lets focus on
integer and floating types.

"""

print(num1/num2)

# Next statement will give you only integer output.
print(num1//num2)

# To get the remainder use module operator '%'
print(num1 % num2)

# Here is Quiz 
# What do you think the output of the below would be

print(.1 + .1 +.1 - .3)
"""
This results into difference between the accuracy and precision. Here
they are sacrifising accuracy for precision. That is why you should not use
floating point for money calculations.Solution to this problem is
to use the decimal module.

"""

import decimal
from decimal import * # * represents everything.

a = Decimal('.10')
b = Decimal('.30')

result = (3*a) - b
print(result)

#%%
# Type Boolean
compare = 14 > 12
print(type(compare))

if compare:
    print(True)
else:
    print(False)

# Note 1 represents True, 0 represents False

#%%
# Type Sequesnce - We have list, tuples and dictionaries

# Let us create a list
list1  = [2, 4, 6, 8, 10]

# To access elements of a list one can make use of indexes
list1[0] # first element of the list

list1[-2] # last element of the list

list1[2] = 66 # Change the elemet of the list by reassissing it

# By making use of same indexing we can iterate all elements using for loop
for i in list1:
    print(i)
    

# Let me define a list of names of people sitting in this class
def main():
    names = ['Mohit', 'Roshan', 'Saurabh']
    #print(names[0:3:2])
    #To get the index of the item
    index = names.index('Mohit')
    print(index)
    # Appened and item
    names.append('Shiva')
    # Insert an item at a particular index
    names.insert(0, 'Shiva')
    # To Remove an item
    names.remove('Shiva')
    #To remove the item from the end of the list
    names.pop()
    # You can also collect the removed item using pop
    collect_removed = names.pop(-1)
    # using join with lists
    print(', '. join(names))
    #print_list(names)

def print_list(li):
    for i in li: print(i, end = ' ', flush = True)
    print()
    
if __name__ == '__main__': main()


#%%
# Tuple Sequence 
tuple1 = (2, 4, 6, 8, 10)
type(tuple1)

tuple1[0]

# But they are immutable that is one can not change the values
# Using Range to create a sequence

seq1 = range(5)
for i in seq1:
    print(i)

# One can also decide in details about the sequence
seq1 = range(5, 50, 2)

for i in seq1:
    print(i)

# Range is immutable

seq1[2] = 44

# So to make it mutable I need to make it a list

seq2 = list(range(5))
seq2[0] = 44

for i in seq2:
    print(i)

# Another example of the above illustrated in the form of a OOP

def main():
    names = ('Mohit', 'Roshan', 'Saurabh')
    # Appened and item
    # names.append('Nikhil')
    # Insert an item at a particular index
    # names.insert(0, 'Rajesh')
    # To Remove an item
    # names.remove('Roshan')
    print_list(names)

def print_list(li):
    for i in li: print(i, end = ' ', flush = True)
    print()
    
if __name__ == '__main__': main()

# You should always prefer to return tuple as a return value of a function
# Any guesses why ?
#%%

# Dictionary Sequence
# Creating a dictionary

dict1 = { 'AA' : 10, 'BB' : 20, 'CC' : 30, 'DD' : 40, 'EE' : 50 }

for i in dict1:
    print(i)
    
# did you notice it only prints the keys and not the values
# One must use .item() method to iterate over both keys and values

for key, value in dict1.items():
    print("key = {}, value = {}".format(key, value))
    
# To change the value, refer to the key and change the value
dict1["AA"] = 22

# Defining a dictionary having a key value pair for roll no. and students

def main():
    students = {'112': 'Rohit', '113': 'Rajesh', '114':'Ganesh', 
                '115' : 'Ahmed', '116' : 'Paresh'}
    print_details(students)
    
def print_details(dict):
    for x in dict: print(f'{x}:{dict[x]}')
    
if __name__ == '__main__': main()


# One can aso define dictionary using dict constructor and key work aruments.
students = dict(name1 ='Rohit', name2 = 'Rajesh', name3 = 'Ganesh', 
                    name4 = 'Ahmed', name5 = 'Paresh')

# now we shall loop over this using key value pairs

for key, value in students.items():
    print(f'{key}:{value}')

# To view only the keys

for key in students.values():
    print(f'{key}')
    
# how do we print value for the key name3?

# Add another item name4 equal to 'Gaga'

# One can use conditional statments on the dictionary keys

print('Present!' if 'name2' in students else 'Absent')

# If you call a key which does not exist then you get an exceptions error

print(students['name33'])

# To avoid the error one should use get()
print(students.get('name33'))
    
#%%

""" 
Remember everything in python is an object. So type is similar to the 
class. Lets see what I mean.

"""

tuple2 = (1,2,3,4)
print(type(tuple2))

# Lets modify the above tuple   

tuple2 = (1,2,'three', [22,33, 'twice'])
tuple3 = (10,2,'three', [23,12, 'thrice'])
print(type(tuple2[1]))
print(type(tuple2[2]))

# Lets see what it returns if I say id() instead of tuple
print(id(tuple2[2]))
print(id(tuple2[2]))

# The numbers are same because there is only one identifier for same number
tuple2[2] is tuple2[2]

# You can see is can be used to compare the items, elements or objects.

#%%
# Checking te type of the object in python
# Use isinstance()

if isinstance(tuple2, tuple):
    print("Yes")
else:
    print('Nope')
    
list2 = [2,3,4,5]
if isinstance(tuple2, list):
    print("Yes")
else:
    print('Nope')

#%%
# A bit about conditional assignment

Are_You_Hungry = True
x = 'feed him now!' if Are_You_Hungry else ' Do not feed'
print(x)

# The above is called as ternary conditional operation and requires both 
# if and else close. It does not work if one of them is missing.

#%%
""" 
Next we will dicsuss about
1. Arthmetic operators
2. Bitwise Operators
    example:
        1. & And
        2. | Or
        3. ^ Xor
        4. << Shift left
        5. >> Shift Right
3. Comparison Operators
4. Boolean Operators
    example:
        1. and
        2. or
        3. not
        4. in       value in set
        5. not in   value not in set
        6. is       Same object id
        7. is       not Not same object id
        
Please ensure that you go through the order of operator precendence

"""

# 2. Bitwise
x = 0x0a
y = 0x02
z = x & y

print(f'(hex) x is {x:02x}, y is {y:02x}, z is {z:02x}')
print(f'(bin) x is {x:08b}, y is {y:08b}, z is {z:08b}')

