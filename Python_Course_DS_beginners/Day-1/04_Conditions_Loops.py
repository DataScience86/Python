# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 13:17:04 2018

@author: Mohit Sharma

@title: Conditions and Loops
"""


temp_today = 28
temp_yesterday = 28

if temp_today > temp_yesterday:
    print('Today temperature is {} degree higher'.format(temp_today - temp_yesterday))
elif temp_today < temp_yesterday:
    print('Today temperature is {} degree lesser'.format(temp_yesterday - temp_today))
else:
    print('No difference in temperature.')


#%%

# There are two basic types of loops. one is for and the other is while.

count = 0

while count != 10:
    count = count + 1
    print(count)
#%%    
# Lets write a program to print only odd numbers
count = 0
while count != 20:
    count = count + 1
    if count % 2 !=0: #% results into reminder also know as mod operator
        print(count)
        
# Question1 :- Print only even number
# Question2 :- Print only prime number

#%%
password = "python"
pw = ''

while pw != password:
    pw = input("Please enter your password: ")

#%%
# lets understand the for loop which is very different than the while

student_names = ['Mohan', 'Rohan', 'Ajay', 'Ankur']

for i in student_names:
    print(i)
    
#%%
"""
There are few controls which you should know to write effective 
and efficient code. 

Following are the controls:
    1. continue
    2. Break
    3. else - is not common to other languages
    

"""
password = "python"
pw = ''

count = 0
max_attempt = 3
auth = False


while pw != password:
    count += 1
    if count > max_attempt: break
    pw = input(f"{count}:Please enter your password: ")
else:
    auth = True

print("Authorised" if auth else "Calling police...." )