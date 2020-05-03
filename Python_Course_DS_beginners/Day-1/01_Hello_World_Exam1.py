# -*- coding: utf-8 -*- 
"""
Created on Tue Mar 27 11:40:54 2018
@author: Mohit Sharma
"""
print("Hello, World!")
print('You can print almost anything')



# In Python version 2 print was a statement. In [python 3] its a function.
y = 24
print('The number is %d'%y) # Legacy code

#%%

#!/user/bin/env python3
""" 
You might see the above command linein some codes. This is called as 
sebang line. It is a comman pattern for the Unix based systems. This 
command helps to invoke the code from a command line while working on 
the linux based systems.Next to #!(shebang) is the path to the executable
that will run the script.
""" 

# To check your python version
import platform

print(platform.python_version())

# You can also include a statement before the output

print('Python Version: {} and {}'.format(platform.python_version(), "star"))

#%%
# Another way of writing this is given below

import platform

def version():
    versioncall()
    
def versioncall():
    print('Python Version: {}'.format(platform.python_version()))
    
if __name__ == '__main__':version()

"""
This is actually a very common pattern in Python. 
By having this conditional statement at the bottom 
that calls main, it actually forces the interpreter 
to read the entire script before it executes any of the 
code. This allows a more procedural style of programming, 
and it's because Python requires that a function is defined 
before it's called.

"""

#%%
# Expression and Statements
# Significance of whitespaces
