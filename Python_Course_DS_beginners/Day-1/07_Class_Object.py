# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 17:28:15 2018

@author: Mohit Sharma
@title: Class Objects

"""

class Employee:
    ID = '555444'
    years_exp = 7
    
    def employee_ID(self):
        print('My Employee ID is {}'.format(self.ID))
        
    def employee_experience(self):
        print('My Work Experience is {} years'.format(self.years_exp))

def main():
    mohit = Employee()
    mohit.employee_experience()
    mohit.employee_ID()


if __name__ == '__main__' :
    main()
    
#%%
# Question1 - Define a class to add two numbers

# Question2 - 