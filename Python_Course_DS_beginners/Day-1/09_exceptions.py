#!/usr/bin/env python3
# what do you think will happen
def main():
    x = int("foo")
        
if __name__ == '__main__':main()

#%%
def main():
    try:
        x = int("datasciencebeginners.com")
    except ValueError:
        print("We caught a value error")
    
if __name__ == '__main__':main()
#%%

def main():
    try:
        x = 3/0
    except Z:
        print("We caught a calculation where division is happened by zero")
    
if __name__ == '__main__':main()

#%%
# We can use multiple error exceptions as well
def main():
    try:
        x = 3/0
    except ValueError:
        print("We caught a value error")
    except Z:
        print("don\'t divide by zero")
    
if __name__ == '__main__':main()

#%%
# So if we do not have error then the script should execute right, lets
# see how we can make that happen




#%%
# It is also good idea to report the error

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

# Changing 25 to " " or passing more than 3 arguments will result into error
# Add try statement to handle the error
def main():
    for i in inclusive_range(2,3,4,5):
        print(i, end = ' ', flush = True)
    print()

if __name__ == '__main__': main()















#%%
def main():
    try:
        for i in inclusive_range(2,3,4,5):
            print(i, end = ' ', flush = True)
        print()
    except ValueError as e:
        print(f'range error: {e}')
        
if __name__ == '__main__': main()
