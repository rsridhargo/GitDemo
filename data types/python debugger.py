'''
1)debugger is used to find the bug in any line of code
2)code is executed line by line
3)after python version of 3.7 we use breakpoint() which is same as pdb.set_trace() in older versions

Commands for debugging :
c -> continue execution
q -> quit the debugger/execution
n -> step to next line within the same function
s -> step to next line in this function or a called function
'''

# python debugging using breakpoint()
'''
#breakpoint()    # in pycharm we can set breakpoint and debug,need not explicitly write breakpoint()
name=input('Enter the name')
age=(input('Enter the age'))
new_age=int(age)+10
print(new_age)
'''

# using pdb.set_trace()

import pdb

pdb.set_trace()
h=int(input('Enter the height'))
l=int(input('Enter the length'))
area=0.5*l*h
print(area)