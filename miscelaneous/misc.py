'''
multiple ways to write prime number code
1) checking divisibles upto n
2)checking divisibles upto n//2        ----optimized
3)checking divisibles upto int((sqrt(n)) ----further optimized
'''
from math import sqrt
def prime(n):
    #for i in range(2, n):
    #for i in range(2, n//2):
    for i in range(2, int(sqrt(n))):
        if n % i == 0:
            print('not prime')
            break
    else:
        print('prime')

prime(57)

'''
Why don't we have any break statements? 
It should be noted that as soon as a function returns something, it shuts down. 
A function can deliver multiple print statements, but it will only obey one return.
'''

'''
ANIMAL CRACKERS: Write a function takes a two-word string and 
returns True if both words begin with same letter
animal_crackers('Levelheaded Llama') --> True
animal_crackers('Crazy Kangaroo') --> False

'''

def animal_crackers(s):
    lst=s.lower().split()   # to give true even for small letter and splitting and saving in list
    return lst[0][0]==lst[1][0]   # lst[0][0] in string means first letter of string
animal_crackers('Levelheaded Llama')
animal_crackers('Crazy Kangaroo')
animal_crackers('Crazy cat')

'''
self keyword represents object e are taking about
1)inside class we use self to refer the object
2)outside class we use object name itself
'''
class Student:
    def __init__(self,name,age):
        self.age=age
        self.name=name

    def info(self):
        print('name of student is ',self.name)   # we have to use self inside class
        print('age of student is ',self.age)

    def display(self):   # self is used refer method
        self.info()

s1=Student('sri',25)
s2 = Student('muki', 35)
print(s1.name)
print(s2.age)
s2.info()
s1.display()

'''
join():

The join() method is a string method and returns a string 
in which the elements of sequence have been joined by str separator.

string_name.join(iterable) 

The join() method takes iterable – objects capable of returning its members one at a time. 
Some examples are List, Tuple, String, Dictionary and Set

1) join() is used for joining list and str
'''

'''
number within 10 of 100 means positive difference should be <=10
in python we abs() to get absolute difference b/w 2 numbers
ex:
num=104
num is within 10 of 100
104-100=4<=10
'''

lst=['1','2','3']   # items in iterable shoud be str
s='-'
s=s.join(lst)
print(s)   # 1-2-3

# Joining with empty string

s=['i' 'am' 'home']
st=' '.join(s)
print(st)

'''
SUMMER OF '69: Return the sum of the numbers in the array, 
except ignore sections of numbers starting with a 6 and extending to the next 9 
(every 6 will be followed by at least one 9). Return 0 for no numbers.
summer_69([1, 3, 5]) --> 9
summer_69([4, 5, 6, 7, 8, 9]) --> 9
summer_69([2, 1, 6, 9, 11]) --> 14

def summer_69(arr):
    total = 0
    add = True
    for num in arr:
        while add:
            if num != 6:
                total += num
                break
            else:
                add = False
        while not add:
            if num != 9:
                break
            else:
                add = True
                break
    return total
'''

'''
inux commands

1)ls--to view all files available on particular location
2)cd F://--to change directory to F drive
3)mkdir 'file name'--to create new folder with name inside single quotes
4)cd 'folder name'--to navigate to folder 
5)cd ..-- to back from folder
6)rm file name--to delete file
7)crtl+A--to move to left most side
8)ctrl+E--to move right side most
9)ls -l-- to see content horizantally
10)ls -l r*-- to see contents starts with r
11)ctrl+l or clear--to control the screen
'''
'''
if __name__ == “__main__”:

1)__name__ is a built in function that allows us to know if the script is being run directly or imported
2)if this file is being imported from another module __name__ will be that modules name 

class Animal:
    def __init__(self,name):
        self.name=name
        
    def eat(self):
        print(self.name,'is am eating')
        
a=Animal('dog')

if __name__=='__main__':
    a.eat()                   # if this code is being imported by othr files 
                            #then function call will not run
    
'''

'''
unittest is used to test the code if its giving desired output
'''
import unittest
class TestCap(unittest.TestCase):  # class extends TestCase from unittest
    def test_one_word(self):
        text = 'sridhar r'
        result = text.title()  # title() capitalizes first letter of evry word
        self.assertEqual(result, 'Sridhar R')

    def test_split(self):
        s = 'hi sridhar welcome'
        self.assertEqual(s.split(), ['hi', 'sridhar', 'welcome']


unittest.main()