
'''
dictionary is unordered data type
Dictionary holds key:value pair.
key should be immutable and no key duplication allowed
value can be mutable and duplication allowed
dictionary can be created using {} or using dict() function

'''
d1={}       # empty dictionary
d2={'name':'sri','age':24,'city':'Bengaluru'}
d3={1:'sri',2:'chethan',3:{4:'muki',5:'dj'}}     # nested dictionary (key in nested dict can be duplicate)
#d4={[1,2]:'A',4:'B'}       # key should be immutable   here list given
print(d1)
print(d2)
print(d3)

# adding element to dictionary
d1[1]='sri'   # adding new element
print(d1)
d2['age']=25    # updating the existing one
d2['address']={1:'hi',2:'welcome'}      # adding a nested dictionary
print(d2)
print(d3.keys())

# Accessing elements from a Dictionary
d2={'name':'sri','age':24,'city':'Bengaluru'}
d3={1:'sri',2:'chethan',3:{4:'muki',5:'dj'}}
print(d2['name'])        # accessing using key
print(d3.get(3))   # using get()  ...it throws no error if key not found
print(d3[3][5])     # accessing nested dictionary  (cant be done for nested

for i in d3:
    print(i,d3[i])     # accessing al elements in dictionary using for loop

# Removing Elements from Dictionary

# using del keyword  we can delete desired key value and entire dict

d2={'name':'sri','age':24,'city':'Bengaluru'}
d3={1:'sri',2:'chethan',3:{4:'muki',5:'dj'}}

del d2['age']    # deleting item using del
print(d2)
del d3[3][5]      # deeting item using del in nested dictionary
print(d3)
#del d2            deletes entire dictionary
print(d2)

# using pop()     we can delete any desired value using key

d2={'name':'sri','age':24,'city':'Bengaluru'}
d3={1:'sri',2:'chethan',3:{4:'muki',5:'dj'}}

d2.pop('city')
print(d2)
d3.pop(3)
print(d3)

# popitem()  deletes random key value

d2={'name':'sri','age':24,'city':'Bengaluru'}
d3={1:'sri',2:'chethan',3:{4:'muki',5:'dj'}}

d2.popitem()    # deletes 'city' randomly
print(d2)

for i in range(1,len(d3)):
    print('Rank '+ str(i) +' '+str(d3.popitem()))

# copy() returns the shallow copy of dictionary
d1={1:'sri',2:'chethu',3:'muki'}
d2=d1.copy()      # gives shallow copy of d1
print(d1)
print(d2)
d2.clear()    # clearing one dict doesnt affects another
print(d1)
print(d2)

# difference b/w shallow and deep copy
d1={1:'sri',2:'chethu',3:'muki'}
d2=d1.copy()   # shallow copy of d1
print('original dict' ,d1)
print('shallow copy of d1',d2)
d3=d1     # deep copy of d1
print('deep copy of d1',d3)

#d2.clear()   # deletes just d2
print('original dict post clear',d1)
print('new dict post clear',d2)

d3.clear()     # deletes both d1 and d3
print('original dict post clear',d1)
print('new dict post clear',d3)

# values()  method returns list of all the values available in a given dictionary.

d1={1:'sri',2:'chethu',3:'muki'}
print(d1.values())       # output is ['sri','chethu','muki']
d2={'sri':100,'chethu':200,'muki':300}
print(sum(d2.values()))    # gives sum of values

# keys()  keys() method in Python Dictionary,
# returns a view object that displays a list of all the keys in the dictionary.

d1={1:'sri',2:'chethu',3:'muki'}
print('keys are',d1.keys())
d1[4]='dj'
print('keys are',d1.keys())   # keys will also be updated


# update() method updates the dictionary with the elements from
# the another dictionary object or from an iterable of key/value pairs.

d1={1:'sri',2:'chethu',3:'muki'}
d2={4:'dj',5:'raj'}
d1.update(d2)    # d2 wil get added to d1 dict
print(d1)
d2.update({6:'sham'})    # any dictionary can be added
print(d2)
d1.update(A='sri' , B='muki')    # ITERABLE WITH KEY VALUE PAIR ADDED
print(d1)
#d1.update('sr')    # error only key value pair can be added
print(d1)


