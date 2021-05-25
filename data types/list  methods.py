#append() adds items at the end of list
'''
list1=[10,20,30.50,'mango','apple']
print('list before append =',list1)
list1.append(50)
list1.append(60.80)
list1.append('grapes')
print('list after append =',list1)

#using for loop to append

list2=[]
for i in range(1,100):
    if i%5==0:
        list2.append(i)
print(list2)

#appending input from user

list1=['mango','orange','grapes','apple']
print(reversed(list1))
for i in reversed(list1):
    print(i)
print(sorted(list1))
print(sorted(list1,reverse=True))



list1=['mango','orange','grapes','apple']
x=input('enter item to be searched')
for j in list1:
    if j==x:
        print('{0} item found at position {1}'.format(x,list1.index(x)))
        break
else:
    print('item not found')


#insert() method

list1=list(x for x in range(1,101) if x%10==0)
print('list before insertion',list1)
list1.insert(1,15)
list1.insert(2,25)
list1.insert(3,35)
list1.insert(30,55)       #adds item at the end as out of range
list1.insert(-40,45)      #adds item at first
print('list after insertion',list1)
'''

#extend() method
# only iterable items can be added

list1=list(x for x in range(1,101) if x%10==0)
print(list1)
list2=[]
list2.extend(list1)

print(list2)

l1=[10,20,30,40]
l2=['mango','grapes','apple']
l3=[]
l1.extend(l2)
l2.extend(l1)
l3.extend(l2)
print(l1)
print(l2)
print(l3)

# remove() removes desired element,working mechanism is based on item , valueError if item not found

lst=[10,20,30,40,50,60,145,687]
print('list before deleting',lst)
lst.remove(145)
lst.remove(687)
#lst.remove(1445)    #ValueError as item not found
print('list after deleting',lst)

for i in lst:
    if i%4==0:
        lst.remove(i)
print(lst)

# pop() deletes item at specified index,working mechanism is on index based,
lst1=['bbad','ssad','mmad','ddad']
lst1.pop()
lst1.pop(1)
#lst1.pop(10)      # index out of range error
print(lst1)

list1=['amango','orange','apple']
for i in list1:
    if i[0]=='a':
        list1.pop(list1.index(i))
print(list1)

# del function,deletes items from the range of index mentioned,if index not mentioned delets all items in list
list1=[1,2,3,4,5,6]
del list1[2:4]

# copy() if we need the original list unchanged when the new list is modified,
# you can use copy() method. This is called shallow copy.

a=[10,20,30,40,50,60]
b=a.copy()
print(b)
b.append('cat')       #change only in new list
print('Old List: ', a)
print('Old List: ', b)


# deep copy()  changes made in new copy will not affects the origical list

import copy    # importing copy module

lis1 = [ 1, 2, 3, 4 ]
lis2 = copy.copy(lis1)
lis3 = copy.deepcopy(lis1)
#print(lis2)
print(lis3)
lis1.append(6)
print(lis1)
print(lis3)

# clear() clears the list to empty list

a=['dog','cat']
print('before clear',a)
a.clear()
print('after clear',a)

# count() return the number of times occurance of value

lis1 = [ 1, 2, 3, 4,1,6,4,1,8,2,6 ]
print(lis1.count(1))

#index() return the index of value (default first occurance in case of duplicate)

#syntax is  list_name.index(element, start, end)

a=['dog','cat','cow','cat','lion','cat']
print(a.index('cat'))
print(a.index('cat',2,4))
#print(a.index('cow',4,8))       # error out of range











