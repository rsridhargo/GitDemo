# insertion order not fixed,no duplicate elements
'''
s=set(x for x in range(5))
print(s)
s={10,20,30,10,50,30,40,30}     # no duplication allowed
print('no duplicate items allowed',s)
s=set()     # creating empty set
print(type(s))

# add() method , adds item to set if item not already in set

s1={10,20,30,40,50}
s1.add(60)           # if we pass iterable it will add as it is
print(s1)
for i in range(101):
    if i%10==0:
        s1.add(i)
print(s1)

# update() updates set with other set,list or tuple
s1={'mango','apple','orange','grapes'}
s2={'banana','lemon'}
s3={'chiku','melon'}
s2.update(s1,s3)        #multiple iterable items can be added
print(s2)

# remove()  removes item if exists if not throws error

def Remove(s1):
    s1.remove('mango')
    print(s1)
s1={'mango','apple','orange','grapes'}
Remove(s1)

# discard() if item not found doesnt throws any error just prints original set
def Discard(s1):
    s1.discard('app')      # no error
    print(s1)
s1={'mango','apple','orange','grapes'}
Discard(s1)

#intersection() return set of common elements in 2 set, if no common return empty set
# & used for intersection.

s1={1,2,3}
s2={2,3,4}
s3={3,4,5}
s4=s1.intersection(s2)   # return common elements of s1 and s2
s5=s1.intersection(s2,s3)  # return common elements of s1 and s2 and s3
print(s4)
print(s5)

# union() return combined set of 2 or more sets   | used for union.
s1={1,2,3,4,5}
s2={4,5,6,7}
s3={7,8,9,10}
s4=s1.union(s2)
print('union of s1 and s2 =',s4)

s5=s1.union(s2,s3)
print('union of s1 and s2 and s3 =',s5)

# difference()  diffrence b/w two sets   - can be used

s1={1,2,3,4,5}
s2={4,5,6,7}
s3={7,8,9,10}
s4=s1-s2
print(s4)
s5=s1.difference(s2,s3)
print(s5)

#symmetric difference ,diffrence b/w two sets   ^ can be used

s1={1,2,3,4,5}
s2={4,5,6,7}
s3={7,8,9,10}
s4=s1^s2
print(s4)
s5=s1.symmetric_difference(s3)
print(s5)

# difference update()  ,updates calling set with difference items b/w 2 sets
s1={1,2,3,4,5}
s2={4,5,6,7}
s3={7,8,9,10}
s1.difference_update(s2)
print(s1)
s1.symmetric_difference_update(s3)     # updates difference b/w s1 and s3 to s1
print(s1)

s={10,20,30,40,(50,60)}     `q1r  # set should have only immutable item,list and set cant be used inside set
print(s)
'''

s1=set([1])
s2=set((2,4,2,4))
s3={3,4}
print(s1)
print(s2)
print(s3)
s2.add(3)
s1.update([2,3])
s3.add(5)
print(s1)
print(s2)
print(s3)
'''
s1.remove(1)
s2.discard(2)
s3.remove(3)
print(s1)
print(s2)
print(s3)
'''

s4=s1.intersection(s2)
s5=s1.intersection(s2,s3)
s6=s1|s2
s7=s1.union(s2,s3)
s8=s1-s2
s9=s1.difference(s2,s3)
s10=s1.symmetric_difference_update(s2)
print(s4)
print(s5)
print(s6)
print(s7)
print(s8)
print(s9)
print(s10)

