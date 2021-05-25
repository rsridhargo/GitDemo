# content and reference comparison
l1=[10,20,30,40]
l2=list(i for i in range(1,41) if i%10==0)
l3=l1
print(l1)
print(l2)
print(l1==l2)         # “==” returns True if two variables have same values(or content).
print(l3 is l1)       # “is” returns True if two variables point to the same object.
print(40 in l1)
print(id(l1))
print(id(l3))
print(id(l2))


x=[[10,20,30],[40,50,60],[70,80,90]]
print(x)
print('row wise representation')
for i in x:
    print(i)

print('matrix representation')
for i in x:
    for j in i:
        print(j ,end='  ')
    print()                #for next line


#WAP TO DELETE DUPLICATE ELEMENTS FROM LIST

a=[1,2,3,5,7,6,8,4,2,6,4,3,9,7]
b=[]
print('Initial list is =',a)

for i in a:
    if i not in b:
        b.append(i)
print('final list is =',b)

l=['RAKESH','RAMESH','SURESH']
l1=[i[0] for i in l ]
print(l1)

l2=[1,2,3,4]
l3=[3,4,5,6]
l4=[x for x in l2 if x not in l3]
l5=[x for x in l2 if x in l3]
print(l4)
print(l5)

#WAP TO FIND UNIQUE VOWELS IN WORD

v=['a','e','i','o','u']
s='sriiiidhaar'
l=[]
for i in s:
    if i in v and i not in l:       #avoids duplication
        l.append(i)
print('list of vowels found in given name are =',l)

#cloning of list using slicing operator

a=[1,2,3,4]
#b=[]
b=a[:]
print(b)

# round() rounds off to the given number of digits if no digits given rounds off to nearest integer
# syntax = round(number, number of digits)
print(round(51.6))     # output 52
print(round(15.1487,2))   # output 15.14

'''
string indexing slicing
syntax  string_name[a:b:c]
a--starting index
b--ending index
c--step index
s[::]   prints entire string
s[::2]  prints even index values from 0
s[1::2]  prints odd index values

'''
s='sridhar'
print(s[::])
print(s[::2])    # prints even index values
print(s[1::2])   # prints odd index values

# WAP to print even indexed and odd indexed values separated by comma
a=int(input('enter number'))
for i in range(a):
    s = input('enter string')
    print(s[0::2],s[1::2])
    break