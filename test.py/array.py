from array import *
'''
val=array('i',[10,20,20,40])
print(len(val))
print(val)

for i in val:     #interating with for loop
    print(i)

length=len(val)   #iterating using while loop
i=0
while i<length:
    print(val[i])
    i+=1


#copy of cloning of array

array1=array('f',[10.0,20.0,30.0,40.0])
newarray=array(array1.typecode,(i*2 for i in array1))   # new array to print twice that of array
print('updated array is =',newarray)

'''

# TO TAKE ARRAY VALUES FROM USER
userArray=array('i',[])     # declaring empty array

n=int(input('enter the range of array'))   #  range input from user

for i in range(n):
    x=int(input('enter next value of an array'))
    userArray.append(x)

print(userArray)

#SEARCH USER INPUT IN ARRAY

val=int(input('enter the value to be searched in an array'))

for j in userArray:
    if j==val:
        print('value entered is exists at index =',userArray.index(j))
        break
else:
    print('value entered is not exists')


print(userArray.index(val))    # single line to get the index

k=0    # initial index
for e in userArray:
    if e==val:                      #TO SEARCH THE INDEX OF USER VALUE
        print('entered value presents at index =',k)
        break
    k+=1
else:
    print('value entered is not exists')





