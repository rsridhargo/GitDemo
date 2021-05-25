'''
1)Numpy(numerical python) is a general-purpose array-processing package.Its written in C and python
2)It provides a high-performance multidimensional array object, and tools for working with these arrays.
3)It is the fundamental package for scientific computing with Python.
4)NumPy array essentially comes in 2 flavors
a) vectors ==1D array
b) matrices ==2D array
5)An array class in Numpy is called as ndarray
6) In Numpy, number of dimensions of the array is called rank of the array

we can numpy for
a)we can perform mathematical operations on array
b)linear algebra related operations
c)random number generation
d)broadcasting

python list vs numpy array

similarities
1)both used to store dat
2) both are mutable
3)both can be indexed and iterable
4) both can be spliced

differences
1) The list can be homogeneous or heterogeneous. (can store different or same datatype items),
where as numpy array is homogeneous
2) python list are by default 1D But we can create a n Dimensional list .But then to it will be 1 D list storing another 1D list
where as numpy array can be multi D.
3)numpy array consumes less memory and faster because they are stored in contagious memory locations
where as lists are stored in non contagiuos memory locations
--contagious memory-- memory blocks where there is no gap b/w 2 memory blocks (array)
--non contagious memory-- memory blocks with gap b/w memory blocks(hash tree)
4)list is in built whre as numpy has to be imported
5) we can mathematical operations on numpy arrays like multiplying array or dividing,
whereas its not supported in python list.
6) if the array contains rows and columns then we can call it as 2d array
'''

'''
data types in numpy
1)bool--True / False
2)int--interger type
   a)int_  --Default integer type (same as C long; normally either int64 or int32)
   b)intc--identical to c (normally int32 or int64)
   c)intp--Integer used for indexing (same as C ssize_t; normally either int32 or int64)
   d)int8--8 bit integer type
   e)int16-- 16 bit(2byte) integer  similarly int32,int64
   f)uint8,uint16,uint32,uint64 are unsigned means only positive values
3)float--float16,float32,float64
4)complex--complex_(complex128)
    a)complex64--Complex number, represented by two 32-bit floats (real and imaginary components)
    b)complex128--Complex number, represented by two 64-bit floats (real and imaginary components)
    
i--for integers (i2-int16,i4-int32,i8-int64)
u--for unsigned integers   (u2-uint16....)
f--float  ((f2--float16,f4--float32,f8--float64)
c-complex (c2--complex16...)
5)default int data type is int32,float32,complex32

type vs dtype
type--The type of a NumPy array is numpy.ndarray; this is just the type of Python object 
it is (similar to how type("hello") is str for example).

dtype--dtype just defines how bytes in memory will be interpreted by a scalar" → 
it also defines the way in which they're interpreted (eg. int32 vs float32).
'''
# we can change the datatype of array using astype() method
import numpy as np
arr=np.array([1,2,3])
print(arr)
print(arr.dtype)   # int32 is type
arr1=arr.astype(float)   # changes the datatype to float
print(arr1.dtype)

# another way to change datatype

n=np.arange(1,11,dtype='u8')
print(n.dtype)
n=np.int64(n)                 # passing n to data type function
print(n)
print(n.dtype)

'''creating data types

#syntax is 
numpy.dtype(object, align, copy)
object-- to be converted to data type object
'''
dt=np.dtype(np.int64)
print(dt)
dt1=np.dtype('u8')      # uint64
print(dt1)

# structred data type
sd=np.dtype([('name','S20'),('age','int8'),('marks','f')])
print(sd)

# creating numpy array using above datatype
array1=np.array([('abc',10,50.00),('xyz',20,30.00)],dtype=sd)
print(array1)
print(array1.dtype)

# creating arrays
# arrays can be created using array() from lists,sets

import numpy as np
l=[1,2,3]
a=np.array([l])    # passing list to create array
print(a)
print(a.dtype)

b=np.array([(1,2,5),(7,6,4)],dtype='int')    # creating 2D array with nested tuples

c=np.array([(1,2,3),(5,6,8),(7,9,2)],dtype='i8',ndmin=4)   # ndmin is the minimum dimensions

'''
arange()
1)arange() is same as range() difference is its gives numpy array whre as range gives python list
2) syntax is 
numpy.arange([start, ]stop, [step, ]dtype=None)
3) output array is always 1D,we can reshape it to multi D using reshape()
4)when the step is float sometimes the stop num is also considered
'''
b=np.arange(10)
print(b)
print(b.dtype)
print(b.ndim)


b=np.arange(10,0,-2)     # step can be -ve also

c=np.arange(1,11,dtype='i4')           # we can mention dtype also
print(c)

'''
linspace
1)stands for linear space
2)works almost similar to arange function with few changes(no of parts)
3)syntax is
numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
    a)start--start of the range
    b)stop--end of range
    c)num--no of parts we need in the specified range,default is 50
    d)endpoint--if false it will exclude stop,default is True
    e)retstep--if True it will return the step or spaces b/w 2 no,default is False
    f)dtype can be changed
4)returns !D array and can be reshaped
'''
d=np.linspace(1,11)    # default parts will be 50
print(d)

d=np.linspace(1,10,10)   # gives 10 equally spaced parts

d=np.linspace(1,10,10,endpoint=False)  # endpoint default is True,if set to False then stop will be excluded

a=np.linspace(1,20,10,retstep=True)    # retstep True means returns step
#print(a.dtype)     # throws error as retstep gives tuple of array and step and dtype cant be obtained

a=np.linspace(1,20,10,dtype=int)   # dtype can be changed

'''
logspace
1)gives linear spaced array similar to linspace
2)In linear space, the sequence starts at base ** start (base to the power of start) 
and ends with base ** stop (see endpoint below).
3) syntax is
numpy.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0)
    a)start--base ** start is the starting value of the sequence.
    b)stop--base**stop is the end of sequense unless endpoint is False
    c)num--integer value saying total no of parts,default is 50
    d)endpoint--if set to false will exclude base**stop,default is True
    e)base--float base,default is 10.00
    f)dtype--can be changed
'''
a=np.logspace(1,5)
print(a)

b=np.logspace(1,5,10,base=2)   # num and base can be changed

c=np.logspace(1,5,10,base=2,endpoint=False,dtype='f4')  # dtype can be changed

'''
zeros
1)gives array filled with zeros of given shape and type
2) syntax is 
numpy.zeros(shape, dtype=float, order='C')
    a)shape-shape of array like 2 or tuple/list of int(4,3)
'''
a=np.zeros(10)

a=np.zeros((4,3),dtype='u4')
print(a)
print(a.dtype)
print(a.ndim)

# same way ones() and empty() functions are used

a=np.ones((4,3),dtype='u4')   # gives array filles with 1's
print(a)
print(a.dtype)
print(a.ndim)

# empty gives random uninitializwd array

e_array=np.empty((2,3))   # uninitialized random enties
print(e_array)

'''
eye()
1)returns array filled with 0's other than kth diagonal from main diagonal which are 1's
2) syntax is 
numpy.eye(N, M=None, k=0, dtype=<class 'float'>, order='C')[source]¶
    a)N--no of rows in output
    b)M--no of columns in output,default will be N
    c)k--index of diagonal(0-main diagonal,+ve--upper to main,-ve--lower to main diagonal)
    d)dtype--default is float
    
[[a, b, c]
 [d, e, f]
 [g, h, i]]
 
 a,e,i---main diagonal (k=0)
 b,f--   upper diagonal(k=1)
 c-------upper diagonal(k=2)
 d,h-----lower diagonal(k=-1)
 g-------lower diagonal(k=-2) 
'''
a=np.eye(5)  # gives 5x5 2D array with 0's and all diagonals with 1

b=np.eye(3,3,k=1)   # prints array with upper diagonal values as 1 and rest as 0
print(b)

c=np.eye(3,4,k=-1)     # 3x4 array with lower diagonal values 1
print(c)

'''
numpy.identity(n, dtype=None)[source]
returns n*n array with 0's other than diagonals
'''
a=np.identity(3,dtype='i4')
print(a)

'''
random
1)used to generate random number
2)4 Compatibility functions
    a)rand--Uniformly distributed values.(gives values from 0 to within 1)
    b)randn--Normally distributed values.
    c)ranf--Uniformly distributed floating point numbers.(random float values b/w 0 and 1)
    d)randint--Uniformly distributed integers in a given range.
'''
import numpy.random   # gives only +ve values
a=np.random.rand(5,5,5)  # 3D array,with 5 rows and 5 columns and 5 blocks
print(a)

import numpy.random
b=np.random.randn(4,3)    # randn
print(b)

import numpy.random
b=np.random.ranf((5,3))    #ranf gives only +ve value b/w 0 and 1
print(b)

# randint syntax is randint(low, high=None, size=None, dtype='l')
# only low is mandatory
# if high is not given then low will become high and low will be 0
# no of random no depend on size,default is 1

a=np.random.randint(5)   # one random no is given
print(a)

a=np.random.randint(5,size=5)   # 5 random no are given
print(a)

a=np.random.randint(2,5,size=(4,4,4))    # gives 3D 4 blocks,4 rows and 4 columns array random nos
print(a)

# ATTRIBUTES OF NUMPY ARRAY
# ndim--used to find the dimenstions of array
arr=np.array([[1,2],[5,7]])
arr.ndim
np.ndim(arr)  # same as above

# shape gives the  tuple of array dimensions
#shape(2,2,3,4,5) no of values inside shape tuple equals the dimensions of array,
# last 2 no will be rows and coulumns of that array

a=np.eye(3,4,k=2)
print(a.shape)    # gives rows and columns

a=np.ones((4,2,3),dtype='f8')   # 4 here is the no of blocks
print(a.shape)

# shape can be changed and new shape should be compatible to old one(size should be same)

a=np.ones((4,3))
print(a.shape)
a.shape=(2,6)    # here 2*6=4*3
#a.shape=(3,3)    # throws error as size mismatch

# size---product of shape (total no of elements in an array)
a=np.ones((4,3))
print(a.size)

#itemsize --gives size of each element of an array in bytes
print(a.dtype)   # float64
print(a.itemsize)  # gives 8 as float64/8=8bytes

'''
NumPy - Array From Existing Data
    a)asarray
    1)this is similar to array with only fewer information
    2)numpy array can be created with list,tuple,dict or nested list
    3)syntax is 
    numpy.asarray(a, dtype = None, order = None)
    
   
'''
l=[10,20,30,40]
import numpy as np
n=np.asarray(l)

tp=((1,2,3),(4,5,6))   # nested tuple
np1=np.asarray(tp)
np1
'''
numpy.frombuffer(buffer, dtype = float, count = -1, offset = 0)
    a)buffer--input string kinda data
    b)count--no of items to be read,all data in default
    c)offset--starting reading of the data,default is 0
'''
s=b'Hello world'  # byte string
np2=np.frombuffer(s,dtype='S1')
np2

s=b'Hello world'  # byte string
np2=np.frombuffer(s,dtype='S1',count=6,offset=5)   # start from index 5 and count is 6
np2
'''
numpy.fromiter
1)used to create numpy 1D array from iterables
2)syntax is
numpy.fromiter(iterable, dtype, count = -1)

'''
import numpy as np
l=list(x*2 for x in range(5))
a=np.fromiter(l,dtype='u4')      # dtype is mandatory

import numpy as np
l=list(range(10))
a=np.fromiter(l,dtype='f4')    


'''
Indexing of numpy array
1)indexing is use [] to get the value of array
2)indexing of numpy array is same as python list
'''
# indexing of 1D array
x=np.arange(10)
print(x[5])          # gives value at index 5 (index always starts with 0)
print(x[2:5])       # gives values from 2 to 5 but excludes value at 5
print(x[2:9:2])      # step is 2

# indexing of 2D array

'''
        -1 -2 -3 -4 -5
         |  |  |  |  |
  0<---[[0  1  2  3  4]--->-2
  1<---[ 5  6  7  8  9]]--->-1
         |  |  |  |  |
         0  1  2  3  4
if 2D array is a then indexing is a[i][j] where i is row index and j is column index
efficient way of indexing 2d array is a[i,j]
'''
import numpy as np
a=np.arange(12,dtype=float).reshape(4,3)
print(a[2])     # gives 2nd row
print(a[2][1])    # gives 2nd rows index 1 values
print(a[2,1])      # recomended

print(a[1:3:2])    # slicing is similar to list slicing

# slicing 3D numpy arrays
'''
array([[[ 0.,  1.],      i=0
        [ 2.,  3.]],

       [[ 4.,  5.],      i=1
        [ 6.,  7.]],

j=0     [[ 8.,  9.],      i=2
j=1      [10., 11.]]])
           |   |
          k=0  k=1
          
syntax is array_name.[i,j,k]  where i=blocks index, j=rows index,k=columns index
slicing will have 3 parts respectively a[1:3:2,1:4:2,2:4:2]

slicing and indexing is same as 2D array except 3 elements will come

The dots (...) represent as many colons as needed to produce a complete indexing tuple.
 For example, if x is an array with 5 axes, then
 
 x[1,2,...] is equivalent to x[1,2,:,:,:],

x[...,3] to x[:,:,:,:,3] and

x[4,...,5,:] to x[4,:,:,5,:].

1)...--can be treated as : in slicing

2)an index can only have a single ellipsis ('...')
'''
import numpy as np
b=np.arange(12,dtype=float).reshape(2,2,3)
print(b[1,2:,:2])

a[...,1,2]   # equals to a[:,1,2]

'''
slicing
1)in slicing we can get multiple values
2)slicing is same as list slicing
a[start:stop:end,start:stop:end,start:stop:end]   for 3D array
'''
'''
advanced indexing
1)integer indexing
2)boolean indexing
3)index can be passed as list or ndarray
4)we can get repeated values
'''
b=np.arange(1,10,2)
l=[1,3,2] # list of indices we need to get
print(b[l])   # passing list of index to numpy array
print(b)

x=np.array([1,4,0,1,4])    # repeated index can be obtained
print(x)
b[x]

# a[1,2]-----gives 1D array
# a[[1,2,3]]---gives 2D array

import numpy as np
n=(np.arange(12)*2).reshape(3,4)
n[[1,0],[2]]                     # gives rows at 1 and 0 and also 2nd column
n[[1,2,0],[2,1,0]]    # we can pass as many as indices separeatly

n=(np.arange(12)*2).reshape(3,4)
row=np.array([[1,2],[2,0]])       # 2D array
column=np.array([[0,2],[1,2]])    # 2D array

n[row,column]                  # gives 2D array as output

'''
array([[ 8, 20],   
       [18,  4]])
    
8--(1,0)
20--(2,0)
18--(2,1)
'''

n[row,2]    # compares each element with 2   first element will be (1,2) (2,2)..

# boolean indexing (condition checking)
# checking the the condition if condition met returns True else False and
# when condition passed to array returns those values which returns True values

n%4==0  # returns True for all values divided by 4

n[n%4==0]   # returns array of the values contained array([ 0,  4,  8, 12, 16, 20])

n[n>10]   # returns all the values > 10

n[n>4]=-1   # gives ndarray with values greater than 4  equals to -1

a = np.arange(12).reshape(3,4)
a[[True,False,True]]   # gives rows of True positions
a[[True,False,True],[True,False,True,False]]   # we should pass booleanb values in correct shape as else eror

'''
mathematical operations
normal linear algebra operations addition,sub power with scalar or other array
'''
import numpy as np
n=(np.arange(12)*3).reshape(4,3)
print(n*2)   # multiplies the all the elements of n to scalar 2

m=(np.arange(12)*4).reshape(4,3)
print(n+m)

# arithmetic operations can be done on any array with scalar and other arrays with same shape

'''
Broadcasting
1)The term broadcasting refers to how numpy treats arrays with different Dimension 
during arithmetic operations which lead to certain constraints, 
the smaller array is broadcast across the larger array so that they have compatible shapes.
2)rules for broadcasting are
    a)the size of each dimension should be same
    b)size of the one of the dimension should be 1
    
steps
    a)check if the dimensions of the 2 arrays same
    b)if not same then 1s will be padded to the left side of the array with fewer dimensions
    c)then check the size of each dimension if same or any one is 1 check next dimension
    d)if conditions not met then two arrays cant be broadcasted
    e)the resultant shape will be array with higher shape and dimensions
'''

x = np.arange(4)   # shape is (4,)
y = np.ones(5)      # shape is (5,)

# dimension is same
# but size of each dimension is not same and neither of them is 1 (4 is not equla to 5)

x = np.arange(4)        # shape is (4,)
xx = x.reshape(4,1)      # shape is (4,1)

''' 
1)dimensions are not same so padding 1 to x array to make it compatible with xx
new shape is ,

x----( 1,   4)
       |    |
       !=  !=    (but both have 1 in size)
xx----(4,   1)

so broadcast rule meets and result array is of shape (4,1)

'''
x = np.arange(4)
z = np.ones((3,4))

'''
x---(1,  4)   # 1 is padded to the left side
     |   |
    !=   =     # 1 dimension size is equal,but second one is not but one array has 1
z---(3,  4)

so criteria is met and gives resultant array of shape (3,4)
'''
x = np.arange(3).reshape(3,1)
y=np.arange(3)*3

'''
x----(3,  1)
      |   |
      !=  !=   both the size are not equal but contains 1
y----(1,  3)


 x will be converted from 
 [[0]
 [1]
 [2]] to [0 0 0
          1 1 1
          2 2 2]
                       
y will be converted from [0 3 6] 

                     to [0 0 0
                        3 3 3
                        6 6 6]
result will be (3,3) shape
'''

'''
array manipulation
reshape
    a)reshapes array without changing the original data
    b)syntax is   numpy.reshape(array, shape, order = ‘C’) 
    c)shape--int or tuple of ints
    d)new shape should be equal to size of old array
    e)changes made are temporary doesnt alters original array
'''
import numpy as np
a=np.arange(8)
x=a.reshape(2,4)   # gives 2D array with 2 rows and 4 columns
y=a.reshape(2,2,2)   # gives 3D array with 2 rows and 2 columns

# no of int says the dimensions of array

m=np.reshape(a,(2,2,2,2))   # a can be passed inside along with tuple of ints

'''
resize
1)it will change the data of the array
2)it will fill with zeros if the size of new shape is more than old one
only when using array_name.resize
3)it will be filled with same repeated values of data if we use numpy.resize(array_name,shape)
4)when we use numpy.resize(a,shape) it will not alter the original array
5)it will alter the original array while using dot notation
'''

import numpy as np
a=np.arange(12)
b=np.resize(a,(5,5))   # fills with same repeated values and doesnt  changes the original array

a.resize(6,4,refcheck=False)   # refecheck is used to  check if the array is used by any others

# above output is filled with 0 if size is more and will change the original array

'''
flatten
1)returns copy of ndarray flattened to 1D
2)changes made doesnt affects the original array as gives copy of the flatten array
2)ndarray.flatten(order='C')
    a)order is 'C' means row major and 'F' means column major
'''

a=np.arange(12).reshape(4,3)
a.flatten()       # default order is C row wise

a.flatten(order='F')   # 1D column wise array

'''
ravel
1)ravel doesnt returns the copy of array only returns reference or view
2)faster than flatten as ravel occupies no memory
3)changes made in ravel affects the original array as no copy is given
'''
a=np.arange(12).reshape(4,3)
b=a.ravel()                  # returns only view or reference
b[2]=100   # original a array also affected
c=np.ravel(a)  # array can be passed to ravel method also

'''
flat
1)gives 1D flat object which can be iterared
'''
a=np.arange(12).reshape(4,3)
a.flat  # gives iterable object

for i in a.flat:   # can be iterated over flattened object gives all the valuues
    print(i)

'''
swapaxes
1)used to swap axes of the array
2)syntax is numpy.swapaxes(array,axes1,axes2)
3)3D array has 3 dimensions x(0),y(1),z(2)
'''
import numpy as np
a=np.arange(10)*2
a.resize(2,4,3,refcheck=False)   # resizing array

a.swapaxes(0,1)

a.swapaxes(0,2)  # columns are swapped
'''
shape of a---(2, 4, 3)
              |  |  |
        axes  0  1  2
              |  |  |
swapped shape(4, 2, 3)
        
a.swapaxes(0,1)  # here we are swapping the axes of x,y and z remains same
 
 if we are swapping 1 means rows and 2 means columns swapping
 
in transpose we can swap the axes of all 3 dimensions whereas in swapaxes we can swap only 2 dimensions
'''
'''
concatenate
1)used to concat 2 or more arrays
2) syntax is 
numpy.concatenate((a1,a2,a3...),axis=0,out=None)
    a)a1,a2...--arrays to be concatenated
    b)axis can be either 0 or 1,o for rows and 1 for coumns
    c)if axis is None then arays will be flattened and concat
    d)dimensions of arrays should be same
    e)if axis is 0 (i.e concat in rows) then columns size should be same and vice versa
    f)if out is mentioned then resultant array will be stored at that location
     
'''
a=np.arange(4)
b=np.arange(5)
#c=np.concatenate(a,b)   # throws error as per the below explaination

'''
        axis  0   1
shape of a---(1,  4)
              |   | -----axis 1 size should be same
shape of b---(1,  5)

here we are concatening at axis 0 so size at axis 1 should be same
'''

a=np.arange(4).reshape(2,2)
b=np.arange(6).reshape(2,3)

c=np.concatenate((a,b),axis=0)   # throws error as axis 1 size is not same(2 and 3)
c=np.concatenate((a,b),axis=1)   # no error as axis 0 size is same 2 and 2

'''
       axis  0   1
shape of a---(2,  2)
              |   | -----axis 1 size should be same
shape of b---(2,  3)

concat at axis 0 throws error as 2 and 3
concat at axis 1 throws no error as 2 equals to 2 
'''

np.concatenate((a,b),axis=1,out=c)   # output is stored at c

'''
stack
1)adds new arrays in new axis where as concat adds in existing array
2)the dimensions of 2 arrays should be same
3)if the dimensions of arrays adding is 2 then resultant array will have dimension of 3
4) all the input arrays should be of same dimensions

vstack
    a)vstack is same as concat in axis 0 for 2 or more dimensiuon arrays
    b)adds vertically in row wise
    
    
hstack
    a)hstack is same as concat in axis 1 for 2 or more dimensiuon arrays
    b)adds horizantally in column wise
    
dstack
    a)dstack is same as concat at axis 3
    b)adds at 3rd axis
    c)works only on arrays of 3 or more Dimensions
    
syntax is   numpy.stack((a1,a2..))-----a tuple of array can be added
'''

'''
split
1)splits ndarray  multiple sub-arrays.
2)syntax is
    numpy.split(array,sections,axis=0)
    a)array---array to be split
    b)sections----int or tuple of ints
3)if one int is given then divides array into equal no of parts,if cant be divided then raises error
4)if tuple of ints is passed then divides array at that index
ex (2,4,6) means divides array at index 2 and 6 and 6 gives subarray of [:2],[2:4],[4:6],[6:] 
5)if index is out of range then gives empty array
'''

import numpy as np
a=np.arange(10)
np.split(a,2)    # [array([0, 1, 2, 3, 4]), array([5, 6, 7, 8, 9])]
#np.split(a,3)    # throws error as total no of parts 10 is not divided by 3

# splitting 2D arrays

a=(np.arange(12)*3+4).reshape(4,3)


'''
while splitting multi D arrays 

shape of a---(4,  3)
              |   |
             2,4  3
if splitting at corresponding axis  the splitting sections should be factor of that axis size 
'''

np.split(a,3)   # here throws error as axis 0 size is not divisible by 3

np.split(a,4)   # works fine as axis 0 size is 4

# splitting at axis 1
np.split(a,3,axis=1)   # works fine


y=(np.arange(6*6*6)*3).reshape(6,6,6)
np.split(y,2)
np.split(y,3,axis=2)

# splitting with tuple of index

x=np.arange(12)+5
# splits array into 3 parts at index 2,4,5 respctively (gives [:2],[2:4],[4:5] and [5:]
np.split(x,(2,4,5))

# splitting 2D array

import numpy as np
a=np.arange(12).reshape(4,3)
np.split(a,(2,1))              # splits at axis 0 row wise

'''
hsplit()
    a)same as split at axis 1
    b)split at columns
    
vsplit()
    a)same as split at axis 0
    b)split at rows
'''
'''
array([[ 0,  1,  2],
       [ 3,  4,  5],
       [ 6,  7,  8],
       [ 9, 10, 11]])
   
np.split(a,(2,1))   # at axis gives [:2],[2:1],[1:]   # same as vsplit()

output:

[array([[0, 1, 2],     #[:2]
        [3, 4, 5]]), 
        
        array([], shape=(0, 3), dtype=int32),   # [2:1] which has no elements so empty list
        
 array([[ 3,  4,  5],           # [1:]
        [ 6,  7,  8],
        [ 9, 10, 11]])]    

np.split(a,(2,1),axis=1)  # splits at axis 1,column wise    # same as hsplit()


[array([[ 0,  1],
        [ 3,  4],   
        [ 6,  7],            # [:2] column wise split
        [ 9, 10]]), 
        
        
array([], shape=(4, 0), dtype=int32), 
        
 array([[ 1,  2],
        [ 4,  5],
        [ 7,  8],
        [10, 11]])]
'''

'''
insert()
1)inserts value at the desired index
2)syntax is ---numpy.insert(arr, obj, values, axis=None)
    a)arr--array to be inputed
    b)obj---index single or tuple of index before which values to be inserted
    c)values---values to be inserted scalar value or ndarray....ndarray should be broadcastable
    d)if axis is None then both array and the value to be added will be flattened and added along 1D
3)if scalar value is added to the multi D array then repeated values will be added to make value compatible
4)insert doesnt returns the copy of new array just returns the view
5)multiple values can be added at the same time at multiple indeces
6)if we are inserting values of different datatype,
 then first it will be type casted to data type of original type of array
'''
m=np.arange(10)
np.insert(m,1,100)   # 100 will be added before index 1
np.insert(m,1,200.754)  # float datatype will be first converted to int type as array is of type int

np.insert(m,(1,2),(100,200))  # multiple values can be inserted (we can add as many values as we can)

np.insert(m,(1,2),(100,200,300))   #throws error as no of indices and values mismatch

# inserting into ndarrays
a=(np.arange(12)+5).reshape(4,3)
np.insert(a,1,10)    # inserting without any axis ,first array will be flattened and then added at the end

'''
array([[ 5,  6,  7],    # np.insert(a,1,10,axis=0)
       [10, 10, 10],    # scalar 10 is extended to in compatible to row size
       [ 8,  9, 10],
       [11, 12, 13],
       [14, 15, 16]])
       
         0   1  2    3
array([[ 5,  6, 10,  7], # np.insert(a,2,10,axis=1)
       [ 8,  9, 10, 10],   # column is added before index 2
       [11, 12, 10, 13],
       [14, 15, 10, 16]])


array([[ 5,  6,  7],    # np.insert(a,(1,2),(10,20,30),axis=0)
       [10, 20, 30],    # adding at multiple index
       [ 8,  9, 10],    shape of array -- ---(4,3)
       [10, 20, 30],    shape of new array---(1,3)
       [11, 12, 13],    adding at axis 0 so size of axis 1 should be same (broadcastable)
       [14, 15, 16]])
       

array([[ 5,  7,  9],    # np.insert(m,(1,2,3),(10,20,30),axis=0)
       [10, 20, 30],
       [11, 13, 15],
       [10, 20, 30],
       [17, 19, 21],
       [10, 20, 30],
       [23, 25, 27]])

array([[ 5, 10,  7, 20,  9, 30],   # np.insert(m,(1,2,3),(10,20,30),axis=1)
       [11, 10, 13, 20, 15, 30],
       [17, 10, 19, 20, 21, 30],   # in adding at axis 1 at index 1 value 10 is added and so on
       [23, 10, 25, 20, 27, 30]])  # [[10],[20],[30],[40]] would have added at each index column wise
       
array([[ 2, 10,  6, 10, 10],
       [14, 20, 18, 20, 22],       # np.insert(a,(1,2),[[10],[20],[30],[40]],axis=1)
       [26, 30, 30, 30, 34],       # ndarray to be inserted should be broadcastable
       [38, 40, 42, 40, 46]])
       
for insering into multi D broadcasting rule should be satisfied
'''
'''
append()
1)append works same as insert but in insert we add elemnts at specified index
2)whereas in append we add elemnts at the end of array
3) syntax is numpy.append(array,values,axis=None)
    a)array--array to be added
    b)values--values to be added of same shape
    c)axis--if none asrray will be flattened and added
4)input array should be of same dimensions
'''
a=np.arange(10)
np.append(a,(10,10,20))  # array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 10, 20])

import numpy as np
a=(np.arange(12)*4+2).reshape(4,3)
np.append(a,([[10],[20],[30],[40]]),axis=1)   # dimensions of the array should be same and broadcastable

'''
delete()
1)deletes the array at the specified index
2)syntax is 
    numpy.delete(array,index,axis=None)
    a)array--input array in which delete has to be done
    b)index--int no or tuple of int
    c)axis--at which axis we have to delete,default is None in which array will be flattened
'''
m=np.arange(12)
np.delete(m,2)    # deletes value at index 2

a=(np.arange(12)*4+2).reshape(4,3)

np.delete(a,2,axis=0)  # deletes index 2 at axis 0 at row

np.delete(a,(2,1),axis=0)  # multiple values can be deleted

np.delete(a,(2,1),axis=1)   # deleting at axis 1

# ALL ARRAY MANIPULATION FUNCTIONS RETURNS NEW ARRAY AS ARRAY IS OF FIXED LENGTH

'''
Matrix
1)matrices are 2D numpy arrays always
2)class matrix will be depreacated in future.
3) syntax is 
    numpy.matrix(data,dtype=None,copy=True)
'''
a=np.matrix('1 2 5;4 7 3;9 5 2',dtype='f4')

'''
inverse of matrix
1)inverse of matrix is defined as below

power of matrix
1)used to multiply the matrix with itself
'''
import numpy as np
a=np.matrix('1 2 3;4 5 6;7 8 9',dtype='u4')

np.linalg.inv(a)    # used to find the inverse of the matrix

np.linalg.matrix_power(a,2)  # gives a*a
np.linalg.matrix_power(a,4)  #gives a*a*a

np.linalg.matrix_power(a,0)  # if we take n as 0 it gives identity matrix
np.linalg.matrix_power(a,-2)  # gives inverse of a and then gives a*a

'''
solving linear equation using matrix
np.linalg.solve(a,b)

2x+3y=14
5x-7y=20

a=[2  3  
   5 -7]
   
b=[14
   20]
   
answer will be [x y]

shape of result is same as b

6x+2y-5z=13
3x+3y-2z=13
7x+5y-3z=26

a=array([[ 6,  2, -5],
       [ 3,  3, -2],
       [ 7,  5, -3]])
       
b=array([13, 13, 26])

np.linalg.solve(a,b) ==array([2., 3., 1.])
'''9

'''
statistical functions
amax
1)returns maximum of an array
2)syntax is 
    numpy.amax(array,axis=None,out=None)
    
amin
1)returns minimum of an array
2)syntax is     
    numpy.amin(array,axis=None,out+None)
3)if any value has NaN then it will be the minimum value

namax
1)gives max value by ignoring any NaN values
2)syntax is
    numpy.amin(array,axis=None,out+None)
    
nanmin
1)returns minimum of an array by ignoring any NaN values
2)syntax is     
    numpy.nanmin(array,axis=None,out+None)

if no axis is passed then the array is flattened and max or min is found
axis 0 means vertically max
axis 1 means horizantal
'''
import numpy as np
a=(np.arange(12)+5).reshape(4,3)
np.amax(a) # gives max after flattening the array

np.nanmin(a,axis=0)

'''
import numpy as np
a=(np.arange(12)+5).reshape(4,3)

(min)5---[[ 5,  6,  7],---7(max)
   8-- [ 8,  9, 10],----10
   11--[11, 12, 13],----13
   14--[14, 15, 16]])---16
        |    |   |
        14   15  16
        
np.nanmin(a,axis=0)   # array([5, 6, 7])

import numpy as np
a=(np.arange(12)+5).reshape(3,2,2)

array([[[ 5,  6],
        [ 7,  8]],

       [[ 9, 10],
        [11, 12]],

       [[13, 14],
        [15, 16]]])
        
np.nanmax(a,axis=0)   # gives  [[13, 14],
                               [15, 16]]])
'''
# numpy.ptp--gives the range(max-min)
# syntax is  numpy.ptp(array,axis=None,out=None)

a=(np.arange(12)+5).reshape(3,4)

np.ptp(a)  # flattens the array and gives 11
'''
a=(np.arange(12)+5).reshape(3,4)
array([[ 5,  6,  7,  8],
       [ 9, 10, 11, 12],
       [13, 14, 15, 16]])
       
c=np.arange(3)
np.ptp(a,axis=1,out=c)

array([3, 3, 3])  
diff b/w max-min along horizantal is 3 
c should be of same size that of resultant matrix to store the results 
'''
'''
percentile
1)used to find the pecrentile of the array
2)syntax is 
    numpy.percentile(array,percentile,axis=None,out=None)
    a)array--input array
    b)percentile--0 to 100 %
    
1)percentile value=(percentile % *(n+1))/100)
2)50 percentile means 50% of values in the array are below and above that value
3)first the array has to be sorted using numpy.sort(array)
    
'''
m=np.arange(10)  # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
np.percentile(m,50)  #4.5 as if 2 no are the mid no then thier avg is taken

# 2D array
'''
b=np.random.randint(1,10,(4,3))

array([[7, 8, 2],
       [6, 6, 2],
       [5, 9, 1],
       [2, 2, 9]])
       
np.sort(b,axis=1)      # array to be sorted along axis=1
array([[2, 7, 8],
       [2, 6, 6],
       [1, 5, 9],
       [2, 2, 9]])

np.percentile(b,25,axis=1)
array([4.5, 4. , 3. , 2. ])  # if we caluclate using p*(n+1)/100 we can confirm  

for 1 column --25*(3+1)/100=1 position along 2 7 8 
       
'''
'''
numpy statistics
1)mean--is the average of all numbers,(sum of no/total no)
2)median--is the middle no of data set
median position=(total nos+1)/2
if data set contains even no then median is the vaerage of middle 2 no

3)mode--most repeated noin a data set

4)variance --is the sum of squares of differences between all numbers and means.
5)Standard Deviation--is square root of variance.It is a measure of the extent to which data varies from the mean. 

a=(10,20,30,40,50,60,70,80)

mean=(10+20+30+40+50+60+70+80)/8=45
median=(40+50)/2=45.00
mode=NA as no number is repeating
variance=((10-45)**2+(20-45)**2+(30-45)**2+(40-45)**2+(50-45)**2+(60-45)**2+(70-45)**2+(80-45)**2)/8=525.00
std=sqrt(variance)=22.9128784747792

'''
b=np.random.randint(1,10,(4,3))
np.mean(b,axis=0)   # array([5.  , 6.25, 3.5 ])
'''
b=np.random.randint(1,10,(4,3))
array([[7, 8, 2],
       [6, 6, 2],
       [5, 9, 1],
       [2, 2, 9]])
       
sorted array np.sort(b)
array([[2, 7, 8],
       [2, 6, 6],
       [1, 5, 9],
       [2, 2, 9]])
       
np.mean(b,axis=0)
array([5.  , 6.25, 3.5 ])

np.sort(b,axis=0)
array([[1, 2, 6],
       [2, 5, 8],
       [2, 6, 9],
       [2, 7, 9]])
       
np.median(b,axis=0)
array([2. , 5.5, 8.5])

np.std(b,axis=0)
array([0.4330127 , 1.87082869, 1.22474487])
'''
'''
iterating of ndarray
1)nditer is used to iterate over the array
'''
a=np.arange(12).reshape(4,3)
iter_object=np.nditer(a)
for (i) in iter_object:
    print(i,end=' ')

iter_object=np.nditer(a,order='C')   # column wise printing