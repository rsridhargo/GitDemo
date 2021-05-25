'''
pandas
1)Pandas is an open-source Python Library providing high-performance
data manipulation and analysis tool using its powerful data structures
2)Using Pandas, we can accomplish five typical steps in the processing and analysis of data,
 regardless of the origin of data — load, prepare, manipulate, model, and analyze.
3)Python with Pandas is used in a wide range of fields
including academic and commercial domains including finance, economics, Statistics, analytics, etc.
4)uses of pandas
    a)Data representation---pandas provide simpler data representation facilities
    this helps in better analysis of data and leads to better results.
    b)Less writing and more work done---its one of the best advantages of pandas,what would have taken
    large codes in python without any libraries can be achieved in pandas with shorter code
    3)Efficiently handles large data---Pandas help to save a lot of time by
    importing large amounts of data very fast.
    4)pandas deals with 3 data structures
        a)Series(1D)
        b)DataFrame(2D)--widely used
        c)Panel(3D)
'''

'''
Series
1)series is 1D array like structure
2)contains homogeneous data
3)size is immutable
4)values of data are mutable
5) syntax is
pandas.Series( data, index, dtype, copy)
    a)data ---input data like list,dict or ndarray
    b)index--index values must be unique and hashable,default is numpy.arange(n)
'''
import pandas as pd
import numpy as np
a=np.array([1,2,3,4])
print(a)
print(a.dtype)
p=pd.Series(a,dtype='u4')    # ndarray can be given as input and dtype can be changed
print(p)

t=(10,20,30,40)
p1=pd.Series(t,index=['a','b','c','d'])    # index can be changed accordingly
print(p1)


# creating series from dict

d={'x':10,'y':20,'z':30}
p2=pd.Series(d)           # note dict keys will be the index

d={'x':10,'y':20,'z':30}
p2=pd.Series(d,index=['a','b','c'])    # here we are passing index along with dict
print(p2)

# if we pass index different one and
# if the value isnt existing then it will be filled with NaN (Not a Number).

p3=pd.Series(5)   # passing just a scalar will give at 0 index
print(p3)

p4=pd.Series(4,index=[1,2,3])   # gives same no at all index if we explicitly pass index values

# Accessing Data from Series with Position
# data can be accessed using position similar to ndarray or list


lst=[10,15,20,24,47]
p4=pd.Series(lst)
print(p4)
print(p4[2])   # gives value at index 2
print(p4[2:4])   # gives index from 2 to index 3 since index 4 is excluded
print(p4[1:4:2])   # step 2
print(p4[:])       # gives entire series
print(p4[::-1])    # reverses the series

print(p4[[1,2,3]])   # values of multiple index can be obtained by passing them into a list
#print(p4[6])        # gives key error if index is not avaialble

l=[10,'sri',20.14,True]
a=pd.Series(l,dtype='O')   # if the values of data are different types then dtype of series is object
print(a)

# we can add 2 series if match found it will do operation else it will give NaN

'''
DataFrame
1)DataFrame is one of the widely used data structure of pandas
2)DataFrame is similar to SQL tables
    a)2D data structure with heterogeneous data with rows and columns
    b)size is mutable
    c)we can perform arithmetic operations on rows and columns
3)syntax to create DataFrame is 
pandas.DataFrame( data, index, columns, dtype, copy)
    a)data--data takes various forms like ndarray, series, map, lists, dict, constants and
     also another DataFrame.
     b)index--index for rows,default is np.arange(n)
     c)columns--name of the columns,default is np.arange(n)
'''
import pandas as pd
my_data=[10,20,30,40]
pd1=pd.DataFrame(my_data)   # index and columns will be default np.arange(n)
print(pd1)

my_index=['a','b','c','d']
my_columns=('w','x','y','z')
my_data=np.random.randn(4,4)
pd1=pd.DataFrame(my_data,index=my_index,columns=my_columns)    # both index and columns is passed
print(pd1)

# whiel creating DtaFrame from dict the values should be list/dict (more than 1 value)
# if not given then we have to use d.items() and parse the values
# dict keys will become columns

d={'sri':10,'muki':20}
a=pd.DataFrame(d)       # throws error as dict values are scalar

# it can be resolved as below

d={'sri':10,'muki':20}
a=pd.DataFrame(d.items())

d=[{'sri':10,'muki':20}]    # using list of dict also works
a=pd.DataFrame(d)

d={'name':['sri','muki','chethu'],'age':(10,20,30)}
pd.DataFrame(d)

my_data=[{'a':1,'b':2,'c':3},{'a':10,'b':20,'c':30,'d':40}]
a=pd.DataFrame(my_data)
print(a)
'''
output is as below
    a   b   c     d
0   1   2   3   NaN  ----since no d value was there in first dict its filled with NaN
1  10  20  30  40.0
'''

# creating DataFrame using Series
n=[pd.Series(np.arange(5)),pd.Series(np.arange(10))]
p=pd.DataFrame(n)

'''
0	1	2	3	4	5	6	7	8	9
0	0.0	1.0	2.0	3.0	4.0	NaN	NaN	NaN	NaN	NaN
1	0.0	1.0	2.0	3.0	4.0	5.0	6.0	7.0	8.0	9.0
'''

# creating dataframe with series and index

my_data={'one':pd.Series([1,2,3],index=[100,200,300]),'two':pd.Series([4,8,6,5],index=[100,200,300,400])}
p=pd.DataFrame(my_data)

'''
one	two
100	1.0	4
200	2.0	8
300	3.0	6
400	NaN	5
'''

my_data={'one':pd.Series([1,2,3],index=[100,200,300]),'two':pd.Series([4,8,6,5],index=[10,20,30,40])}
p=pd.DataFrame(my_data)

'''
if the index is different from that of first series then 
if data is not found it will be filled with NaN
    one	two
10	NaN	4.0
20	NaN	8.0
30	NaN	6.0
40	NaN	5.0
100	1.0	NaN
200	2.0	NaN
300	3.0	NaN
'''

# column selection

import numpy as np
import pandas as pd
import numpy.random
my_index='A B C D'.split()
my_column='W X Y Z'.split()
a=pd.DataFrame(np.random.randn(4,4),index=my_index,columns=my_column)
print(a)
print(a['W'])           # gives single column
print(a[['W','Y']])      # gives multiple coulum ns

# adding columns to existing dataframe

import numpy as np
import pandas as pd
import numpy.random
my_index='A B C D'.split()
my_column='W X Y Z'.split()
a=pd.DataFrame(np.random.randn(4,4),index=my_index,columns=my_column)
a['V']=pd.Series(np.random.randn(4),index=my_index)    # adding new column with newly created series

a['U']=a['Y']+a['X']     # adding new column with adding 2 existing columns

# deleting dataframe data
'''
drop
1)used to drop dataframe rows or columns
2)syntax is
DataFrame.drop(self, labels=None, axis=0, index=None, 
columns=None, level=None, inplace=False, errors='raise')
3)can be either deleted using columns or rows
4)for deleting columns without column keyword axis should be 1
5)raises keyerror if the key is not found

'''
import numpy as np
import pandas as pd
import numpy.random
my_data=np.arange(12).reshape(4,3)
my_panda=pd.DataFrame(my_data,columns='A B C'.split())
my_panda.drop(2)  # drops at single value index 2
my_panda.drop([1,2,3])  # can also be dropped at multiple indecis by passing indices as list
my_panda.drop(columns=['B','C'])  # dropping columms
my_panda.drop(['B','C'],axis=1)   # if no columns keyword then axis should be 1 to drop using columns
my_panda.drop(1,inplace=True)  #    # if inplace is True then deletion affects original dataframe

'''
DataFrame.pop(str)
1)pops out the passed label
2)takes column name as input and returns deleted column with index
'''
df = pd.DataFrame([('falcon', 'bird', 389.0),
                   ('parrot', 'bird', 24.0),
                   ('lion', 'mammal', 80.5),
                   ('monkey', 'mammal', np.nan)],
                  columns=('name', 'class', 'max_speed'))
df.pop('name')

'''
del
1)del can be used to delete 1 column at a time
2)entire dataframe can be deleted
'''
df = pd.DataFrame([('falcon', 'bird', 389.0),
                   ('parrot', 'bird', 24.0),
                   ('lion', 'mammal', 80.5),
                   ('monkey', 'mammal', np.nan)],
                  columns=('name', 'class', 'max_speed'))

del df['class']   # deletes class column
del df   # deletes entire dataframe

'''
pandas.DataFrame.loc
1)used for accessing rows or columns with labels
2)single value like df.loc['A']
3)multiple values like df.loc[['A','B','C']] can be passed
4)a slice object with labels like df.loc['A":'B'] can also be passed
'''
import numpy as np
import pandas as pd
d={'names':['sri','muki','chethu'],'age':[10,20,30],'place':['XYZ','ABC',"PQR"]}
p=pd.DataFrame(d,index='A B C'.split())
p.loc['A']   # single label can be get
p.loc[['A','B','C']]   # array of labels can be passed
p.loc['A':'C']         # slice object can be passed
p.loc['A':'C':2]        # slice fun with label and step

p.loc['B','age']        # row and column values can be obtained

p.loc['A':'C','age']     # row slicing and column

p.loc[[True, False, True]]   # returns rows with True,boolean length should be same as index length

p.loc[[True, False, True],[False, False, True]]   # we can also choose columns

p.loc[p['age']>10,['age','place']]           # choosing rows on condition and multiple columns

'''
pandas.DataFrame.iloc
1)is used to access the dataframe index or columns using positional index
2)working mechanism everything same as loc function

pandas.DataFrame.at
1)serves same purpose as loc label based indexing
2)can be fetched single value at a time
3)should be used when we want to set or get single value
'''
'''
sorting of a dataframe
DataFrame.sort_values(by,axis=0,ascending=True,inplace=False)
    a)by--single or list of column names to sort by
    b)axis--axis along which sort
    c)ascending--True default if given False then sorts on descending
    d)inplace--if True then affects the original array,default is False
'''
import pandas as pd
import numpy as np
a=pd.DataFrame(np.random.randint(1,20,(4,4)),columns=list('ABCD'))

a.sort_values([0,1],axis=1)  # sorted only index 0 and 1

a.sort_values(list('ABCD'),axis=0)   # SORTS ALL THE VALUES ALONG AXIS 1

# if sorted using multiple coulumn then first column will be sorted and
# index and other columns are adjusted which looks like unsorted
'''
reindexing of dataframe
1)we can re-index and columns of the dataframe
2)if we give new index or columns and if the values not existing then we can fill with NaN
3)syntax is
    DataFrame.reindex(labels,index=None,columns=None,fill_value=nan)
4)if fill_values is given and if the value doesnt exists for the index it fills with 
5)reindex_like is used to set index of one df using other df
'''

import numpy as np
import pandas as pd
a=pd.DataFrame(np.random.randint(1,20,(4,5)),columns=list('ABCDE'))

new_column=list('DEFGH')
a.reindex(columns=new_column)

a.reindex(index=[1,2,3,4],columns=list('ABCDG'),fill_value='missing')   # fill_value can be either int or string

a.reindex_like(b)   # new view of df having index and columns of df b is returned
'''
more on index
set_index
1)set df index using existing columns of df
2)syntax is     DataFrame.set_index(keys,drop=True,append=False,inplace+False)
    a)key--single or tuple of columns which need to be set as new index
    b)drop--if set to True it will delete the column which is set as new index,default is True
    c)append--if set to True will append the column to existing index
    d)inplace--if set to True then changes made will be refleted in original array 
'''
a=pd.DataFrame(np.random.randint(1,20,(5,5)),columns=list('ABCDE'))
a.set_index('B')  # sets column 'B' as new index

a.set_index('B',drop=False)   # sets B as new index and B is not deleted from the columns

a.set_index('B',append=True,drop=False) # B is appended to existing index

a.set_index(['B','D'],append=True,drop=False,inplace=True)  # list of columns can be set as index

'''
			A	B	C	D	E
B	D					
0	16	8	8	16	13	8	15
1	8	10	9	8	8	10	13
2	5	9	13	5	16	9	8
3	3	19	12	3	17	19	16
4	18	13	19	18	15	13	19
'''
'''
DtaFrame.reset_index
1)rest the index to default index (np.arange(n))
2)syntax is
    DataFrame.reset_index(level,drop=False,inplace=False)
    a)level--comes in multi index dataframe
    b)drop--if set to True then index is not added to the columns
'''
a=pd.DataFrame(np.random.randint(1,20,(5,5)),index=list('ABCDE'),columns=list('PQRST'))

a.reset_index()  # index is resetted to default and existing index is added to columns

a.reset_index(drop=True,inplace=True)   # index is dropped from adding to column and original df is affected
'''
	P	Q	R	S	T
0	2	1	7	16	16
1	19	18	16	2	6
2	1	10	12	16	9
3	2	14	9	6	3
4	14	6	9	10	11
'''
'''
iterating over a dataframe
1)iterating over a series is considered same as iterating over array
2)we have 3 methods to iterate over a DataFrame
a)iteritems()--iterating over dataframe with index as key and column values as Series object
b)iterrows()--iterating with column name as key and index values as Series object
c)itertuples()--gives named tuples (dict like tuple objects)  
d)DataFrame.loc--can be used to iterate over index 
'''
import pandas as pd
import numpy as np
a=pd.DataFrame(np.random.randint(1,20,(5,5)),columns=list('PQRST'),index=list('ABCDE'))

for index,column in a.iteritems():   # index(label) as key and column values as series
    if index=='R':
        break          # printing only 2 index
    print(index,column)  # index will be column name and column will be

'''
P               index(labels) as key and column values as series object
A    10
B     1
C     8
D     4
E     7
Name: P, dtype: int32
Q
A    10
B     3
C    12
D     4
E     4
Name: Q, dtype: int32
'''
for i,j in a.iterrows():   # column name as key and index values as series
    if i== 'C':
        break          # printing only 2 index
    print(i, j)   # i is rows name and j is series object

'''
A 
P    10
Q    10
R    10
S    18
T    19
Name: A, dtype: int32
B 
P     1
Q     3
R     7
S     2
T    13
Name: B, dtype: int32
'''
for i in a.itertuples():
    print(i)

'''
Pandas(Index='A', P=10, Q=10, R=10, S=18, T=19)
Pandas(Index='B', P=1, Q=3, R=7, S=2, T=13)
Pandas(Index='C', P=8, Q=12, R=7, S=19, T=6)
Pandas(Index='D', P=4, Q=4, R=2, S=16, T=3)
Pandas(Index='E', P=7, Q=4, R=1, S=17, T=19)
'''
for i in range(len(a)):
    print(a.iloc[i])

'''
filtering dataframe values using conditional rows and columns
'''
import pandas as pd
import numpy as np
a=pd.DataFrame(np.random.randint(1,20,(5,5)),columns=list('PQRST'),index=list('ABCDE'))

a>5  # gives True if the value meets the condition else False

filt=a['P']%2==0    # gives True if the condition is met

a[filt]    # returns df if the condition is met
'''
	P	Q	R	S	T   # observe P values are divided by 2
A	10	5	3	5	4
E	6	15	6	1	8
'''

filt1=(a['T']%4==0) | (a['T']%3==0)   # using multiple conditions (or,and doesnt works use & or |)

a[filt1]
'''
	P	Q	R	S	T    # T is meeting the given condition
A	10	5	3	5	4
B	15	17	12	8	8
D	19	5	2	16	6
E	6	15	6	1	8
'''
a.loc[filt1,'T']  # gives only T value which meets the condition

a.loc[filt1,['T','P']]  # multiple columns can be used

a.loc[-filt1]  # if we put - ve sign then results opposite to condition will be printed

'''
updating rows and columns and data in Dataframe
updating column
1)a.columns=list('abcd') we can update with new column names
2)we can use str functions accordingly to update
'''
import numpy as np
import pandas as pd
a=pd.DataFrame(np.random.randint(20,50,(4,4)),columns=list('ABCD'))

a.columns=list('abcd')   # updating with new column

a.columns=a.columns.str.lower()  # changing to lower case using str function
a.columns=a.columns.str.replace('a','z')
a.columns.str.contains('c')  # returns boolean

# updating rows we can use loc and iloc
import numpy as np
import pandas as pd
a=pd.DataFrame(np.random.randint(20,50,(4,4)),columns=list('ABCD'))

b.loc[2]=[11,14,15,8]  # using loc we can update entire row

b.loc[3,'C']   # updating single row value

b.loc[3,'A':'C']=[10,13,18]  # updating multiple row values

# we can use iloc for same purpose

'''
updating using methods like apply,map,appymap

apply
1)apply a function along the axis of a datframe
2)we can custom functions or lambda or pre-defined functions
3)syntax is
pandas.DataFrame.apply(func,axis=0,raw=False,result_type=none)
    func--function to be applied can be custom or lambda
    axis--axis 0 or 1
    result_type--expand--expands list like object into columns
                reduce--gives series like object than expanding,opposite to expand
                broadcast--result will be broadcast to original type
4)for series object each element of series is applied
5)for Dataframe object each row is applied
6)result type comes only for DataFrame object
'''
s=pd.Series(np.linspace(1,20,10,),index=list('abcdefghij'))
s.apply(np.square)   # applying numpy existing function

# custome function
def f(n):
    if n<15 and n>10:
        return n+10

s.apply(f)

# dataframe apply
import numpy as np
import pandas as pd
a=pd.DataFrame(np.random.randint(5,30,(4,4)),columns=list('ABCD'))

a.apply(np.sqrt)   # applying existing numpy funct

a.apply(np.sum,axis=0) # sum along axis 0(vertically)
a.apply(np.sum,axis=1)  # sum along horizantally

a.apply(lambda x:x*2)  # we can use lambda function also

a.apply(lambda x:[1,2,4],axis=1)   # list will be added as series
'''
A	B	C	D         
0	26	30	38	30
1	12	46	44	32
2	16	36	50	36
3	54	40	32	44

apply(lambda x:[1,2,4],axis=1)

0    [1, 2, 4]
1    [1, 2, 4]
2    [1, 2, 4]
3    [1, 2, 4]
dtype: object

# if we use result_type='expand'
a.apply(lambda x:[1,2,4],axis=1,result_type='expand')

	0	1	2   # list object is expanded as columns
0	1	2	4
1	1	2	4
2	1	2	4
3	1	2	4

a.apply(lambda x:[1,2,4],axis=0,result_type='reduce')
A    [1, 2, 4]
B    [1, 2, 4]
C    [1, 2, 4]
D    [1, 2, 4]
dtype: object

a.apply(lambda x:pd.Series([1,2,3,4]),result_type='broadcast')
A	B	C	D
0	1	1	1	1
1	2	2	2	2
2	3	3	3	3
3	4	4	4	4
'''
# for string data we can use string functions

d={'name':['sri','muki','chethu'],
   'email':['sri@gmail.com','muki@gmail.com','chethu@gmail.com']}
df=pd.DataFrame(d)

def f(x):
    return x.upper()
df['email'].apply(f)   # apply for series

def f(x):
    return x.str.upper()
df.apply(f)     # apply for DataFrame

'''
df['email'].apply(f)

0       SRI@GMAIL.COM
1      MUKI@GMAIL.COM
2    CHETHU@GMAIL.COM
Name: email, dtype: object

df.apply(f)

	name	email
0	SRI	SRI@GMAIL.COM
1	MUKI	MUKI@GMAIL.COM
2	CHETHU	CHETHU@GMAIL.COM

NOTE: STRING FUNCTIONS IN APPLY ON SERIES CAN BE DONE WITHOUT USING STR ACCESSOR
SINCE IN SERIES APPLY IS APPLIED TO EACH ELEMENT OF SERIES

BUT WHILE APPLYING STRING FUNCTIONS TO DATAFRAME WHERE APPLY IS APPLIED ON EACH SERIES OF DATAFRAME
WE SHOULD USE STR ACCESSOR...
'''

'''
pandas.DataFrame.applymap(func)
1)applymap is applicable only for dataframe
2)elementwise operation
'''
d={'name':['sri','muki','chethu'],
   'email':['sri@gmail.com','muki@gmail.com','chethu@gmail.com']}
df=pd.DataFrame(d)

df['name'].apply(lambda x:x.upper())   # elemntwise applied,whereas in apply series wise

import numpy as np
import pandas as pd
a=pd.DataFrame(np.random.randint(10,50,(4,4)),columns=list('ABCD'))

def f(n):
    if n%2==0:
        return n
a.applymap(f)
'''
	A	B	C	D
0	NaN	26.0	20	30.0
1	NaN	NaN	12	14.0
2	NaN	NaN	40	NaN
3	18.0	NaN	20	NaN
'''
'''
map
pandas.series.map(args,na_action=None)
    na_action---if set to 'ignore' then avoids applying function to missing value
    
1)map takes dict,func,series as input
2)applicable for only series
'''
s=pd.Series(np.random.randint(4,20,5))

def g(n):
    if n%2==0:
        return n+10
    else:
        return n
s.map(g)  # applying custom function
'''
0    14
1    22
2    13
3    15
4     7
dtype: int64
'''
s1=pd.Series(['sri','muki','chethu'])
d={'sri':'myself','muki':'friend 1','chethu':'friend 2'}
s1.map(d)  # applying dict
'''
0       sri
1      muki
2    chethu
dtype: object

0      myself
1    friend 1
2    friend 2
dtype: object
'''
s1.map('my name is {} '.format)
'''
0       my name is sri 
1      my name is muki 
2    my name is chethu 
dtype: object
'''
s2=pd.Series([1,2,3,np.nan,4])
s2.map('I am a {}'.format)

s2.map('I am a {}'.format,na_action='ignore')  # if value is NaN then na_action avoids applying func
'''
s2.map('I am a {}'.format)

0    I am a 1.0
1    I am a 2.0
2    I am a 3.0
3    I am a nan
4    I am a 4.0
dtype: object

s2.map('I am a {}'.format,na_action='ignore')

0    I am a 1.0
1    I am a 2.0
2    I am a 3.0
3           NaN
4    I am a 4.0
dtype: object
'''
'''
DIFFERENCE B/W MAP,APPLYMAP AND APPLY

MAP                      |  APPLYMAP                 |  APPLY
1)defined on series      1)defined on Dataframe      1)defined on both series and DF
2)accepts dict,series    2)accepts only function     2)accepts only function
and function as input     as input                    as input

3)map is element wise    3)element wise for          3)works on element wise for DF
 operation for series         dataframe               most suitable for complex operations
 
4)map is meant for      4)is good for elementwise    4)apply is for applying any  
mapping values from       transformations across      function that cannot be vectorised
one domain to another,     multiple rows/co           (i.e cant be converted to series)
so is optimised for      df[['A', 'B', 'C']].applymap  df['sentences'].apply
performance               (str.strip))                  (nltk.sent_tokenize))
 df['A'].map
 ({1:'a', 2:'b', 3:'c'}))
                            
'''
'''
replace
pandas.DataFrame.replace(to_replace=None,values=None,regex=False,limit=None,
inplace=False,method='pad)

    a)to_replace--single value,list or dict or series or regex
    --if to_replace is dict for series {'a':'b','x':'y'} then value should be None
    a will be replaced with b and x will be replaced with y
    
    -->if to_replace is dict for dataframe {'a':'1','x':'y'} then value 1 will be found
    in column a and value y in column x are found and replaced with given values
    
    --> if to_value is nested dict for dataframe {'a':{'b':'c'}} then 
    look in column a for value b and replace it with c
    
    --> if to_replace is None then regesx should be a string  compiled regular expression, 
    or list, dict, ndarray or Series of such elements.
    
    b)value--must be scalar,list,dict
    c)inplace--if set to True then affects the original dataframe
    d)regex--regular expression if set True then to_replace should be string
    e)method
    
1)this is one of the rich function as it has many options
'''
s=pd.Series(np.arange(1,10,2))
s.replace(to_replace=5,value=10)  # replaces 5 with 10
s.replace(to_replace=[3,7,9],value=10) # replaces list of values with scalar 10
s.replace(to_replace=[3,7,9],value=[10,20,30])  # replacing list of values

# dataframe replace

df=pd.DataFrame(np.random.randint(20,50,(4,4)),columns=list('abcd'))
df.replace(to_replace=[26,27,48,49],value=10)  # replaces list of values with scalar 10

df.replace(to_replace=[26,27,48,49],value=[10,20,30,40]) # replacing list of values with another list

d={'first_name':['sri','muki','chethu','abhi'],
   'place':['abc','def','pqr','xyz'],'age':(10,20,30,40)}

df1=pd.DataFrame(d)

df1.replace(to_replace={'place':'def'},value='mno')
# looks in place column and replaces def value with mno
df1.replace(to_replace={'place':'def','first_name':'abhi'},value='mno')

df1.replace(to_replace={'first_name':{'sri':'me','muki':'my friend'}})
# looks in first_name column and replaces sri and muki

# using regex

df1.replace(to_replace='^c[a-z]*',value='friend',regex=True) # starts with c and 0 or more
df1.replace(regex={'^a[a-z]*i$':'friend','sri':'me'}) # regex dict can be passed
'''
first_name	place	age
0	sri	abc	10
1	muki	def	20
2	chethu	pqr	30
3	abhi	xyz	40

df1.replace(regex={'^a[a-z]*i$':'friend','sri':'me'})

	first_name	place	age
0	me	abc	10
1	muki	def	20
2	chethu	pqr	30
3	friend	xyz	40
'''
df1.replace(to_replace={'age':{10:20,40:50}}) # replacing column values
df1['age'].replace({10:20,20:30,30:40})

s = pd.Series([10, 'a', 'a', 'b', 'a'])

s.replace({'a':'none'}) # replaces a with none

s.replace('a')   # method default is pad so previuos values will be padded

'''
s.replace({'a':'none'})

0      10
1    none
2    none
3       b
4    none
dtype: object

s.replace('a')

0    10
1    10
2    10
3     b
4     b
dtype: object
'''
'''
aggregate
pandas.DataFrame.aggregate(func,axis=0,*args,**kwargs)
1)used for Aggregating using one or more operations over the specified axis.
    a)func--functions used for aggregating the data functions,dict,list of func used
    b)axis--0 default ,1 for column wise
    c)*args and **kwargs used for function inputs
'''
df=pd.DataFrame(np.random.randint(10,20,(4,4)),columns=list('ABCD'))
df.aggregate(['min','mean','max'])   # list of functions can be applied
df.aggregate({'A':'max','B':'min'})   # different function applied for different columns

def f(n):
    if n%2==0:
        return n
    else:
        return 0

df.loc[2].agg(func=f,axis=0)  # user defined function is passed
# user defined function can be applied to only induvidual column not for dataframe
# like apply function

import pandas as pd
import numpy as np
d={'name':['Kohli','ABD','Gayle'],'country':['India','Africa','Windies'],'runs':(10000,8000,9000)}
df1=pd.DataFrame(d)

df1.loc[1:2,'name']=['abd','gaylee']  # updating values
def f(n):
    return n.upper()
df1['country'].aggregate(f)  # applying string function
'''
name	country	runs
0	Kohli	India	10000
1	ABD	Africa	8000
2	Gayle	Windies	9000


0      INDIA
1     AFRICA
2    WINDIES
Name: country, dtype: object
'''

'''
rename
pandas.DataFrame.rename(mapper,axis=None,level=None,errors='ignore')
    a)mapper---dict or like function
    b)axis--(0,1) if 0 index will be renamed as per the mapper and if 1 columns
    c)errors--if set to 'raise' throws error if any lable in input dict is missing
    default is ignore which ignores new labels
    
1)used to change either index or columns names
2)labels in input mapper should be unique
'''
import numpy as np
import pandas as pd
d={'first_name':['sri','muki','chethu','abhi'],
   'place':['abc','def','pqr','xyz'],'age':(10,20,30,40)}

df1=pd.DataFrame(d)

df1.rename(mapper={'first_name':'name'},axis=1)   # changes the 'first_name' to 'name'
df1.rename(mapper={2:4,3:9},axis=0)  # changes in axis 0 (default)
'''
	first_name	place	age
0	sri	abc	10
1	muki	def	20
2	chethu	pqr	30
3	abhi	xyz	40

df1.rename(mapper={2:4,3:9},axis=0)

	first_name	place	age
0	sri	abc	10
1	muki	def	20
4	chethu	pqr	30
9	abhi	xyz	40
'''

def f(n):
    return n.upper()
df1.rename(mapper=f,axis=1)  # user defined function is applied to mapper

def g(n):
    if n%2==0:
        return n**2
    else:
        return n**3
df1.rename(mapper=g,axis=0)   # passing function

df1.rename(mapper={2:4,3:9,4:16},axis=0,errors='raise')  # throws error as no label 4 found

'''
rename_axis()

pandas.Dataframe.rename_axis(mapper,axis=0,inplace=False,copy=True)

1)used to rename axis names
'''
import numpy as np
import pandas as pd
d={'first_name':['sri','muki','chethu','abhi'],
   'place':['abc','def','pqr','xyz'],'age':(10,20,30,40)}

df1=pd.DataFrame(d)

df.rename_axis('digit',inplace=True)   # changes index name as digit

df.rename_axis(mapper='details',axis=1,inplace=True)  # changes column name as details
'''
details	first_name	place	age
digit			
0	sri	abc	10
1	muki	def	20
2	chethu	pqr	30
3	abhi	xyz	40
'''
'''
multi index dataframes
1)pandas series and dataframes can have multi level index
2)there are 4 ways to create multi index
'''
'''
pandas.Multiindex.from_arrays(arrays,names)
    a)list or sequence of arrays,each array is one level of index,len(arrays) is no of level
    b)names--names of the levels of the index
'''
arrays=[(1,2,3,4),('red','green','blue','white')]  # all arrays must be of same size
ind1=pd.MultiIndex.from_arrays(arrays)
ind1.levshape # gives length of

s1=pd.Series(np.arange(4),index=ind1)
'''
1  red      0   # first 2 rows are index and 3rd one is data
2  green    1
3  blue     2
4  white    3
dtype: int32
'''
ind2=pd.MultiIndex.from_arrays(arrays,names=['number','colour'])  # gives names for multi index

'''
number  colour
1       red       0
2       green     1
3       blue      2
4       white     3
dtype: int32
'''
array1=(('ABC','XYZ','PQR'),('I','II','III'))
ind2=pd.MultiIndex.from_arrays(array1,names=['index','sub_index'])
s2=pd.Series(np.arange(3),index=ind2)



'''
index  sub_index
ABC    I            0
XYZ    II           1
PQR    III          2
dtype: int32

s2.index.get_level_values(2)  # 2 is the index 
gives Index(['a', 'b', 'c'], dtype='object', name='super_sub_index')
'''
df=pd.DataFrame((np.arange(9)+3).reshape(3,3),index=ind2,columns=list('ABC'))
'''
			A	B	C
index	sub_index	super_sub_index			
ABC	I	a	3	4	5
XYZ	II	b	6	7	8
PQR	III	c	9	10	11
'''
'''
panads.Multiindex.from_product(iterables,names)
    a)iterables--iterables like list,tuple,set
1)used to make multi index from 2 or more iterables
2)size will be product of size of 2 iterables
'''
itr1=('ABC','XYZ','PQR')
itr2=('I','II','III')
ind3=pd.MultiIndex.from_product(iterables=(itr1,itr2),names=('first','second'))

s=pd.Series(np.arange(9),index=ind3)  # size will be 9

df=pd.DataFrame(np.random.randint(10,100,(9,4)),      # dataframe rows should be 9
                index=ind3,columns=('one','two','three','four'))
'''
first  second
ABC    I         0
       II        1
       III       2
XYZ    I         3
       II        4
       III       5
PQR    I         6
       II        7
       III       8
dtype: int32

		    one	two	three four
first second				
ABC	I	    72	15	 76	    15
    II      53	63	 20	    29
    III	    71	86	 75	    62
XYZ	I	    46	43	 85	    43
    II	    34	65	 72	    72
    III	    10	76	 22	    91
PQR	I	    24	58	 89	    23
    II	    21	41	 44	    66
    III	    27	27	 96	    40
'''
itr1=('ABC','XYZ','PQR')
itr2=('I','II','III')
itr3=['i','ii','iii']
ind3=pd.MultiIndex.from_product(iterables=(itr1,itr2,itr3),names=(1,2,3))
df=pd.DataFrame(np.random.randint(10,100,(27,4)),index=ind3)


# complexity increases with no of iterables increased

'''
pandas.multiindex.from_tuples(tuples,names)
1)each tuple is index of row/column
2) no of items in each tuple is the level of index
'''
t=((1,'red'),(2,'blue'),(3,'green'),(4,'white'))
ind=pd.MultiIndex.from_tuples(tuples=t,names=(1,2))
df=pd.DataFrame(np.random.rand(4,4),index=ind)
'''
		0	1	2	3
1	2				
1	red	0.393269	0.913827	0.128977	0.565289
2	blue	0.915843	0.124038	0.128118	0.724498
3	green	0.599014	0.213207	0.541747	0.997855
4	white	0.418027	0.758946	0.940412	0.751921
'''
t=((1,'red','i'),(2,'blue','ii'),(3,'green','iii'),(4,'white','iv'))
ind=pd.MultiIndex.from_tuples(tuples=t,names=(1,2,3))
df=pd.DataFrame(np.random.rand(4,4),index=ind)

'''
pandas.Multiindex.from_frame(df,names)
    a)df--dataframe to convert to multiindex
    b)names--if no names gives column names will be taken as names
'''
df=pd.DataFrame(np.random.rand(4,4),columns=list('ABCD'))
ind=pd.MultiIndex.from_frame(df)
df1=pd.DataFrame(np.random.rand(2,4),columns=list('ABCD'),index=ind)

# no of rows should be same as index rows(old df which converted to multiindex)

'''
MultiIndex([( 0.8157526406563383, 0.7720109677026104),
            (0.44776096231972673, 0.1898541760619028)],
           names=['A', 'B'])
           
		A	B	C	D
A	B				
0.815753	0.772011	0.804531	0.261643	0.096980	0.574027
0.447761	0.189854	0.202097	0.895151	0.502507	0.488651
'''
'''
accessing values of multiindex dataframes
1)we can use same accessing methods of dataframe like loc,iloc
'''
itr1=('ABC','XYZ','PQR')
itr2=('I','II','III')
ind3=pd.MultiIndex.from_product(iterables=(itr1,itr2),names=('first','second'))

df=pd.DataFrame(np.random.randint(10,100,(9,4)),      # dataframe rows should be 9
                index=ind3,columns=('one','two','three','four'))

df['one'] # accessing single column
df[['one','two']]   # accessing multiple columns

df.loc['ABC']   # accessing single row
df.loc[['ABC','PQR']]   # accessing multiple rows

df.loc['ABC','one']  3 accessing single column and row

df.loc['ABC','one':'two']   # using slicing

df.loc[('ABC','II')]   # accessing different level indexes

'''
note:tuples and lists are not treated identically in pandas indexing
    --> tuples used for specifying multi level indexes
    --> list used for multi level values
    
    a list of tuples indexes several complete MultiIndex keys
    whereas a tuple of lists refer to several values within a level
    
df.loc[('ABC','II')]    # means list of tuple gives different level values
df.loc[(['ABC','XYZ'],['I','II'])]   # means tuple of list gives values at same level

in simple terms 
    -->list is used for same level accessing
    -->tuple is used for multi level accessing 
'''
'''
cross section
xs()
1)returns cross section of Series /dataFrame
pandas.DataFrame.xs(key,axis=0,level=None,drop_level=True)
    a)key--label or tuple of labels 
    b)axis--either 0 or 1 ,0 for rows
    c)level--in case of multiindex,indicates which level is used
    d)drop_level--if set to False returns object same level as self,True is default 
'''
i=pd.MultiIndex.from_product([['ABC','XYZ'],['A','B']],names=(1,2))
df=pd.DataFrame(np.random.randint(10,30,(4,4)),index=i,columns=['one','two','three','four'])

df.xs('ABC')   # returns full data of row ABC

df.xs(('ABC','A'))  # accessing multi level labels (observe tuple is used)

df.xs('A',level=2)   # accessing at specified level and index

df.xs(['one','three'],axis=1) # accessing at axis 1

'''
swapping lels of multiindex
panads.DataFrame.swaplevel(i=-2,j=-1)
    a)i--first level to be swapped
    b)j--secxond level to be swapped
'''
i=pd.MultiIndex.from_product([['ABC','XYZ'],['A','B']],names=(1,2))
df=pd.DataFrame(np.random.randint(10,30,(4,4)),index=i,columns=['one','two','three','four'])

df.swaplevel(1,2)    # swapping levels
'''
		one	two	three	four
1	2				
ABC	A	19	21	27	22
B	12	29	13	26
XYZ	A	17	15	11	17
B	23	15	16	28

		one	two	three	four
2	1				
A	ABC	19	21	27	22
B	ABC	12	29	13	26
A	XYZ	17	15	11	17
B	XYZ	23	15	16	28
'''
'''
dealing with missing data
missing data is always a problem in every scenario
1)missing values in python is represented as NaN(not a number)
2)Nan is of type float64
3) NaT is the time equivelant of NaN
4) isna(),notna(),notnull(),notna() used to check

'''
# inserting missing data
s=pd.Series([1,2,3])
s[2]=None   # none can also be used
s[1]=np.nan

# calculations with missing dat
# 1) statistical functions like mean,max will ignore all NaN values
# 2) while summing the data all NA values will be ignored
# 3) if all values are NA then result will be 0
s=pd.Series([1,2,3])
s[2]=None
s.sum()   # ignores all NaN values
s.mean()

# finding missing values
# we can use isna(),notna(),isnull(),notnull()
import pandas as pd
import numpy as np
i=pd.date_range(start='20/03/2020',periods=5,freq='d')
d={'temp':[30,32,24,35,33],'humidity':[12,15,18,20,21],'rainy':['yes','no','yes','yes','no']}
df=pd.DataFrame(d,index=i)
df[2]=np.nan

df.isna()
df.notnull()

'''
filling missing data 
pandas.DataFrame.fillna(value=None,method=None,axis=None,limit=None,inplace=False)
    a)value--value to be filled in NA data can be scalar,dict,series,dataframe
    b)method--method to fill values,default is None
        i)backfill--bfill---filled with next value
        ii)pad--forward fill--ffill--filled with previous value
    c)axis--axis along which values are filled
    d)limit---limit to fill the consecutive NA values if given 2 then 2 consecutive NA values will be filled
            
'''
i=pd.date_range(start='20/03/2020',periods=5,freq='d')
d={'temp':[30,32,24,35,33],'humidity':[12,15,18,20,21],'rainy':['yes','no','yes','yes','no']}
df=pd.DataFrame(d,index=i)

df.iloc[0,2]=np.nan   # setting few values NA
df.iloc[1,0]=np.nan

df.fillna(0)  # filling missed data with 0

df.fillna(0,axis=0) # filling with scalar axis is not going to naffect

df.fillna(value={'temp':0,'rainy':'na'}) # filling with dict values key is column name

df.fillna(value={'temp':0,'rainy':'na'},limit=1) # filled only one time

df.fillna(method='bfill')  # fill with next value

df.fillna(method='ffill')  # fill with previus value

df.fillna(method='ffill',axis=1)  # filling with axis

'''
	temp	humidity	rainy      # in ffill if there are no not null value
2020-03-20	30.0	12.0	NaN     before a NaN value then it wont be filled
2020-03-21	30.0	15.0	no    
2020-03-22	30.0	15.0	no     # in other words in ffill if value starts with 
2020-03-23	35.0	15.0	yes     NaN then it wont be filled
2020-03-24	35.0	15.0	no


	temp	humidity	rainy      # in bfill if there are no not null value
2020-03-20	30.0	12.0	no      after NaN then values wont be filled
2020-03-21	35.0	15.0	no
2020-03-22	35.0	NaN	yes
2020-03-23	35.0	NaN	yes
2020-03-24	NaN	NaN	no

ffill is also called pad
'''
'''
dropping missing data values
pandas.DataFrame.dropna(axis=0,how='any',thresh=None,inplace=False,subset=one)
    a)axis--axis along which values to be dropped
    b)how--if set 'all' then all values should be NA then drop
            default is 'any' if any value is NA then drop
    c)thresh--int,optional if any int is set then that many non_NULL values to be present
    d)subset--labels along other axis to consider for drop
                ex; if dropping rows then this will be the list of columns to include
'''
i=pd.date_range(start='20/03/2020',periods=5,freq='d')
d={'temp':[30,32,24,35,33],'humidity':[12,15,18,20,21],'rainy':['yes','no','yes','yes','no']}
df=pd.DataFrame(d,index=i)

df.dropna()   # drop all rows which contains NaN values
df.dropna(axis=1) # drop all columns which contains NAN values

df.dropna(how='all')  # drops row where all values are NaN values

df.dropna(thresh=2)  # drop any rows which does not have minimum 2 not null values

df.dropna(thresh=3,axis=1)   # drop columns which doesnt have 3 not null values

df.dropna(subset=['temp','rainy'])  # drops row where temp and rainy columns have no NaN values
'''
	temp	humidity	rainy
2020-03-20	30.0	12.0	NaN
2020-03-21	NaN	15.0	no
2020-03-22	NaN	NaN	NaN
2020-03-23	35.0	NaN	yes
2020-03-24	NaN	NaN	no

	temp	humidity	rainy     # df.dropna(subset=['temp','rainy'])
2020-03-23	35.0	NaN	yes       # subset of columns should not have any NaN values

	rainy
2020-03-20	NaN     # df.dropna(thresh=3,axis=1)
2020-03-21	no      3 drops columns which doesnt have minimum 3 not null values
2020-03-22	NaN
2020-03-23	yes
2020-03-24	no
'''
'''
interpolate
1)used to fill values according to different methods
2)syntax is
pandas.DataFrame.interpolate(method='linear',axis=0,limit=None,
limit_direction='forward',limit_area=None,downcast=None,**kwargs)
    a)method--interpolation(insertion of something) techniques
        1)linear--ignores index and treats values are equally spaced
        2)time--works on daily or higher resolution data,index should be timestamp
        3)index--actual numerical values of index will be used,if index is unique
            then works as linear method
        4)nearest--fills with nearest ffill values
        5)polynomial and spline requires order to be specified explicitly
        
    b)axis--axis along which interpolations is done
    c)limit--no of consecutive NaN to be filled
    d)limit_direction--{'forward':'ffill','backward':'bfill'}
    
'''
import numpy as np
import pandas as pd
a=pd.DataFrame(np.random.randint(10,100,(4,4)),columns=list('ABCD'))

a.iloc[[1,3],[1,3]]=None
a.loc[2,'C']=None

a.interpolate()   # default method is linear

a.interpolate(method='time')  # throws error as index should be timestamp
a.interpolate(method='index')  # actual index values is given
a.interpolate(method='nearest') # ffill values will be used

a.interpolate(limit_direction='backward',axis=1)  # fills in bfill way along axis 1
a.interpolate(limit_area='outside')

'''
various methods to add new columns
1)using insert()

pandas.DataFrame.insert(loc,column,value,allow_duplicates=False)
    a)loc--location of the new column ,int value
    b)column--string,name of the column
    c)value--value to be inserted list,tuple,series, or single values
    d)allow_duplicates--if set to True allows duplicate columns
'''
a=pd.DataFrame(np.random.randint(20,50,(4,4)),list('abcd'))
a.insert(loc=2,column='new',value=25,allow_duplicates=False)
# single value will be added along all rows

a.insert(loc=4,column='new1',value=[10,20,27,43],allow_duplicates=False)
# length of elements in df we are adding should be same as new column value

a.insert(loc=3,column='new',value=55,allow_duplicates=True)
# duplicate columns can be added if allow_duplicates is set to True
'''
0	1	2	3
a	37	47	20	29
b	33	35	34	44
c	22	31	20	36
d	39	49	44	28

0	1	new	2	new1	3  # a.insert(loc=4,column='new1',
a	37	47	25	20	10	29    value=[10,20,27,43],allow_duplicates=False)
b	33	35	25	34	20	44
c	22	31	25	20	27	36
d	39	49	25	44	43	28

0	1	new	new	2	new1	3    # duplicate columns can be added if 
a	37	47	25	55	20	10	29     allow_duplicates is set to True
b	33	35	25	55	34	20	44
c	22	31	25	55	20	27	36
d	39	49	25	55	44	43	28
'''
'''
pandas.DataFrame.append(other,ignore_index=False,verify_integrity=False,sort=False)
    a)other--other data to append like dict,series or dataframe or list of them
    b)ignore-index--if set to True ignore the index of both the objects,default is false
    c)verify_integrity--if set to True raises valueerror if duplicate index present
    d)sort--if True sorts the columns of self and others
    
1)other size should be same as self
2)if columns in others that not present in self,will be added to self 
3)rows will be added at the end always
'''
x=pd.DataFrame([[1,2],[4,5]],columns=['b','c'])
y=pd.DataFrame([[6,7],[8,9]],columns=['a','c'])

x.append(y,sort=False)   # appends x to y
x.append(y,ignore_index=True,sort=True)  # index of both self and others ignored

x.append(y,sort=True,verify_integrity=True)  # index should be unique

# while appending dict to dataframe ignore_index should be True

x=pd.DataFrame([[1,2],[4,5]],columns=['b','c'])
d={'b':10,'d':20}
x.append(d,ignore_index=True )
'''
	b	c        x
0	1	2
1	4	5

    a	c         y
0	6	7
1	8	9

	b	c	a     x.append(y,sort=False)
0	1.0	2	NaN   
1	4.0	5	NaN   note that columns are not sorted
0	NaN	7	6.0
1	NaN	9	8.0

    a	b	c
0	NaN	1.0	2      x.append(y,ignore_index=True,sort=True)
1	NaN	4.0	5 
2	6.0	NaN	7       observe that columns are sorted 
3	8.0	NaN	9       index of both self and other ignored

    b	c	d 
0	1.0	2.0	NaN        x.append(d,ignore_index=True )
1	4.0	5.0	NaN        observe if new colum is added 
2	10.0 NaN 20.0           values missed in original df will be filled with NaN
'''
'''
pandas.concat(objs,axis=0,join='outer',keys=none,names=None,ignore-index=False,
        verify_integrity=False,sort=False,level=None,sort=False)
        
        a)objs--sequesnce of series or dataframe 
        b)axis--axis along which the objects be concatened
        c)join--how to handle indexes on other axis,
           -->if used 'outer' means union(combination) of indexes   (default)
           -->if used 'inner' means intersection (common) indexes
        d)keys--sequence passed to form multiple index,default is None
            passed keys as the outermost level
        e)levels--specific levels to use for multiindex,if not passed inferred from keys
        f)ignore_index--if True ignore the index of objects
        g)verify_itegrity--if True then raises error if contains duplicate index
        h)names--names of levels of in case of multiindex
        i)sort--sort indexes of other axis if True (no affect in case join=inner)
                     
'''
x=pd.DataFrame([[1,2],[4,5]],columns=['b','c'])
y=pd.DataFrame([[6,7],[8,9]],columns=['a','c'])

pd.concat(objs=(x,y),sort=False)
'''
	b	c
0	1	2
1	4	5

	a	c
0	6	7
1	8	9


    b	c	a         pd.concat(objs=(x,y),sort=False)
0	1.0	2	NaN
1	4.0	5	NaN       join='outer'  default
0	NaN	7	6.0
1	NaN	9	8.0

	b	c	a	c     pd.concat(objs=(x,y),sort=False,axis=1)
0	1	2	6	7
1	4	5	8	9     along axis 1

        b	c	a
x	0	1.0	2	NaN     pd.concat(objs=(x,y),sort=False,axis=0,keys=['x','y'])
    1	4.0	5	NaN
y	0	NaN	7	6.0      keys are to for multiindex
    1	NaN	9	8.0
    
    x	    y
    b	c	a	c      pd.concat(objs=(x,y),sort=False,keys=['x','y'],axis=1)
0	1	2	6	7       
1	4	5	8	9

		c
x	0	2      # pd.concat(objs=(x,y),sort=False,keys=['x','y'],axis=0,join='inner')
    1	5       
y	0	7      # only common columns are concatened
    1	9
    

'''
# concating series with dataframe
x=pd.DataFrame([[1,2],[4,5]],columns=['b','c'])
pd.concat((x,s))
'''
	b	c	0
0	1.0	2.0	NaN     # pd.concat((x,s))
1	4.0	5.0	NaN
0	NaN	NaN	0.0
1	NaN	NaN	1.0
'''
# adding 3 dataframes
x=pd.DataFrame([[1,2],[4,5]],columns=['b','c'])
y=pd.DataFrame([[6,7],[8,9]],columns=['a','c'])
z=pd.DataFrame([[5,7],[8,1]],columns=('d','c'))
pd.concat(objs=(x,y,z),keys=('x','y','z'),sort=False,axis=0,
          join='inner',names=('one','two'))
'''
	        c
one	two	
x	0	    2        pd.concat(objs=(x,y,z),keys=('x','y','z'),
    1	    5           sort=False,axis=0,join='inner',names=('one','two'))
y	0	    7
    1	    9         observe that names for multiindex is set
z	0	    7
    1	    1
    
we cant concat objcets with different multi index 
ex: we cannot add dataframe and multi index dataframe
'''

'''
group by
1)Group by is used to split the data into groups based on some criteria 
2)group by operation invlves one of the below operations on original object
    a)splitting of object
    b)applying function
    c)combining the results
syntax is
DataFrame.groupby(self, by=None, axis=0, level=None, as_index: bool = True,
sort: bool = True, group_keys: bool = True,squeeze: bool = False, observed: bool = False)

    a)by=label names or list of names used to group
    b)level=int value,level of multindex 
'''
import pandas as pd
import numpy as np
df=pd.read_csv("C:\\Users\\SRI\\Downloads\\student_result1.csv")
g=df.groupby(by='Section')
g.groups  # gives grouped data
g.get_group('Section')  # gives sections data
for i,j in g:   # we can iterate over grouped data
    print(i)e3
    print(j)

g['Percentage'].value_counts().loc('A')   # gives percentage column and A values
'''
	Student ID	Section	Class	Study hrs	Social Media usage hrs	Percentage
0	1001	A	10	2	3	50
1	1002	B	10	6	2	80
2	1003	A	10	3	2	60
3	1004	C	11	0	1	45
4	1005	C	12	5	2	75

'''
# df.value_counts(normalised=True) is used to count the total values of a particular column

df=pd.read_csv("C:\\Users\\SRI\\Desktop\\stackoverflow.csv")
df['ConvertedComp'].value_counts()   # gives selected labels count
g['SocialMedia'].value_counts().loc['India']   # gives socialmedia count for india
# since grouped data of selected label is series we can apply loc and iloc
g['ConvertedComp'].agg(['mean','max']).loc['India']  # we can apply agg functions

g['LanguageWorkedWith'].apply(lambda x:x.str.contains('Python')).sum()
# used to get count of countries used python.....apply is applied to a grouped ser5ies

'''
pandas.Series.str.contains
1)used to match a pattern within a string of series
2) syntax is pandas.Series.str.concat(pat,case=True,flags=0,na=nan,regex=True)
    a)pat--pattern to be matched,either a string or regex
    b)case--if False becomes case insensitive,default is true
    c)falgs---like IGNORECASE
    d)na--filling of missing values,default is NaN
    e)regex--if True pat should be regex otherwise pat will be string literal
'''

d={'name':['sri','muki','chethu'],'place':['abc','xyz','pqr']}
m=pd.DataFrame(d)
m['name'].str.contains(pat='sri')   # str is applicable only for Series not for dataframe
m['name'].str.contains(pat='^[mca][a-z]+',na='NULL',regex=True)  # missing values will be filled with NULL
m['name'].str.contains(pat='sri',na='NULL',regex=False)   # when regex is false then pat will be treated as string


# pd.Series.unique()    # gives unique values of a series

'''
DataFrame.join
1)used to join columns of another dataframe
2)joins on index to index based unless explicitly mentioned any key
3)syntax is   pd.DataFrame.join(other,on=None,how='left',lsuffix='',rsuffix='',sort=False)
    a)other--other dataframe,series or list of dataframes,series name should be set
    b)on--any other columns name to join on,passed column should be present in both dfs
    c)how--how to join two dfs,default is left
        1)'left'--use the index from calling frames(self)
        2)'right'--use the index of right frame
        3)'inner'--use the intesection(common) index of both
        4)'outer'--use union(combined) index of both
    d)lsuffix--if overlapping columns present in both then we can set name for left column
    e)rsuffix--if overlapping columns present in both then we can set name for right column
4)keys that we are join on will be the index not columns(in merge key that we join will be column)
       
'''
a=pd.DataFrame({'key':['key1','key2','key3','key4','key5','key6']
                   ,'A':['A1','A2','A3','A4','A5','A6']})
b=pd.DataFrame({'key':['key1','key2','key3'],'A':['A1','A2','A3'],
                'B':['B1','B2','B3']})
c=pd.DataFrame({'key':['key1','key2','key3','key4'],'C':['C1','C2','C3','C4']})
'''
	key_l	A_l	key_r	A_r	B      # a.join(other=b,lsuffix='_l',rsuffix='_r',how='left')
0	key1	A1	key1	A1	B1     # left frame index is used
1	key2	A2	key2	A2	B2
2	key3	A3	key3	A3	B3
3	key4	A4	NaN	NaN	NaN        # observe missing data is filled with NaN
4	key5	A5	NaN	NaN	NaN
5	key6	A6	NaN	NaN	NaN

	key_l	A_l	key_r	A_r	B        
0	key1	A1	key1	A1	B1    # a.join(other=b,lsuffix='_l',rsuffix='_r',how='inner')
1	key2	A2	key2	A2	B2    3 common index is used
2	key3	A3	key3	A3	B3

	key	A_l	A_r	B
0	key1	A1	A1	B1       # a.join(other=b.set_index('key'),
1	key2	A2	A2	B2            lsuffix='_l',rsuffix='_r',on='key',how='inner')
2	key3	A3	A3	B3   # to join on any column that should be set as index
'''
'''
Dataframe.merge
1)used to merge dataframe or named sries with another dataframe like sql style join
2)if joining on columns then index will be ignored
3) syntax is  pd.DataFrame.merge(right,how='inner',on=None,left_on=None,right_on=None,
    left_index=False,right_index=False,indicator=False,suffixes=['_x','_y'],
    validate=None,sort=False,copy=True)
    
    a)right--other object to mege,dataframe or named series
    b)how--sql style joins,['left','right','inner','outer']
    c)on--column name to be merged,column or list of columns
    d)left-on--column or index names to join on left dataframe(if a column which will
    have few common values and right different column which will have few common values will
    be used for merge) 
    e)right-on--column or index names to join on right dataframe
    d)left_index---use the index of left frame as a key to join
    e)right_index---use the index of right frame as a key to join
    f)indicator--if True gives new column '_merge' which says if the key present in both frames 
    g)suffixes--names if overlapping columns are present,default is ['_x','_y']
    h)sort--to sort merged keys
    i)validate--used to check if duplicate present in keys
        one--unique
        many--duplicates allowed
        1)“one_to_one” or “1:1”: check if merge keys are unique in both left and right datasets.
        2)“one_to_many” or “1:m”: check if merge keys are unique in left dataset.
        3)“many_to_one” or “m:1”: check if merge keys are unique in right dataset.
        4)“many_to_many” or “m:m”: allowed, but does not result in checks.
        
'''
a=pd.DataFrame({'key':['key1','key2','key3','key4'],
                'A':['A1','A2','B1','B2'],'B':['B1','B2','B3','B4']})
b=pd.DataFrame({'key':['key1','key2','key3','key4','key5'],
                'A':['A1','A2','A3','A2','A3'],'B':['B1','B2','B3','A1','A2']})

'''
	key_x	A_x	B_x	key_y	A_y	B_y  #a.merge(right=b,how='outer',left_on='A',right_on='B')
0	key1	A1	B1	key4	A2	A1   # observe common elemnts from left_on='A' and
1	key2	A2	B2	key5	A3	A2        right_on='B' are used
2	key3	B1	B3	key1	A1	B1
3	key4	B2	B4	key2	A2	B2
4	NaN	NaN	NaN	key3	A3	B3

	key_left	A	B_left	key_right	B_right
0	key1	A1	B1	key1	B1
1	key2	A2	B2	key2	B2            # a.merge(right=b,how='outer',on='A'
2	key2	A2	B2	key4	A1               ,suffixes=['_left','_right'])
3	key3	B1	B3	NaN	NaN 
4	key4	B2	B4	NaN	NaN          # providing custom suffixes  
5	NaN	A3	NaN	key3	B3
6	NaN	A3	NaN	key5	A2

key_left	A_left	B_left	key_right	A_right	B_right
0	key1	A1	B1	key1	A1	B1
1	key2	A2	B2	key2	A2	B2   # a.merge(right=b,how='inner',
2	key3	B1	B3	key3	A3	B3      suffixes=['_left','_right'],left_index=True,right_index=True,sort=True)
3	key4	B2	B4	key4	A2	A1    # indexes will be used to merge

	key_x	A_x	B	key_y	A_y	_merge
0	key1	A1	B1	key1	A1	both     # a.merge(right=b,how='outer',on='B',indicator=True)
1	key2	A2	B2	key2	A2	both
2	key3	B1	B3	key3	A3	both     # observe last columns saying values present in frames
3	key4	B2	B4	NaN	NaN	left_only
4	NaN	NaN	A1	key4	A2	right_only
5	NaN	NaN	A2	key5	A3	right_only

append()-->
    1)used for appending new dataframes at the end of df along axis=0 only
    2)equal to (axis=0,join='outer') of the concat
    
concat()-->
    1)gives the flexibility to join based on the axis( all rows or all columns)
    2)multiple dfs can be concatened using only index
    3)multi level index can be added
    
join()-->
1)join is used to join 2 or more dfs based on index of dfs or specific key
2)defaut joins on index
3)default join is left

merge()-->
1)merge is used to to merge olny 2 dfs based on common columns of dfs
2)default join is on columns,can be changed to join on index by setting left_index=True
3)default join is inner

it is always recommended to use merge while merging vertically and concat during horizantally
join is subset of merge

 merge is more versatile than join at the cost of requiring more detailed inputs
'''
'''
pandas datetime
pd.Timestamp(ts-input,tz=none,year=None,month=None,day=None,hour=None,minute=None,
        second=None,millisecond=None,tzinfo=None)
        
        a)ts_input---sting like date input---'2020/16/05:10:20:15'
        rest all can be optional
'''
import pandas as pd
import numpy as np
a=pd.Timestamp(year=2011,month=11,day=20,hour=10,
               minute=20,second=30,microsecond=40,tz='Asia/Kolkata')
print(a)
print(pd.Timestamp.now())   # gives current datetime

b=pd.Timestamp(pd.Timestamp(ts_input='2020/10/20 10:20:30',tz='Asia/Kolkata'))
# date can be inputted as string

'''
pd.to_datetime()
to convert any other formats like int,float,string,list,tuple,series,dfs to datetime
'''
pd.to_datetime(arg='2020/05/20')
pd.to_datetime(arg='20/05/2020',dayfirst=True)   # if we give dayfirst=True then first para will be parsed as day
x=pd.to_datetime(arg='20/05/2020',format="%Y-%m-%d")  # we can pass our format
x.day_name()  # gives day name
x.month_name()  # gives month name   many other methods can be called

# while reading data from either csv,excel or any other data source we can set date parse
d_parcer=lambda x:pd.datetime.strptime(x,'%Y-%m-%d %I-%p')  # passing lambda func for date parsing
df=pd.read_csv("https://raw.githubusercontent.com/CoreyMSchafer/code_snippets/master/Python/Pandas/10-Datetime-Timeseries/ETH_1h.csv",parse_dates=['Date'],date_parser=d_parcer)

'''
dt is similar to str which can be applied to sereis for accessing date methods
'''
df['Date'].dt.day_name()  # day name can be obtained for entire date series
df['Date'].dt.month_name()  # month name of all dates

df['Date'].min()   # minimum date is given
df['Date'].max()   # max (recsent) date is given

'''
Timedelta
1)represents a durations,difference b/w 2 dates or times
2)equivelent to pythons datetime.timedelta

pd.Timedelta(value,unit)
    1)value--value in string,interger
    2)unit--like Hour,day
'''

pd.Timedelta('2 days 3hours',unit='h')
pd.Timedelta(10,unit='d')   # unit is given in days
df['Date'].max()-df['Date'].min()   # diff b/w 2 dates gives timedelta

# filtering date as per the requirement
filter=(df['Date']>='2019') & (df['Date']<'2020')
df[filter]

f=(df['Date']>=pd.to_datetime('2019-05-18')) & (df['Date']<pd.to_datetime('2020-05-18'))
# using time specific filter

# we can set date as index using set_index()
# setting datetime as index has advantage like we can retriev data using partial index

import pandas as pd
import numpy as np
df=pd.read_csv("https://raw.githubusercontent.com/codebasics/py/master/pandas/14_ts_datetimeindex/aapl.csv",index_col='Date',parse_dates=['Date'])
df['2017']   # just by passing partial value we can retrive data for entire month
df['2017-05']  # data for entire month
df['Close'].min() # min of close column

'''
resampling
1)Resampling generates a unique sampling distribution on the basis of the actual data
2)We can apply various frequency to resample our time series data
3)most commonly used time series frequency are
    a)'D'--daily frequency
    b)'W'--weekly frequency
    c)'M'--month end frequency
    d)'SM'--semi-month end frequency (15th and end of month)
    e)'Q'--quarter end frequency
    
resampling Only valid with DatetimeIndex
'''
x=df['Close'].resample(rule='W')    # resampling on weekly frequency
x.max()  # we can find max of resampled data
x.agg(['max','min','mean'])  # multiple func can be applied

%matplotlib inline    # used to plot graphs
x.agg(['max']).plot()

# we can resample entire df
a=df.resample(rule='Q')
a['High'].mean().plot(kind='bar')   # bar graph plot

'''
date_range
1)used to get fixed frequency DatetimeIndex. 
2)syntax is pandas.date_range(start,end=None,period=None,freq=None,tz=None)
    a)start--start date for the date frequency
    b)end=end point
    c)period--no of parts
    d)freq--frequency for parts 
        1)'D'--day 
        2)'2D'--2 day frq
        3)'W'--week
        4)'M'--montly
        5)'SM'--semi month(15th and end date)
        6)'Q'--quarterly 
        7)'B'--business day excludes weekend
    e)tz--time zone like 'US/central','Asia/tokyo','Asia/Kolkata'
    
among start,end,period,fre  any 3 are mandatory
'''
import pandas as pd
import numpy as np
a=pd.date_range(start='05/05/2020',end='15/05/2020',freq='D')

a=pd.date_range(start='05/05/2020',periods=5,freq='D')

a=pd.date_range(start='05/05/2020',periods=5,freq='M')

a=pd.date_range(start='05/05/2020',periods=5,freq='Q',tz='Asia/Kolkata')

# to change the date frequency we can use pandas.asfreq

b.asfreq('W',method='pad')   # method can be bfill,ffill or pad


# setting our own Custombusinessday

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

a=CustomBusinessDay(calendar=USFederalHolidayCalendar())  # passing us calender
x=pd.date_range(start='2020/12/22',periods=5,freq=a)

'''
reading and writing data to different sources
csv file
1)csv(comma separated value) is one of the imp file system
pandas.read_csv() is used to read data from a csv file
    1)filepath_or_buffer--URL or file path 
    2)sep--stands for separator default is ','(comma)
    3)index_col--makes passed column as index instead of 0,1,2,3
    4)header--makes passed row as header
    5)use_cols--use only passed columns (list of strings) to form a dfs
    6)squeeze--if set to True only one column is passed and gives a pandas series
    7)skiprows--skips passed rows in dfs
    8)nrows--read only specified no of rows
    9)na_values--list of values,values which should be converted to NaN(some values 
    will be like 'not available','na,we can submit dictionary)
    10)date_parcer--if any column contains date in other forms like string to be parced
    11)parse_dates--date parcing method
     
writing to csv
    
pandas.to_csv()
    1)filepath--path of new file to be written
    2)sep--sep to be set,default is ','
    3)index--if set to True,index will not be written ignored
    4)columns--only these columns will be written to new file
    5)header--if False ignores header
'''
import pandas as pd
import numpy as np
a=pd.read_csv(filepath_or_buffer="C:\\Users\\SRI\\Downloads\\student.csv")
a=pd.read_csv(filepath_or_buffer="C:\\Users\\SRI\\Downloads\\student.csv",header=2)
# making our own row as header,row index should be passed(index starts from 0)
a=pd.read_csv(filepath_or_buffer="C:\\Users\\SRI\\Downloads\\student.csv",index_col='Class')
# making our own column as index
a=pd.read_csv(filepath_or_buffer="C:\\Users\\SRI\\Downloads\\student.csv",
              usecols=['Student ID','Section','Class'])
# using columns which are required
a=pd.read_csv(filepath_or_buffer="C:\\Users\\SRI\\Downloads\\student.csv",skiprows=2)
# skipping rows,2 rows will be skipped
a=pd.read_csv(filepath_or_buffer="C:\\Users\\SRI\\Downloads\\student.csv",nrows=2)
# only 2 rows will be used
d={'Class':['na'],'Study hrs':'not','Percentage':'abc'}
a=pd.read_csv(filepath_or_buffer="C:\\Users\\SRI\\Downloads\\student.csv",na_values=d)
# missing na_values of different forms are to be replaced with NaN with dictinory

'''
reading excel
pandas.read_excel()
    1)io--url or file path
    2)sheet_name=0--sheet name can be int or string like 1,2 or sheet1
    3)index_col--column name to be used as index if list of columns is passed then
            combined into multi index
    4)usecols
    5)converters--takes dict,key will be int or string and value will be function 
            which takes a cell value and transforms to other value
    
writing data to excel
pandas.to_excel()
    1)excel_writer--filepath
    2)sheet_name
    3)index--if True then ignores columns
    4)header--if True then ignores header
    5)startrow--row after which data will be written
    6)startcol--col after which data is written
'''
pd.read_excel(io="C:\\Users\\SRI\\Desktop\\book1.xlsx")  # use xlsx extension
pd.read_excel(io="C:\\Users\\SRI\\Desktop\\book1.xlsx",sheet_name='Sheet2')
# sheet name is case sensitive,can also be int starting from 0

def conv(cell):
    if (cell == 'not') or (cell == 'na'):
        return None
    return cell

pd.read_excel(io="C:\\Users\\SRI\\Desktop\\book1.xlsx",
              converters={'Study hrs': conv, 'Class': conv})

# converter is used to transform values

# writing data to excel

x.to_excel(excel_writer='book.xlsx')
x.to_excel(excel_writer='book.xlsx',header=True,index=True)  # index and header will be ignored
x.to_excel(excel_writer='book.xlsx',startcol=2,startrow=1) # after 2 row starts writing

# writing multiple dfs data to single excel
with pd.ExcelWriter('booook.xlsx') as writer:
    a.to_excel(writer,sheet_name='sheet_a')
    b.to_excel(writer,sheet_name='sheet_b')   # we can add as many as dfs

'''
json data
1)a data storing format similar to xml
2)stands for javascript object notation
3)much similar to python dictionary
4)bool values are true and false
5)null is none value
6)file extension is .json
7)used for transmitting data over network connections
8)it is used for transfer of data b/w browser and server

json methods
json.load(f)--Load json data from file(or file like object)
json.loads(s)--Load json data from string
json.dump(j,f)--write json object to file
json.dumps(j)--output json object as string
'''
import json
address={}
address['sri']={
    'name':'sri',
    'address':'Bangalore',
    'phone':98745782
}

address['chethu']={
    'name':'chethu',
    'address':'Vijipura',
    'phone':9872457621
}
s=json.dumps(address)   # outputing json as string
with open('new.txt','w') as f:   # writing the same to a file
    f.write(s)

a='{"sri": {"name": "sri", "address": "Bangalore", "phone": 98745782},' \
  ' "chethu": {"name": "chethu", "address": "Vijipura", "phone": 9872457621}}'

x=json.loads(a)   # loading str data to json  observe no much diffrence but only type changes
x["sri"]   # since json is like dict we can aceess values based on key

for i in x["sri"]:
    print(i)

with open('new.json') as f:
    new_python_object=json.load(f)    # converting json data to python object

with open('new.json') as f:
    new_json=json.dump(f)    # converting python object data to json

'''
writing data to json
pandas.DataFrame.to_json()
    1)path_or_buf---file path to write data
    2)orient--Indication of expected JSON string format.
        allowed values are: {‘split’, ‘records’, ‘index’, ‘columns’, ‘values’, ‘table’}.
        split--dict like
        records--list like
    3)lines--if set to True then splits the list by delimiter comma and orient should be records
        
reading data from json
pandas.read_json()
1)path_or_buf---file path to write data
    2)orient--Indication of expected JSON string format.
        allowed values are: {‘split’, ‘records’, ‘index’, ‘columns’, ‘values’, ‘table’}.
        split--dict like
        records--list like
'''
df = pd.DataFrame([['a', 'b'], ['c', 'd']],
                  index=['row 1', 'row 2'],
                  columns=['col 1', 'col 2'])
df.to_json(path_or_buf='new_json.json',orient='records')  # in list like form
df.to_json(path_or_buf='new_json.json',orient='records',lines=True)
# lines if set to True will split the list of dict separated by comma to new line

# reading data from json
pd.read_json(path_or_buf='new_json.json',orient='records')
pd.read_json(path_or_buf='new_json.json',orient='records',lines=True)
# splits line by line

'''
reading data from SQL database
pandas.read_sql_table()
    1)table_name---str,table name
    2)con---sqlalchemy connection engine
    3)index_col---column to be used as index
    4)columns--list of columns to be selecteed from sql table
    5)chunksize--If specified, returns an iterator where chunksize is the number of rows to include in each chunk.
    
pandas.read_sql()
same as above parameters

pandas.read_sql_query()
same as above parameters

writing data back to sql from dataframe
pandas.DataFrame.to_sql()
    1)name--str,name of sql table
    2)if_exists--exists{‘fail’, ‘replace’, ‘append’}, default ‘fail’
        tells how to behave if table already exists
        1)'fail'--raise error if table already exists
        2)'replace'--drop a table before before inserting new values
        3)'append'--insert new values into existing table
        
    3)index--if False ignores index,else writes index as column
        

'''
import pandas as pd
import pymysql  # to connect to mysql and python objects
from sqlalchemy import create_engine
engine=create_engine('mysql+pymysql://root:Sri@6363131614@localhost:3306/ig_clone')
# engine = create_engine("mysql+pymysql://USER:PASSWORD@HOST:PORT/DBNAME")
df=pd.read_sql_table("likes",con=engine,columns=['photo_id','user_id'])
print(df)

# to read sql query
query="select * from users limit 10"
df=pd.read_sql_query(sql=query,con=engine)
print(df)

query="select * from users "
df=pd.read_sql_query(sql=query,con=engine,chunksize=10)
for i in df:   # chunksize gives the mentioned no of parts and
    print(i)     # gives generator which can be iterated over

# writing data to sql

x=df.to_sql(name='new',index=False,con=engine)  # if index is false then ignores index
print(x)

x=df.to_sql(name='new',index=True,con=engine,if_exists='append')
# appen will add values to existing table