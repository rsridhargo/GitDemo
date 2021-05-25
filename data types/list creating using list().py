l=eval(input('enter the number'))   #eval used for conversion of
print(l)                            # string to any other type based on user input
print(type(l))


#CREATING LIST USING list()

l=list (range(1,10))        #using range inside list()
print('list using range is =',l)

l=list('sri')          #using string inside list()
print('list using string is =',l)

l=list('abc' + 'def')
print('list of string concatenation is =',l)


string1='my name is sridhar'
splitList=string1.split()
print('list using split is =',splitList)
#syntax str.split(separator, maxsplit)


'''
separator : The is a delimiter. The string splits at this specified separator.
If is not provided then any white space is a separator.

maxsplit : It is a number, which tells us to split the string into
maximum of provided number of times. If it is not provided then there is no limit.
'''

s1='my,name,is,sri'
l=s1.split(',',4)   #split separated by , and maximum of 4 times
print('split list with , is =',l)
#word = 'CatBatSatFatOr'
#l=word.split('',4)
