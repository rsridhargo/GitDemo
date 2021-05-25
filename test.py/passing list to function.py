'''

def count(l1):
    even=0
    odd=0
    for i in l1:
        if i%2==0:
            even+=1
        else:
            odd+=1
    return even,odd

l1=(10,20,30,40,21,45,789)

e,o=count(l1)
print('even no are :{} and odd no are:{} '.format(e,o))

'''

def count(lst):
    index=0
    for i in lst:
        if len(i)>5:
            index+=1
    return index
lst=list(input('enter the list'))

a=count(lst)
print(a)
