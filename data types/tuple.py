#tuple is similar to list but are immutable.
# if creating tuple with single item use comma at the end
#to reverse tuple1[::-1]

tuple1 = (0 ,1, 2, 3)
print(tuple1[::-1])     #reverses tuple as (3, 2, 1, 0)
a=[1,2,3]
b=list(x+5 for x in a)
print(b)

t=(10,20,30,40,50,60)    # updating tuple converting to list
l=list(t)
print(l)
for i in l:
    if i%4==0 or i%15==0:
        l.remove(i)
print(l)
t=tuple(l)
print('tuple after update',t)

t=(10,20,30)+(40,)      # adding element to tuple
print(t)

