'''
for i in range(5):

    for j in range(5-i):
        print('# ', end='')
    print()

'''

def is_leap(year):
    return year%4==0 and (year%100!=0 or year%400==0)

r=is_leap(2020)
print(r)





