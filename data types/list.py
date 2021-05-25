'''
l1=[10,80.20,30,41.15,54.14]
print(len(l1))
print(min(l1))
print(max(l1))
print(sorted(l1))
print(sorted(l1,reverse=True))
print(reversed(l1))
for i in reversed(l1):
    print(i)
'''
import math


# Complete the solve function below.
def solve(meal_cost, tip_percent, tax_percent):


    meal_cost = float(input('enter number'))

    tip_percent = int(input('enter number'))

    tax_percent = int(input('enter number'))

    toatlCost=round(meal_cost*(1+(tip_percent/100)+(tax_percent/100)))
    print(toatlCost)

solve(100,50,50)

