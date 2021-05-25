'''
class Cylinder:
    pi=3.14
    def __init__(self,height,radius):
        self.height=height
        self.radius=radius

    def vol(self):
        return (Cylinder.pi*self.height*(self.radius*self.radius))

    def surface_area(self):
        return ((2*Cylinder.pi*self.radius)*(self.radius+self.height))

    # to print object in printable form
    def __str__(self):
        return '{} and {}'.format(self.vol(),self.surface_area())


c=Cylinder(2,3)
c.height
c.surface_area()
c.vol()
print(c)

from math import sqrt
class Line():
    def __init__(self,c1,c2):
        self.c1=c1
        self.c2=c2

    def distance(self):
        x1,y1=self.c1
        x2,y2 = self.c2
        a=sqrt(x2-x1)
        b=sqrt(y2-y1)
        return sqrt(a+b)


    def slope(self):
        x1, y1 = self.c1
        x2, y2 = self.c2
        return (y2-y1)/(x2-x1)
    def __str__(self):
            return '{} and {}'.format(self.distance(), self.slope())
cr1=(3,2)
cr2=(8,10)
l=Line(cr1,cr2)
print(l)

'''

class Account:
    def __init__(self,owner,balance=0):
        self.owner=owner
        self.balance=balance

    def deposit(self,amount):
        if amount>10000:
            print('daily limit exceeded,please enter lower amount')
        else:
            self.balance+=amount
            print('Deposit Accepted')
            return self.balance


    def withdraw(self,amount):
        if amount<=self.balance:
            self.balance-=amount   # we need not use self for amount since its instance variable for withdrawal
            print('Withdraw Accepted')
            return 'remaining balance is {}'.format(self.balance)
        else:
            print('Sorry entered funds Unavailable!')

    def __str__(self):
        return 'Account owner is {} and balance avaialble is {} INR'.format(self.owner,self.balance)

acct1 = Account('Jose',100)
print(acct1)
acct1.deposit(1000)
print(acct1)
print(acct1.withdraw(1000))








