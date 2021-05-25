'''
To generate OTP using python code we use below 2 functions
1)random.random(): This function returns any random number between 0 to 1.
2)math.floor(): It returns floor of any floating number to a integer value.
'''
'''
# approach 1 (string array)

import random as r
import math as m

def otpgen():

    # all alpah numerics stored as string,if we want just number we can give from 0 to 1
    s='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # declare empty string
    otp=''
    length=len(s)
    for i in range(6):   # length of otp can be changed using range
        otp+=s[m.floor(r.random()*length)]
    return otp

if __name__=='__main__':
    res=otpgen()
    print(res)
    
'''
# using string cancatenation

def otp_gen(s):
    length=len(s)
    otp=''
    for i in range(0,length,2):   # to get otp of squares of even index digits
        otp+=str(int(s[i])**2)
    return otp

if __name__=='__main__':
    otpgen=otp_gen('4365188')
    print(otpgen[:6])            # to get 4 number otp we give 4 index


