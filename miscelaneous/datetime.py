'''
1)we can import python module datetime for date,time
2)we can use dir(datetime) to get more info about the package
3) for date giving 3 arguments (year,month and day) mandatory
4)

from datetime import time,datetime,date
d=date(year=2020,month=2,day=10)
print(d)
print(d.year)
print(d.month)
print(d.day)
today_date=date.today()
print(today_date)           # to get today date we have to use today() method


time class
1)if we dont pass arguments(hours,minutes,seconds and microseconds) default values 0 will be taken
t=time()  # 00:00:00:0000
print(t)
t1=time(10,20,25,15741)

datetime class

1)to get both date and time
2) date parameters (year,month,day) mandatory
3)datetime.now() is used for current time and date

dt=datetime(1994,8,7,12,14,85,5754)
print(dt)

to get time in dd/mm/yyyy,Hrs:Min:Sec we use strptime() method
from  datetime import date,datetime,time
dt=datetime.now()
d1=(dt.strftime('%d/%m/%y'))
d2=(dt.strftime('%H:%M:%S'))
d3=(dt.strftime('%d/%m%/y, %H:%M:%S'))
print(d1)
print(d2)
print(d3)


'''