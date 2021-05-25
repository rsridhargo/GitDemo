'''
1)Logging is a means of tracking events that happen when some software runs
2)it is very important to have logging file in case of any server crash,logging file is very useful in
finding the root cause analysis and debugging
3)it also provides statistical info about the application like number of requests and etc
4)6 levels of message we have in python
a) CRITICAL--50
b) ERROR--40
c) WARNING--30 (if no level is set by default warning will be set)
d) INFO--20
e) DEBUG--10
f) NOTSET--0

NOTSET<DEBUG<INFO<WARNING<ERROR<CRITICAL

we can set the values instead of names
only higher levels and warning levels will be displayed
default filemode will be append
'''
import logging

logging.basicConfig(filename='log.txt',format='%(asctime)s %(message)s',filemode='w',level=logging.INFO)
logging.debug('in debug level')
logging.info('in info level')
logging.warning('in warning level')
logging.error('in error level')
logging.critical('in critical level')

# if filename is not given then it will print the output onto console

# logging program to catch exception msg

import logging

logging.basicConfig(filename='log.txt', format='%(asctime)s %(message)s', filemode='a', level=logging.INFO,
                    datefmt='%d/%m/%Y %H:%M:%S %p')

logging.info('new request came')
try:
    x = int(input('enter first number'))
    y = int(input('enter second number'))
    print(x / y)

except ZeroDivisionError as msg:
    print('cant divide by zero')
    logging.exception(msg)

except ValueError as msg:
    print('provide valid int ')
    logging.exception(msg)

logging.info('new request ended')