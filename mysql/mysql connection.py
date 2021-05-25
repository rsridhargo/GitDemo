'''import mysql.connector
mydb=mysql.connector.connect(host='localhost',user='root',passwd='Sri@6363131614',database='sridhar')
my_cursor=mydb.cursor()
sql='select * from new'
values=[('sri',10),('chethu',30),('abhi',35)]
my_cursor.execute(sql)
results=my_cursor.fetchall()
for i in results:
    print(i)
'''

# fetching existing databases
import mysql.connector
mydb=mysql.connector.connect(host='localhost',user='root',password='Sri@6363131614')
my_cursor=mydb.cursor()
sql='show databases'
my_cursor.execute(sql)
for i in my_cursor:      # for loop for fetching each db avaialble
    print(i)


#my_cursor.fetchall-----it will print all the columns as 'tuples' in a list

import mysql.connector
mydb=mysql.connector.connect(host='localhost',user='root',
                             password='Sri@6363131614')
my_cursor=mydb.cursor()
sql='create database new'
my_cursor.execute(sql)
mydb.commit()     # always required for changes made to reflect

# its always recommended to escape values in any query,to prevent sql injections
# which is common sql web hacking to destroy or misuse of database
# we can Escape values by using the placholder %s method

sql='update student set name=%s where id=%s'
val=('mukesh',6)
my_cursor.execute(sql,val)
mydb.commit()


'''
ENUM is used to choose the values from a list of values

ex: create table new
(
id int primary key,
gender enum('M','F','O'),
name varchar(100)
);

here gender is choosen only b/w the given 3 values
'''
