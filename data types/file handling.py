'''
We can use python for handling files i.e to read,write and other options
1)We use open () function in Python to open a file in read or write mode
2) syntax is open(filename,mode),there are 4 mode we can open files
----“ r “, for reading.(default if no mode provided)
----“ w “, for writing.
----“ a “, for appending.
----“ r+ “, for both reading and writing
'''

file=open('text' , 'r')
for i in file:
   print(i)              # printing contents of file

print(file.readline())   # to read line by line if we readline() again it will read next line

# another way to read file is to use read()
file_1=open('file1','r')
print(file_1.read())       # to get entire contents of file

print(file_1.read(10))       # to get first 10 characters of contents of file


'''
Creating a file using write() mode
once file is written we must close
The close() command terminates all the resources in use and frees the system of this particular program.
once written contents will be copied and to read contents file has to be opened in read mode
'''
file_2=open('file3','w')
file_2.write('They are also working ')
file_2.close()   # to close file

# if the file is not available to write it creates us a new file
# if we write on a file all the things present in file will be lost

'''
to append contents to other file
'''
f3=open('file3','a')               # already existing contents wont be last
f3.write('\n appending to file3 ')

# program to copy contents from one file to another

f1=open('file1','r')  # reads data
f2=open('file2','w')   # writes f1 data onto f2
for each in f1:
   f2.write(each)

# to reverse content of file onto another file
f3=open('file1','r')
content=f1.read()
print(content)
f4=open('file4','w')
f4.write(content[::-1])
f4.close()

'''
1)Using  along with with() function to deal with files a good option since it has auto clean up for files
2)once we use the file we must close it else it will lead to an error,with() function has auto closing
'''

with open('f','r') as f:  # here f is the file object
   print(f.read(5))   # everything has to be inside the with block

   f.seek(0)           # to reset the next print to desired point

   print(f.read(10))   # usually this prints where it left off in first print..but seek() resets it to 0

#print(f.read())          # throws error since statement outside the suit

# to read and write

with open('file1', 'r') as rf:
   with open('file4', 'w') as wf:
      data=rf.read()
      for i in data:
         new_data=wf.write(i)
      print(new_data)

