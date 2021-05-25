'''
Regular expressions
1)imported from re module
2)applicable to only string type
3) RE are language independent
4) Re is used for
a) validations(like email and mobile number validations)
b) pattern matching (searching for particular match)
c) to develop translators like compiler,assembler,interpreter
'''
'''
metacharacters
1) []--A set of characters --"[a-m]" 
2) .--Any character (except newline character)-- "he..o" 
3) ^--Starts with --"^hello"
4) $--ends with --'ends$'
5) *--zero or more occurrences (end+1 is considered in case of *)
6) +--one or more occurrences
7) ?--either zero or one occurrences
8) {m} --Exactly the specified number of occurrence  (ex:{3} 3 occurrences)
9) {m,n} --minimum of m and maximum of n number of occurrence  (ex:{1,3} 1 or 2 or 3 occurrences)

note: we can club above meta characters
ex1: a{2}a* --minimum 2 a should be there and followed by 0 or more occurrences of a
ex2: a+b+--one or more a and b

sets

1)[abc]---returns match if any one of either a or b or c is found
2)[^abc]---returns any matches other than a or b or c
3)[a-z]---returns matches of lower case letters from a to z
4)[A-Z]---returns matches of upper case letters from A to Z
5)[a-zA-Z]---returns any upper or lower alphabets
6)[0-9]----returns matches for digits 
7)[a-zA-Z0-9]---returns for any alphanumeric
8)[^a-zA-Z0-9]---returns for any non alphanumeric
9) \s--returns a match if any whitespaces found
10) \S--returns a match if any non whitespaces found
11) \d-- return match of digits  [0-9]
12) \D returns match of non digits   [^0-9]
13) \w--returns alphanumeric  [a-zA-Z0-9]
14) \W--returns other than alpahnumeric  [^a-zA-Z0-9]

if we add ^ at the beginning it excludes the followed words from search

'''
'''
Function compile()
1)Regular expressions are compiled into pattern objects, 
which have methods for various operations such as searching 
for pattern matches or performing string substitutions.
2)finditer() function is used to create target string object
3)object created is called Regex object
'''
import re
count=0
pattern=re.compile('python')    # pattern object to be searched
matcher=pattern.finditer('i am learning python')   # target string
for match in matcher:
    count+=1
    print('match is avaialble at index: ',match.start())
print('number of times pattern found is ',count)


'''
match()
1) checks if the sting begins with the pattern if begins returns match object
2) if not begins with pattern return None
3)fullmatch() searches if the string is completely matches to the pattern
'''
import re
s='baa5'
m=re.match('aa',s)   # return None as string not begins
fm=re.fullmatch('baa5',s,re.IGNORECASE)   # return fullmatch object
'''
search
1)The search() function searches the string for a match, and returns a Match object if there is a match
2)If there is more than one match, only the first occurrence of the match will be returned. 
If no matches are found, the value None is returned.
3)match object returns the 
a)span()--a tuple containing start and end positions of the match
b) string -- returns the string passed into the search function
c)group() -- returns the part of the string where there was a match.
'''
import re

s='search for term1 but another term is not available because only term1 is available and only term1 will be found'
match=(re.search('ai',s))
print(match)
print(match.start())   # return start length of search word
print(match.end())     # return end length of search word
print(match.span())    #returns range of search word
print(match.string)
print(match.group())

# example

patterns=['term1','term2']
s='search for term1 but another term is not available'

for pattern in patterns:
    if re.search(pattern,s):
        print('match found\n',pattern)
    else:
        print('match not found\n',pattern)

'''
split()
1)This will split the string based on the given pattern and results the resulting list
2)if max size is given splits at most that time else default zero will be taken as max split

syntax is

re.split(pattern, string, maxsplit=0, flags=0)

3)flags is not mandatory,still we can pass that parameter eg: flags = re.IGNORECASE, In this split, 
case will be ignored.
'''
s='abc def ghi jkl mno pqr stu'
print('after splitting is: ',re.split('\W',s))

'''
findall()
1) returns matching substring as a list
2)if no match found returns empty list
'''
import re
s='ags334gg4gshd35hdgaw6agsa'
m=re.findall('\d',s)
print(m)

'''
sub()
1) substitutes match with another string
2)subn() is same as sub() but only difference is it gives output as tuple and count 
syntax is 
re.sub(pattern, replacement, string, count=0, flags=0)
'''
import re
s='affa33hh5fj7dh5hdh4hd2hd9a'
m=re.sub('\d','#',s,2)   # if count is not given it will by default replaces throught  string
print(m)

import re
s='affa33hh5fj7dh5hdh4hd2hd9a'
m=re.subn('\d','#',s,4)          # output is ('affa##hh#fj#dh#hdh#hd#hd#a', 8)
print(m)


s='sd..sssdded...sdsedssdd...ssdd..dd'
pattern1='sd*'
pattern2='sd+'
pattern3='sd?'
pattern4='sd{2,3}'
pattern5='s[sd]+'

print(re.findall(pattern1,s))
print(re.findall(pattern2,s))
print(re.findall(pattern3,s))
print(re.findall(pattern5,s))


'''
ex: write a regular expression to identify all keywords as per the below rules
1)only alphabets and digits and # allowed
2)first letter should be lowe letter from a to k   --[a-k]
3)second letter should be a number divisible by 3   --[0369]
4)length of a keyword should be at least 2

[a-k][0369][a-zA-Z-0-9#]* is the regular expression 
'''
s = input('Enter the identifier')
m = re.fullmatch('[a-k][0369][a-zA-Z-0-9#]*', s)   # fullmatch is used to match complete string
if m != None:
    print(s, 'is a valid identifier')

else:
    print(s, 'is not a valid identifier')

# regular expression to check valid mobile number

s=(input('enter a number'))
if re.fullmatch('[9876][0-9]{9}',s)!=None:
    print('valid mobile number')
else:
    print('invalid mobile number')

# to find if the entered word is correct we use fullmatch (which matches entire string)
# to check if the entered word starts with particular string we use match
# to check numbers exists and others we can use either findall or search
# to replacement we can use sub and if we want count we can use subn()

# program to read a file and collect all avid mobile numbers and copy them to another file

import re
file_name=input('Enter the file to be opened\n')
file_handler=open(file_name,'r')
output_f=open('output','w')
for line in file_handler:
    r=re.findall('[6789][0-9]{9}',line)
    for i in r:
        if i not in output_f:
            output_f.write(i+'\n')

# RE are used for web scraping(to find the content from url and get data

import re,urllib
import urllib.request
u=urllib.request.urlopen('https://www.redbus.in/info/contactus')  # getting contact numbers from redbus.com
text=u.read()
numbers=re.findall('[0-9]{3}[- ][0-9]{7}',str(text),re.IGNORECASE)
for num in numbers:
    print(num+'\n')

# program to find valid mail id

import re
s=input('enter mail id\n')
m=re.fullmatch('\w[a-zA-Z0-9_.]*@[a-z]+[.][a-z]+',s)
print(m)
if m!=None:
    print('valid mail id')
else:
    print('invalid mail id')

