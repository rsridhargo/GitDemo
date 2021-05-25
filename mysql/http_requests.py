'''
1)python requests is python library used for getting requests from web pages
2)HTTP is a set of protocols designed to enable communication between
clients and servers. It works as a request-response protocol between a client and server.
3)A web browser may be the client, and an application on a computer
that hosts a web site may be the server.
4)requests library provides with basic authentication and streaming downloads from web pages

    payload--info of the response is called payload
    response.contents--to view the response in bytes we use contents
    response.text--to view response in text

    to convert bytes content to string we can use
    response.encoding='utf-8'

    to get the content in dictionary form we use response.json()

5)response.headers
    a)gives useful info such as content type,size,total characters and date and more
    b)headers is dictionary which can be accessed using keys
            response.headers['Content-Type']   # key is not case sensitive for accessing

6)to customize our url request we can use query string parameters and passed onto GET
 and it will be added to our url

'''
import requests
p=('whitefield')   # custom parameters will be appended to url
response=requests.get("https://www.google.com/search?q=bangalore",params=p)
response.encoding='utf-8'   # converting bytes to string
print(response.content)   # to view the contents in bytes
print(response.text)     # to view the content in text
print(response.json())    # to view the data in jason form
print((response.headers))   # to view the headers ,returns the dictionary
print(response.url)         # to see url

'''
custom headers
1)we can pass headers to our custom 
2)All header values must be a string, bytestring, or unicode
3)Furthermore, Requests does not change its behavior at all based on which
 custom headers are specified
'''
import requests
h={'Content-Type':'sridhar'}
response=requests.get("https://www.google.com/search?q=bangalore",params=p,headers=h)
print(response.headers['Content-Type'])   # since headers gives dict can be accessed using key

'''
post request
1)post() method sends the data to specified url
2)post() is used when you want to send some to the server

syntax is
            requests.post(url,data={key:value},json={key:value},args)
            url--url to request(required)
            data--dict,list,tuple or files to send to specified url (optinal)
            json--json object to send to specified url (optional)
            args--zero or more  named arguments like timeout,auth 
            
3)in post() rge data is added to message body
'''
import requests
pload = {'username':'Olivia','password':'123','place':'xyz'}
r = requests.post('https://www.programcreek.com/python',data = pload)
print(r.text)


'''
1)When the method is GET, all form data is encoded into the URL, 
appended to the action URL as query string parameters. 
With POST, form data appears within the message body of the HTTP request.

2)In GET method, the parameter data is limited to what we can stuff into 
the request line (URL). Safest to use less than 2K of parameters, 
some servers handle up to 64K.No such problem in POST method 
since we send data in message body of the HTTP request, not the URL.

3)Only ASCII characters are allowed for data to be sent in GET method.
There is no such restriction in POST method.

4)GET is less secure compared to POST because data sent is part of the URL. 
So, GET method should not be used when sending passwords or other sensitive information.
'''

# copying image to other file by downloading from internet

import requests
r=requests.get("https://images.unsplash.com/photo-1537526358797-e732f762d6af?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjExMDk0fQ&auto=format&fit=crop&w=750&q=80")
with open('new.png','wb') as f:
    f.write(r.content)

print(r.url)

'''
authorisation

1)we can pass login credenstials to server,if the valid creden then 
 authotisation will be true

2)authentication refers to giving user permission to access data from server 
since everyone cant access the data from server

if authentication is success 

{
  "authenticated": true, 
  "user": "sri"
}

if not no message will be displayed but gives 401 (unauthorized) status code
'''
import requests
# authorization header is passed in auth header as tuple
r=requests.get("http://httpbin.org/basic-auth/sri/1234567",auth=('sri','1234567'))
print(r.text)

#when we pass authentication username and password in tuple,requests is applying creden
# using HTTPBasicAuth so we can explicitly pass the credens to HTTPBasicAuth

import requests
from requests.auth import HTTPBasicAuth

response=requests.get("http://httpbin.org/",auth=HTTPBasicAuth('sri','12345'))
print(response)

'''
Digest Authentication
Another very popular form of HTTP Authentication is Digest Authentication,
 and Requests supports this out of the box as well:
 
'''
import requests
from requests.auth import HTTPDigestAuth
r=requests.get("http://httpbin.org/",auth=HTTPDigestAuth('sri','12345'),timeout=5)
# we can add timeout argument also just to ask the server to give result after 3 sec
print(r)