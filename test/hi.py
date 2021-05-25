import requests
r=requests.get("https://unsplash.com/photos/oaCD9WYdNlU")
with open('new.png','wb') as f:
    f.write(r.content)