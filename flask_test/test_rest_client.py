import _ssl
import json
import ssl
import urllib.request

import requests

gcontext = ssl.SSLContext(_ssl.PROTOCOL_TLSv1)
r = urllib.request.urlopen('https://127.0.0.1:5000/', context=gcontext)

print(r.read())

payload = {'username': 'bob', 'email': 'bob@bob.com'}
json_data = json.dumps(payload).encode('utf8')
req = urllib.request.Request(url='https://127.0.0.1:5000/', data=json_data,method='PUT')
with urllib.request.urlopen(req, context=gcontext) as f:
    pass

print(f.status)
print(f.reason)

