import requests
import json

url = "https://api.imagga.com/v2/tags"

querystring = {"image_url":"https://qodebrisbane.com/wp-content/uploads/2019/07/This-is-not-a-person-2-1.jpeg"}

headers = {
    'accept': "application/json",
    'authorization': "Basic ur api key"
    }

response = requests.request("GET", url, headers=headers, params=querystring)
data = json.loads(response.text.encode("ascii"))

for i in range(6):
    tag = data["result"]["tags"][i]["tag"]["en"]
    print(tag)
