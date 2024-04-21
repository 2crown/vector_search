import os
from dotenv import load_dotenv
import pymongo
import requests
import datasets



load_dotenv()
MONGO_URI = os.getenv('MONGO_URI')
HUGGING_FACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')
EMBEDDING_URI = os.getenv('EMBEDDING_URI')




client = pymongo.MongoClient(MONGO_URI)
db = client.sample_mflix
collection = db.movies


##items = collection.find().limit(5)
##for item in items:
##    print(item)


hf_token = HUGGING_FACE_TOKEN
embedding_url = EMBEDDING_URI

def generate_embedding(text: str) -> list[float]:

  response = requests.post(
    embedding_url,
    headers={"Authorization": f"Bearer {hf_token}"},
    json={"inputs": text})

  if response.status_code != 200:
    raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")

  return response.json()

#for doc in collection.find({'plot':{"$exists": True}}).limit(50):
#    doc['plot_embedding_hf'] = generate_embedding(doc['plot'])
#    collection.replace_one({'_id': doc['_id']}, doc)

query = "love and war in an alien land"

results = collection.aggregate([
  {"$vectorSearch": {
    "queryVector": generate_embedding(query),
    "path": "plot_embedding_hf",
    "numCandidates": 100,
    "limit": 4,
    "index": "PlotSemanticSearch",
      }}
])

for document in results:
  print(f'Movie Name: {document["title"]},\nMovie Plot: {document["plot"]}\n')

  