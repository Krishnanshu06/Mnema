import os
import re
from pymongo import MongoClient
from spacy.lang.en import English
import torch
from sentence_transformers import SentenceTransformer , util
from datetime import datetime
from mistralai import Mistral
import numpy as np
from private import MongoClientID,MongoCollectionName,MongoDBName,MistralApiKey

model = "mistral-large-latest"
client = Mistral(api_key=MistralApiKey)


nlp = English()
nlp.add_pipe("sentencizer")
model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                            device = 'cuda')



class MongoDBfunctions:

    def __init__(self,MongoClientID,MongoDBName,MongoCollectionName):

        self.client = MongoClient(MongoClientID)
        self.db = self.client[MongoDBName]  # Database name
        self.collection = self.db[MongoCollectionName]    # Collection name
    
    

    def add_data(self,user_id, data_entry):
        try:
            self.collection.update_one(
                {"_id": user_id},
                {"$push": {"data": data_entry}},
                upsert=True
            )
        except Exception as e:
            return e
        else:
            return 201



    def add_embeddingList(self,user_id, embedding_list):
        try:
            self.collection.update_one(
                {"_id": user_id},
                {"$push": {"embeddingList": embedding_list}},
                upsert=True
            )
        
        except Exception as e:
            return e;
        else:
            return 201;



    def read_data(self,user_id):
        
        user = self.collection.find_one({"_id": user_id})
        
    
        if user:
            return user.get("data", []) 
        else:
            print("No user found with ID:", user_id)
            return None



    def read_embeddings(self,user_id):

        user = self.collection.find_one({"_id": user_id})
        
        if user:
            return user.get("embeddingList", [])  # Returns stored embeddings
        else:
            print("No user found with ID:", user_id)
            return None



def createDataDict(text: str,mood: int) -> dict:

    date = datetime.today().strftime('%d/%m/%Y')
    charCount = len(data)
    data = text.replace("\n", " ").strip()
    JournalData = {"date":date,"charCount":charCount,"mood":mood,"data":data}

    return JournalData


def getSentences(longStr: str) -> list:
    sentences = list(nlp(longStr).sents)
    return sentences


def makeChunks(journalData: dict , maxChar: int=1000) -> list[dict]:
   
    journals = journalData
    output = []

    if journals['charCount'] >= maxChar:
        sentences = getSentences(journals["data"])
        combined_sentences = []
        sum_char_count = 0
        for sentence in sentences:
            sentence = str(sentence)            # sabse chutiya bug
            if sum_char_count+len(sentence) < maxChar:
                sum_char_count+=len(sentence)
                combined_sentences.append(str(sentence))
            else:
                output.append({"date":journals["date"],"charCount":sum_char_count,"mood":journals['mood'],"data":(f'On {journals["date"]} ' + " ".join(combined_sentences))})
                combined_sentences = []
                combined_sentences.append(str(sentence))
                sum_char_count = len(sentence)
            
        if sum_char_count>0:
            output.append({"date":journals["date"],"charCount":sum_char_count,"mood":journals['mood'],"data":(f'On {journals["date"]} ' + " ".join(combined_sentences))})

    else:
        output.append({"date":journals["date"],"charCount":journals["charCount"],"mood":journals['mood'],"data":f'On {journals["date"]} ' + journals["data"]})
 

    return output



def ChunksToDB(Chunks,mongoObj,userID):

    output = []
    for chunk in Chunks:
        embedding = model.encode(chunk['data'],
                                            batch_size=16,
                                            convert_to_tensor=False)

        output.append(embedding)
        mongoObj.add_data(userID,chunk)
            
    

    mongoObj.add_embedding(userID,output)



def GetTopChunks(query,userID,mongoObj,noResults=5):
    
    queryEncodings = model.encode(query,convert_to_tensor=True)
    
    strList = []
    
    EmbeddingList = mongoObj.read_embeddings(userID)[0]
    chunks = mongoObj.read_data(userID)[0]

    embeddingTensors = torch.stack([torch.tensor(embedding) for embedding in EmbeddingList])

    dot_scores = util.dot_score(a=queryEncodings,b=embeddingTensors)
    top_results = torch.topk(dot_scores,k=noResults)
    indexes = (top_results[1][0]).tolist()

    for idx in indexes:
        strList.append(chunks[idx]['data'])
    
    combinedStr = ' \n'.join(strList)
    return combinedStr



def GetDatedChunks(date_list,userID,mongoObj):

    user = mongoObj.collection.find_one({"_id": userID})

    if not user:
        print("No user found with ID:", userID)
        return None

    filtered_data = [entry[0]['data'] for entry in user.get("data", []) if entry[0]['date'] in date_list]
    strData = ' \n'.join(filtered_data)
    return strData



def getGeneration(query,dataRetrived,client):

    chat_response = client.chat.complete(
        model= model,
        messages = [
            {
                "role": "user",
                "content": f"From the below given data about me, answer {query}. the data provided is- {dataRetrived}",
            },
        ]
    )
    output = chat_response.choices[0].message.content
    return output





def TakeJorunals(text,mood,userID):  #feed a journal into the database

    dataDict = createDataDict(text,mood)
    chunks= makeChunks(dataDict)
    ChunksToDB(chunks,userID)


def GenerateFromQuery(query,userID):  #get the embedding from the database and generate result

    topResults = GetTopChunks(query,userID)
    return getGeneration(query,topResults,client)

def GenerateFromDate(dateRange,query,userID):

    topResults = GetDatedChunks(dateRange,userID)
    return getGeneration(query,topResults,client)






mongo = MongoDBfunctions(MongoClientID,MongoDBName,MongoCollectionName)








