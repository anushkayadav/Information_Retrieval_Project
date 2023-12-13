import json
import pandas as pd
from pprint import pprint
from tqdm import tqdm
import faiss
import numpy as np
import pickle
import torch
from transformers import AutoTokenizer, AutoModel

test_data=pd.read_pickle('preprompt_100_s.pkl')
test_data=test_data[['id', 'input', 'profile', 'output', 'input_keywords', 'input_summary',]]

tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
model = AutoModel.from_pretrained('facebook/contriever')

# Mean pooling
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def get_contriver_embs(sentences, batch_size=32):
    all_embeddings = []

    # Process sentences in batches
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch_sentences = sentences[i:i+batch_size]
        inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():  # Ensure no gradients are calculated
            outputs = model(**inputs)
        batch_embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        all_embeddings.append(batch_embeddings.cpu().detach().numpy())  # Detach and convert to numpy

    # Concatenate all batch embeddings
    embeddings = np.vstack(all_embeddings)
    return embeddings



def get_relevant_docs_cont(query, doc_list):
  texts = [doc['abstract'] for doc in doc_list]
  doc_embeddings = get_contriver_embs(texts)

  # Create a FAISS index
  dimension = doc_embeddings.shape[1]
  index = faiss.IndexFlatL2(dimension)
  index.add(doc_embeddings)

  # Function to search documents
  def search_documents(query, k=3):
      query_embedding = get_contriver_embs([query])
      distances, indices = index.search(query_embedding, k)
      return [doc_list[i] for i in indices[0]]

  top_docs = search_documents(query)
  return top_docs



cont_res=[]
for row in tqdm(range(test_data.shape[0])):
  query = test_data.iloc[row]['input']
  top_ret=get_relevant_docs_cont(query, test_data.iloc[row]['profile'])
  cont_res.append(top_ret)

  if(row%10==0):
    # Save to a pickle file
    with open("cont/data_cont"+str(row)+".pkl", "wb") as file:
      pickle.dump(cont_res, file)


test_data['contriever']=cont_res

test_data.to_pickle('cont_res_scholar_100.pkl')
