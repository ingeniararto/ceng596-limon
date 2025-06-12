import os
import numpy as np
import pandas as pd
import pyterrier as pt

from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import CrossEncoder

from limon import settings
from django.shortcuts import render

if not pt.java.started():
    pt.init()

scaler = MinMaxScaler()

class CrossEncoderReranker(pt.Transformer):
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2', top_k=100, text_field='text'):
        print("*"*200)
        self.model = CrossEncoder(model_name)
        self.top_k = top_k
        self.text_field = text_field

    def transform(self, res):
        reranked_dfs = []
        for qid, group in res.groupby('qid'):
            query = group['query'].iloc[0]
            doc_nos = group["docno"].tolist()
            docs = [docno_text_dict.get(doc_no, "") for doc_no in doc_nos]
            group = group.copy()
            group[self.text_field] = docs

            top_k = min(self.top_k, len(docs))
            limited_group = group.head(top_k)
            pairs = [(query, doc) for doc in limited_group[self.text_field]]

            scores = self.model.predict(pairs)

            full_scores = np.zeros(len(group))
            full_scores[:len(scores)] = scores

            group['crossencoder_score'] = full_scores
            group['score_norm'] = scaler.fit_transform(group[['score']])
            group['crossencoder_score_norm'] = scaler.fit_transform(full_scores.reshape(-1, 1))

            group['combined_score'] = 0.4 * group['crossencoder_score_norm'] + 0.6 * group['score_norm']

            reranked = group.sort_values(['combined_score', 'rank'], ascending=[False, True])
            reranked.reset_index(drop=True, inplace=True)
            reranked['old_rank'] = reranked['rank']
            reranked['rank'] = reranked.index

            reranked_dfs.append(reranked)

        final = pd.concat(reranked_dfs) if reranked_dfs else pd.DataFrame()
        return final


raw_index_path = os.path.join(settings.INDICES_DIR, 'vaswani_positional_raw')
positional_index_path = os.path.join(settings.INDICES_DIR, 'vaswani_positional')

dataset = pt.get_dataset('irds:vaswani')
docno_text_dict = {doc['docno']: doc['text'] for doc in dataset.get_corpus_iter()}

raw_index = pt.IndexFactory.of(raw_index_path)
positional_index = pt.IndexFactory.of(positional_index_path)

bm25 = pt.terrier.Retriever(positional_index, wmodel="BM25", controls={"qe": "off", "proximity": "on"})
bm25_raw = pt.terrier.Retriever(raw_index, wmodel="BM25")
prf = pt.rewrite.RM3(index_like=positional_index, fb_terms=20, fb_docs=3)
bert = bm25 >> CrossEncoderReranker()

def index(request):
    query = request.GET.get('query', '').strip()
    search_type = request.GET.get('search_type', 'BM25-Raw')

    if search_type == "BM25":
        model = bm25
    elif search_type == "PRF":
        model = bm25 >> prf >> bm25
    elif search_type == "BERT":
        model = bert
    else:
        model = bm25_raw

    returned_docs = model.search(query=query).head(10)

    results = []
    for idx, row in returned_docs.iterrows():
        results.append({
            'doc_id': row["docno"],
            'content': docno_text_dict[row["docno"]],
            'rank': row["rank"] + 1
        })

    context = {
        'query': query,
        'search_type': search_type,
        'results': results
    }
    return render(request, "index.html", context)
