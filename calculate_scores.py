#!/bin/python3

# BM25 versions paper: https://link.springer.com/content/pdf/10.1007/978-3-030-45442-5_4.pdf
# Elasticsearch 7.9.1 documentation: https://elasticsearch-py.readthedocs.io/en/7.9.1/index.html

import ir_datasets  # python 3.8+ interpreter required
from elasticsearch import Elasticsearch  # pip install elasticsearch==7.9.1
import numpy as np
import math
import pandas as pd


# Name of the indexes
index_name = "ct-index"
query_index_name = "ct-query-index"
# Name of the dataset
dataset_name = "clinicaltrials/2021/trec-ct-2021"
# Instantiate a client instance
client = Elasticsearch("http://localhost:9200")
# Download queries
dataset = ir_datasets.load(dataset_name)


################################################################################


def format_query(query_text):
    query_formatted = {
        "size": 10000,
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "content": query_text
                            }
                        }
                    ],
                    "minimum_should_match": 1,
                    "boost": 1.0
                }
            },
            "sort": [
                {
                    "_score": {
                        "order": "desc"
                    }
                } 
            ]
        }
    return query_formatted


################################################################################


# Example function to create an index of a database using elasticsearch
def index_documents():
    print("Indexing documents") 
    # Create index
    mappings = {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 1
            }
        },
        "mappings": {
            "properties": {
                "content": {
                    "type": "text",
                    "fielddata": True,
                    "term_vector": "with_positions_offsets_payloads",
                    "store": True,
                    "analyzer": "whitespace"
                }
            }
        }
    }
    resp = client.indices.create(index=index_name, body=mappings, ignore=400)
    
    # Index dataset
    for doc in dataset.docs_iter():
        # Get the 'best' content from document
        content = doc.detailed_description
        if not len(doc.detailed_description):
            content = doc.summary
        if not len(doc.summary):
            content = doc.eligibility
        if not len(doc.eligibility):
            content = doc.title
        
        # Index mapping
        body = {
            "content": content
        }
        # Call the index API
        resp = client.index(index=index_name, id=doc.doc_id, body=body)


################################################################################


def index_queries():
    print("Indexing queries")
    # Create index
    mappings = {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 1
            }
        },
        "mappings": {
            "properties": {
                "content": {
                    "type": "text",
                    "fielddata": True,
                    "term_vector": "with_positions_offsets_payloads",
                    "store": True,
                    "analyzer": "whitespace"
                }
            }
        }
    }
    resp = client.indices.create(index=query_index_name, body=mappings, ignore=400)
    
    # Index queries
    for query in dataset.queries_iter():
        # Index mapping
        body = {
            "content": query.text
        }
        # Call the index API
        resp = client.index(index=query_index_name, id=query.query_id, body=body)


################################################################################


# Standard search from Elasticsearch
def standard_search():
    # Generate relevance dictionary
    relevance = {}
    for qrel in dataset.qrels_iter():
        if qrel.query_id not in relevance:
            relevance[qrel.query_id] = {qrel.doc_id: qrel.relevance}
        else:
            relevance[qrel.query_id].update({qrel.doc_id: qrel.relevance})

    # Calculate evaluations
    ndcg5_dict = {}
    ndcg10_dict = {}
    precision10_dict = {}
    reciprocal_dict = {}

    for query in dataset.queries_iter():
        # Call the search API
        resp = client.search(index=index_name, body=format_query(query.text))

        # Calculate NDCG@5
        dcg = 0
        idcg = 0
        
        # Ideally ranked list
        relevant_list = sorted(relevance[query.query_id].items(), key=lambda kv: kv[1])
        relevant_list.reverse()
        
        for i in range(5):
            doc_id = resp['hits']['hits'][i]['_id']
            dcg += relevance.get(query.query_id, 0).get(doc_id, 0)/math.log2(i+2)
        
        for i in range(min(len(relevant_list), 5)):
            idcg += relevant_list[i][1]/math.log2(i+2)

        ndcg = dcg / idcg
        ndcg5_dict[query.query_id] = ndcg

        # Calculate NDCG@10
        dcg = 0
        idcg = 0

        for i in range(10):
            doc_id = resp['hits']['hits'][i]['_id']
            dcg += relevance.get(query.query_id, 0).get(doc_id, 0)/math.log2(i+2)

        for i in range(min(len(relevant_list), 10)):
            idcg += relevant_list[i][1]/math.log2(i+2)

        ndcg = dcg / idcg
        ndcg10_dict[query.query_id] = ndcg

        # Calculate precision@10
        relevant_count = 0
        for i in range(10):
            doc_id = resp['hits']['hits'][i]['_id']
            if relevance.get(query.query_id, 0).get(doc_id, 0) > 0:
                relevant_count += 1

        precision = relevant_count/10
        precision10_dict[query.query_id] = precision

        # Calculate reciprocal rank
        rr = -1
        for i in range(len(resp['hits']['hits'])):
            if relevance.get(query.query_id, 0).get(resp['hits']['hits'][i]['_id'], 0) > 0:
                rr = i
                break
        
        rr += 1 #we start counting at 1
        rr = 1/rr
        reciprocal_dict[query.query_id] = rr

    return [ndcg5_dict, ndcg10_dict, precision10_dict, reciprocal_dict]


################################################################################


# BM25L implementation
def bm25l():
    print("Running BM25L")
    # Set global variables
    N = 0  # total documents
    for doc in dataset.docs_iter():
        N += 1
    k_1 = 0.9  # k_1: BM25 parameter, 0.9 used in comprison paper
    b = 0.4  # b: BM25 parameter, 0.4 used in comparison paper
    delta = 0.5  # delta boost constant: BM25L paper suggests 0.5
    
    # Average document length and,
    # Document frequency per term in dataset
    total_document_length = 0
    df_dict = {}
    for doc in dataset.docs_iter():
        resp = client.termvectors(index_name, id=doc.doc_id, term_statistics=True)
        terms = resp["term_vectors"]["content"]["terms"]
        total_document_length += len(list(terms.keys()))
        for t in terms:
            df_dict[t] = terms[t]["doc_freq"]
        
    L_avg = total_document_length / N  # average document length

    f = open("bm25l.csv", "w")
    f.write("query_id,document_id,score\n")
    count = 0
    for query in dataset.queries_iter():
        # Get terms from query
        resp = client.termvectors(query_index_name, id=query.query_id)
        query_terms = list(resp["term_vectors"]["content"]["terms"].keys())
        
        # Retrieve 10k documents using search
        resp = client.search(index=index_name, body=format_query(query.text))
        hitlist = resp["hits"]["hits"]
        document_id_list = [x["_id"] for x in hitlist]
        
        # Go through query results
        for doc_id in document_id_list:
            # Get term vectors for document
            resp = client.termvectors(index_name, id=doc_id, term_statistics=True)
            term_vector = resp["term_vectors"]["content"]["terms"]
            
            # Perform scoring calculation
            score = 0
            for term in query_terms:
                # Variables
                df_t = df_dict.get(term, 0)
                tf_td = term_vector[term]["term_freq"] if term in term_vector else 0
                L_d = len(list(term_vector.keys()))  # document length
                c_td = tf_td / (1 - b + b * (L_d / L_avg))
                
                # Parts of calculation
                log_frac = (N + 1) / (df_t + 0.5)
                frac = ((k_1 + 1) * (c_td + delta)) / (k_1 + c_td + delta)
                
                # Combine parts
                score += math.log(log_frac) * frac
            # save score for query, doc pair in file
            f.write("{},{},{}\n".format(query.query_id, doc_id, score))
    f.close()


################################################################################


# BM25+ implementation
def bm25plus():
    print("Running BM25+")
    # Set global variables
    N = 0  # total documents
    for doc in dataset.docs_iter():
        N += 1
    k_1 = 0.9  # k_1: BM25 parameter, 0.9 used in comprison paper
    b = 0.4  # b: BM25 parameter, 0.4 used in comparison paper
    delta = 0.5  # delta boost constant: BM25L paper suggests 0.5
    
    # Average document length and,
    # Document frequency per term in dataset
    total_document_length = 0
    df_dict = {}
    for doc in dataset.docs_iter():
        resp = client.termvectors(index_name, id=doc.doc_id, term_statistics=True)
        terms = resp["term_vectors"]["content"]["terms"]
        total_document_length += len(list(terms.keys()))
        for t in terms:
            df_dict[t] = terms[t]["doc_freq"]
        
    L_avg = total_document_length / N  # average document length

    f = open("bm25plus.csv", "w")
    f.write("query_id,document_id,score\n")
    count = 0
    for query in dataset.queries_iter():
        # Get terms from query
        resp = client.termvectors(query_index_name, id=query.query_id)
        query_terms = list(resp["term_vectors"]["content"]["terms"].keys())
        
        # Retrieve 10k documents using search
        resp = client.search(index=index_name, body=format_query(query.text))
        hitlist = resp["hits"]["hits"]
        document_id_list = [x["_id"] for x in hitlist]
        
        # Go through query results
        for doc_id in document_id_list:
            # Get term vectors for document
            resp = client.termvectors(index_name, id=doc_id, term_statistics=True)
            term_vector = resp["term_vectors"]["content"]["terms"]
            
            # Perform scoring calculation
            score = 0
            for term in query_terms:
                # Variables
                df_t = df_dict.get(term, 0)
                tf_td = term_vector[term]["term_freq"] if term in term_vector else 0
                L_d = len(list(term_vector.keys()))  # document length
                
                # Parts of calculation
                log_frac = (N + 1) / df_t if df_t > 0 else (N + 1)
                frac = ((k_1 + 1) * tf_td) / (k_1 * (1 - b + b * (L_d / L_avg)) + tf_td)
                
                # Combine parts
                score += math.log(log_frac) * (frac + delta)
            # save score for query, doc pair in file
            f.write("{},{},{}\n".format(query.query_id, doc_id, score))
    f.close()


################################################################################


# Performance measures from our BM25 implementations
def bm25_stats(filename):
    ndcg5_dict = {}
    ndcg10_dict = {}
    precision10_dict = {}
    reciprocal_dict = {}

    # Generate relevance dictionary
    relevance = {}
    for qrel in dataset.qrels_iter():
        if qrel.query_id not in relevance:
            relevance[qrel.query_id] = {qrel.doc_id: qrel.relevance}
        else:
            relevance[qrel.query_id].update({qrel.doc_id: qrel.relevance})


    # Calculate score for every query ID
    df = pd.read_csv(filename)
    queries = df['query_id'].unique()
    for query_id in queries:

        # Get scores for this query
        query_df = df[df['query_id'] == query_id]

        # Sort by calculated BM-score, ordered descending
        sorted_query_df = query_df.sort_values('score', ascending=False)

        # Determine the ideal list from labels
        relevant_list = sorted(relevance[str(query_id)].items(), key=lambda kv: kv[1])
        relevant_list.reverse()

        # Calculate NDCG@5
        top5_df = sorted_query_df.head(5)
        dcg = 0
        idcg = 0

        i = 0
        for _, row in top5_df.iterrows():
            doc_id = row['document_id']
            dcg += relevance.get(str(query_id), 0).get(str(doc_id), 0)/math.log2(i+2)
            i += 1
        
        for i in range(min(len(relevant_list), 5)):
            idcg += relevant_list[i][1]/math.log2(i+2)
            
        ndcg = dcg / idcg
        ndcg5_dict[query_id] = ndcg

        # Calculate NDCG@10
        top10_df = sorted_query_df.head(10)
        dcg = 0
        idcg = 0

        i = 0
        for _, row in top10_df.iterrows():
            doc_id = row['document_id']
            dcg += relevance.get(str(query_id), 0).get(str(doc_id), 0)/math.log2(i+2)
            i += 1
        
        for i in range(min(len(relevant_list), 10)):
            idcg += relevant_list[i][1]/math.log2(i+2)
            
        ndcg = dcg / idcg
        ndcg10_dict[query_id] = ndcg

        # Calculate precision@10
        top10_df = sorted_query_df.head(10)

        relevant_count = 0
        for _, row in top10_df.iterrows():
            doc_id = row['document_id']
            if relevance.get(str(query_id), 0).get(str(doc_id), 0) > 0:
                relevant_count += 1

        precision = relevant_count/10
        precision10_dict[query_id] = precision

        # Calculate reciprocal rank
        rr = len(sorted_query_df) #last possible position
        i = 1
        for _, row in sorted_query_df.iterrows():
            doc_id = row['document_id']
            if relevance.get(str(query_id), 0).get(str(doc_id), 0) > 0:
                rr = i
                break
            i += 1
        
        rr = 1/rr
        reciprocal_dict[query_id] = rr
    
    return [ndcg5_dict, ndcg10_dict, precision10_dict, reciprocal_dict]

################################################################################


def main():
    # Indexing can be disabled if index exists (it takes a while)
    index_documents()
    index_queries()
    bm25l()
    bm25plus()
    
    results_list = []
    results_list.append(["default"] + standard_search())
    results_list.append(["bm25l"] + bm25_stats("bm25l.csv"))
    results_list.append(["bm25plus"] + bm25_stats("bm25plus.csv"))

    with open("results.csv", "w") as f:
        f.write("name,ndcg5,ndcg10,precision10,reciprocal\n")
        for name, ndcg5_dict, ndcg10_dict, precision10_dict, reciprocal_dict in results_list:
            ndcg5_mean = np.mean(np.array(list(ndcg5_dict.values())))
            ndcg10_mean = np.mean(np.array(list(ndcg10_dict.values())))
            precision10_mean = np.mean(np.array(list(precision10_dict.values())))
            reciprocal_mean = np.mean(np.array(list(reciprocal_dict.values())))
            f.write("{},{},{},{},{}\n".format(name, ndcg5_mean, ndcg10_mean, precision10_mean, reciprocal_mean))

if __name__ == "__main__":
    main()

