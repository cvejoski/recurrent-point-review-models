import pickle
from pprint import pprint

import numpy as np
import pandas as pd
import scipy
import spacy
import tqdm
from bson.binary import Binary
from bson.objectid import ObjectId
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

nlp = spacy.load('en')

spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS


def tokenizer(x):
    """
    Create a tokenizer function
    """
    return [tok.text for tok in nlp.tokenizer(x) if tok.text != ' ']


client = MongoClient('localhost:27017')
db = client['yelp']
feature_names = db['feature_names']

COLLECTION_NAMES = ['review_shopping']
NUM_FEATURES = [2000]

for collection_name in COLLECTION_NAMES:
    collection = db[collection_name]
    for N_F in NUM_FEATURES:
        # BUILD DICTIONARY
        BUILD_VOCAB = True
        vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None,
                                     stop_words=spacy_stopwords, max_features=N_F)
        if BUILD_VOCAB:
            reviews = pd.DataFrame(list(collection.find({}, {'_id': 0, 'text': 1}))).values.flatten()
            vectorizer.fit(reviews)
            feature_names.update_one({f'bow_size': N_F, 'collection_name': collection_name},
                                     {'$set': {'features': vectorizer.get_feature_names()}},
                                     upsert=True)

        CONVERT = True
        if CONVERT:
            data = collection.find({}, {'_id': 1, 'text': 1})
            operations = []
            for doc in tqdm.tqdm(data, total=collection.count_documents({}),
                                 desc=f'Insert BoW field in {collection_name} with size {N_F}'):
                bow = vectorizer.transform([doc['text']])
                bow = csr_matrix(np.rint(np.log(1 + bow.toarray().flatten())))
                operations.append(
                        UpdateOne({'_id': ObjectId(doc['_id'])},
                                  {'$set': {f'bow_{N_F}': Binary(pickle.dumps(bow, protocol=2))}}, upsert=False))
                if len(operations) % 5000 == 0:
                    try:
                        result = collection.bulk_write(operations, ordered=False)
                        pprint(result.bulk_api_result)
                    except BulkWriteError as bwe:
                        pprint(bwe.details)
                    operations = []
            try:
                result = collection.bulk_write(operations, ordered=False)
                pprint(result.bulk_api_result)
            except BulkWriteError as bwe:
                pprint(bwe.details)
            operations = []
        GROUP_BY = True
        GROUP_FIELDS = [('user', '$user_id'), ('business', '$business_id')]
        if GROUP_BY:
            for field_name, field in GROUP_FIELDS:
                collection.aggregate([{'$sort': {'convertedDate': 1}},
                                      {'$group': {'_id': field,
                                                  'time': {'$push': '$timestamp'},
                                                  'bow': {'$push': f'$bow_{N_F}'}
                                                  }
                                       },
                                      {'$out': f'{collection_name}_by_{field_name}_{N_F}'}], allowDiskUse=True)
                if f'{collection_name}_by_{field_name}' not in db.list_collection_names():
                    collection.aggregate([{'$sort': {'convertedDate': 1}},
                                          {'$group': {'_id': field,
                                                      'time': {'$push': '$timestamp'},
                                                      'text': {'$push': '$text'}
                                                      }
                                           },
                                          {'$out': f'{collection_name}_by_{field_name}'}], allowDiskUse=True)
                grouped_col = db[f'{collection_name}_by_{field_name}_{N_F}']
                N = grouped_col.count_documents({})
                c = grouped_col.find({}, {'bow': 1})
                bulk = grouped_col.initialize_unordered_bulk_op()
                count = 0
                for doc in tqdm.tqdm(c,
                                     f"Stack the list of bows in sparse matrix for {collection_name}_by_{field_name}_{N_F}",
                                     N):
                    bow_matrix = scipy.sparse.vstack(list(map(pickle.loads, doc['bow'])))
                    bulk.find({'_id': doc['_id']}).update_one(
                            {'$set': {'bow': Binary(pickle.dumps(bow_matrix, protocol=2))}})
                    count += 1
                    if count % 1000 == 0:
                        print(bulk.execute())
                        bulk = grouped_col.initialize_unordered_bulk_op()
                print(bulk.execute())

        SPLIT_DATA = True
        GROUP_FIELDS = [('user', '$user_id'), ('business', '$business_id')]
        if SPLIT_DATA:
            for field_name, _ in GROUP_FIELDS:
                data_bow = list(
                        db[f'{collection_name}_by_{field_name}_{N_F}'].find({'$where': 'this.time.length >= 5'}))
                train_bow, test_bow = train_test_split(data_bow, test_size=.2, random_state=42)
                validation_bow, test_bow = train_test_split(test_bow, test_size=.5, random_state=42)
                data_text = list(db[f'{collection_name}_by_{field_name}'].find({'$where': 'this.time.length >= 5'}))
                train_text, test_text = train_test_split(data_text, test_size=.2, random_state=42)
                validation_text, test_text = train_test_split(test_text, test_size=.5, random_state=42)
                collection_list = db.list_collection_names()
                if f'{collection_name}_by_{field_name}_{N_F}_train' in collection_list:
                    db[f'{collection_name}_by_{field_name}_{N_F}_train'].drop()
                if f'{collection_name}_by_{field_name}_{N_F}_validation' in collection_list:
                    db[f'{collection_name}_by_{field_name}_{N_F}_validation'].drop()
                if f'{collection_name}_by_{field_name}_{N_F}_test' in collection_list:
                    db[f'{collection_name}_by_{field_name}_{N_F}_test'].drop()
                if f'{collection_name}_by_{field_name}_train' in collection_list:
                    db[f'{collection_name}_by_{field_name}_train'].drop()
                if f'{collection_name}_by_{field_name}_validation' in collection_list:
                    db[f'{collection_name}_by_{field_name}_validation'].drop()
                if f'{collection_name}_by_{field_name}_test' in collection_list:
                    db[f'{collection_name}_by_{field_name}_test'].drop()

                db[f'{collection_name}_by_{field_name}_{N_F}_train'].insert_many(train_bow)
                db[f'{collection_name}_by_{field_name}_{N_F}_validation'].insert_many(validation_bow)
                db[f'{collection_name}_by_{field_name}_{N_F}_test'].insert_many(test_bow)

                db[f'{collection_name}_by_{field_name}_train'].insert_many(train_text)
                db[f'{collection_name}_by_{field_name}_validation'].insert_many(validation_text)
                db[f'{collection_name}_by_{field_name}_test'].insert_many(test_text)
