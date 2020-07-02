#!/usr/bin/env python
# coding: utf-8

# # Few-Shot Learning Email Classification with Pre-Trained Word2Vec Embeddings

'''
Code for retraining the XG boost model
'''

import os
import re
from random import seed
import spacy
import pandas as pd
import numpy as np
import joblib
import xgboost
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
# from random import sample

from wordfile import func
from xgb_inp import EMBEDDINGS_INDEX


def train():
    '''
    Load the dataset, clean the subject and body, then do data sampling for training the model.
    '''
    seed(40)
    np.random.seed(40)
    df = pd.read_csv("./emaildataset.csv", usecols=['Subject', 'Body', 'Class'])
    df.head()

    nlp = spacy.load('en')

    my_stop =['\'d', '\'ll', '\'m', '\'re', '\'s', 'a', 'cc', 'subject', 'http', 'gbp',
              'usd', 'eur', 'inr', 'cad', 'thanks', 'acc', 'id', 'account', 'regards',
              'hi', 'hello', 'thank you', 'greetings', 'about', 'above', 'across',
              'after', 'afterwards', 'alone', 'along', 'among', 'amongst', 'amount',
              'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway',
              'anywhere', 'around', 'as', 'at', 'because', 'before', 'beforehand',
              'behind', 'below', 'beside', 'besides', 'between', 'both', 'bottom',
              'but', 'by', 'ca', 'call', 'can', 'could', 'did', 'do', 'does',
              'doing', 'down', 'due', 'during', 'each', 'eight', 'either', 'eleven',
              'else', 'elsewhere', 'everyone', 'everything', 'everywhere', 'fifteen',
              'fifty', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'four',
              'from', 'front', 'further', 'he', 'hence', 'her', 'here', 'hereafter',
              'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself',
              'his', 'how', 'hundred', 'if', 'indeed', 'into', 'it', 'its', 'itself',
              'just', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'many',
              'may', 'me', 'meanwhile', 'might', 'mine', 'more', 'moreover', 'much',
              'must', 'my', 'myself', 'name', 'namely', 'neither', 'nevertheless',
              'next', 'nine', 'no', 'nobody', 'now', 'nowhere', 'of', 'off', 'often',
              'on', 'one', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours',
              'ourselves', 'out', 'own', 'per', 'perhaps', 'please', 'quite', 'rather',
              're', 'really', 'regarding', 'same', 'she', 'side', 'since', 'six', 'sixty',
              'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'somewhere',
              'such', 'ten', 'that', 'the', 'their', 'them', 'themselves', 'then',
              'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
              'thereupon', 'these', 'they', 'third', 'this', 'those', 'three',
              'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top',
              'toward', 'towards', 'twelve', 'twenty', 'two', 'under', 'up', 'upon',
              'us', 'using', 'various', 'via', 'we', 'well', 'whatever', 'whence',
              'whenever', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon',
              'wherever', 'whether', 'which', 'while', 'whither', 'whoever', 'whole',
              'whom', 'whose', 'with', 'within', 'yet', 'you', 'your',
              'yours', 'yourself', 'yourselves', '\'m', '\'re', '’s']

    def get_only_chars(text):
        text = text.replace("-", " ") #replace hyphens with spaces
        text = text.replace("\t", " ")
        text = text.replace("\n", " ")
        text = text.replace("n't", " not")
        text = text.replace("l've", "l have")
        text = text.replace("d've", "d have")


        text = nlp(text)
        text = " ".join(token.orth_ for token in text if not token.is_punct | token.is_space)
        text_short = ""

        for i in text.lower().split():
            if func(i) is not None:
                text_short += func(i) + " "
            else:
                text_short += i + " "

        text_short = text_short.rstrip()
        text = " ".join([i for i in text_short.lower().split() if i not in my_stop])
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        text = " ".join([i for i in text.split() if len(i) != 1])

        return text


    for _ in range(df.shape[0]):
        # merge subject and body strings
        df['Text'] = (df['Subject'] + " " + df['Body'])


    def converter(var):
        try:
            return ' '.join([var.lower() for var in str(var).split()])
        except AttributeError:
            return None  # or some other value


    df['Text'] = df['Text'].apply(converter)
    df['Text'] = df['Text'].apply(lambda x: get_only_chars(x))

    df = df.drop_duplicates('Text')
    df = df.sample(frac=1).reset_index(drop=True)


    ## set the by default to(in case there are no arguments later on, error handling):
    # num_classes = df.Class.unique() # no. of classes we consider (as dataset has many classes)
    # sample_size = 5 # the number of labeled sampled we’ll require from the user
    # smallest_sample_size = min(df['Class'].value_counts())


    # Generate samples that contains K samples of each class
    def gen_sample(sample_size, num_classes, df):

        df_1 = df[(df["Class"] < num_classes+1)].reset_index().drop(["index"], axis=1).reset_index().drop(["index"], axis=1)

        train_set = df_1[df_1["Class"] == np.unique(df_1['Class'])[0]].sample(sample_size)
        train_index = train_set.index.tolist()

        for i in range(1, num_classes):
            train_2 = df_1[df_1["Class"] == np.unique(df_1['Class'])[i]].sample(sample_size)
            #train_2 = df_1[df_1["Class"] == np.unique(df_1['Class'])[i]]
            # .sample(sample_size, replace = True)
            train_set = pd.concat([train_set, train_2], axis=0)
            train_index.extend(train_2.index.tolist())

        test_set = df_1[~df_1.index.isin(train_index)]
        return train_set, test_set


    # Encoding categorical variables

    l_e = LabelEncoder()
    df['Class'] = l_e.fit_transform(df['Class'])


    def transform_sentence(text, Embeddings_Index):

        # def preprocess_text(raw_text, model=Embeddings_Index):
        def preprocess_text(raw_text):

            raw_text = raw_text.split()
            return list(filter(lambda x: x in Embeddings_Index.keys(), raw_text))

        tokens = preprocess_text(text)

        if not tokens:
            return np.zeros(300)

        vec = [Embeddings_Index[i] for i in tokens]
        text_vector = np.mean(vec, axis=0)
        return np.array(text_vector)



    if not os.path.exists('./pkl_objects'):
        os.mkdir('./pkl_objects')

    joblib.dump(l_e, './pkl_objects/labelencoder.pkl')


    # Return accuracy score of ML model
    def return_score_xgb(sample_size, num_classes, df):

        train_set, test_set = gen_sample(sample_size, num_classes, df=df)

        x_train = train_set['Text'].values
        y_train = train_set['Class'].values
        x_test = test_set['Text'].values
        y_test = test_set['Class'].values

        x_train_mean = np.array([transform_sentence(x, EMBEDDINGS_INDEX) for x in x_train])
        x_test_mean = np.array([transform_sentence(x, EMBEDDINGS_INDEX) for x in x_test])

        # XG Boost
        clf = xgboost.XGBClassifier()
        clf.fit(x_train_mean, y_train)
        # eval_set = [(x_train_mean, y_train), (x_test_mean, y_test)]
        # clf.fit(x_train_mean, y_train, early_stopping_rounds=10,
        # eval_metric="merror", eval_set=eval_set, verbose=True)


        joblib.dump(clf, './pkl_objects/clf.pkl')

        y_pred = clf.predict(x_test_mean)

        # evaluate predictions
        accuracy = accuracy_score(y_pred, y_test)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

        return accuracy_score(y_pred, y_test)



    # all_accuracy_xgb = {2: [], 3: [], 4: [], 5: [], 6: [], 7: []}

    # for num_samples in range(1, 40):

    #     for num_cl in range(2, 7):
    #         all_accuracy_xgb[num_cl].append(return_score_xgb(num_samples, num_cl, df))



    all_accuracy = {0: []}

    for num_samples in range(1, 41):
        all_accuracy[0].append(return_score_xgb(num_samples, len(df.Class.unique()), df))


train()