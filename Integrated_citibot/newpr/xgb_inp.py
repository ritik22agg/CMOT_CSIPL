"""
Using inp function from this file to predict output class of email
PyLint Score: 9.08/10
"""
import re
import spacy
import numpy as np
import joblib

from wordfile import func
import xgboost


EMBEDDINGS_INDEX = {}
with open('./glove.6B.300d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:], dtype='float32')
        EMBEDDINGS_INDEX[word] = coeffs
    f.close()

NLP = spacy.load('en')

MY_STOP = ["'d", "'ll", "'m", "'re", "'s", "'ve", 'a', 'cc', 'subject', 'http',
           'gbp', 'usd', 'eur', 'inr', 'cad',
           'thanks', "acc", "id", 'account', 'regards', 'hi', 'hello',
           'thank you', 'greetings', 'about', 'above',
           'across', 'after', 'afterwards', 'against', 'alone', 'along',
           'already', 'also', 'although', 'am', 'among',
           'amongst', 'amount', 'an', 'and', 'another', 'any', 'anyhow',
           'anyone', 'anything', 'anyway', 'anywhere',
           'are', 'around', 'as', 'at', 'be', 'became', 'because', 'become',
           'becomes', 'becoming', 'been', 'before',
           'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
           'between', 'both', 'bottom', 'but', 'by',
           'ca', 'call', 'can', 'could', 'did', 'do', 'does', 'doing', 'down',
           'due', 'during', 'each', 'eight',
           'either', 'eleven', 'else', 'elsewhere', 'every', 'everyone',
           'everything', 'everywhere', 'fifteen', 'fifty',
           'first', 'five', 'for', 'former', 'formerly', 'forty', 'four',
           'from', 'front', 'further', 'get', 'give',
           'go', 'had', 'has', 'have', 'he', 'hence', 'her', 'here',
           'hereafter', 'hereby', 'herein', 'hereupon',
           'hers', 'herself', 'him', 'himself', 'his', 'how', 'however',
           'hundred', 'i', 'if', 'in', 'indeed', 'into',
           'is', 'it', 'its', 'itself', 'just', 'keep', 'last', 'latter',
           'latterly', 'least', 'less', 'made', 'make',
           'many', 'may', 'me', 'meanwhile', 'might', 'mine', 'more',
           'moreover', 'mostly', 'move', 'much', 'must',
           'my', 'myself', 'name', 'namely', 'neither', 'nevertheless', 'next',
           'nine', 'no', 'nobody', 'now',
           'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto',
           'or', 'other', 'others', 'otherwise',
           'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per',
           'perhaps', 'please', 'put', 'quite',
           'rather', 're', 'really', 'regarding', 'same', 'say', 'see', 'seem',
           'seemed', 'seeming', 'seems', 'serious',
           'several', 'she', 'should', 'show', 'side', 'since', 'six', 'sixty',
           'so', 'some', 'somehow', 'someone',
           'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such',
           'take', 'ten', 'than', 'that', 'the',
           'their', 'them', 'themselves', 'then', 'thence', 'there',
           'thereafter', 'thereby', 'therefore', 'therein',
           'thereupon', 'these', 'they', 'third', 'this', 'those', 'though',
           'three', 'through', 'throughout', 'thru',
           'thus', 'to', 'together', 'too', 'top', 'toward', 'towards',
           'twelve', 'twenty', 'two', 'under', 'unless',
           'until', 'up', 'upon', 'us', 'used', 'using', 'various', 'very',
           'via', 'was', 'we', 'well', 'were',
           'whatever', 'whence', 'whenever', 'whereafter', 'whereas', 'whereby',
           'wherein', 'whereupon', 'wherever',
           'whether', 'which', 'while', 'whither', 'whoever', 'whole', 'whom',
           'whose', 'will', 'with', 'within',
           'would', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves',
           '‘d', '‘ll', '‘m', '‘re', '‘s', '‘ve',
           '’d', '’ll', '’m', '’re', '’s', '’ve']


def get_only_chars(text):
    """
    cleaning of text
    remove white spaces, tabs, newlines, non alphabets
    convert to lower case
    replace financial abbreviations with full form
    remove punctuation and stopwords
    lemmatize
    """
    text = text.replace("-", " ")
    text = text.replace("\t", " ")
    text = text.replace("\n", " ")

    text = text.rstrip()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    newt = ""

    for i in text.lower().split():
        if func(i) is not None:
            newt += func(i) + " "
        else:
            newt += i + " "

    newt = newt.rstrip()

    text = " ".join(i for i in newt.lower().split())
    text = " ".join(token for token in text.split() if token not in MY_STOP)

    doc = NLP(text)
    normalized = " ".join(token.lemma_ for token in doc)
    doc = " ".join(token.orth_ for token in NLP(normalized)
                   if not token.is_punct | token.is_space)
    return doc


def transform_sentence(text, EMBEDDINGS_INDEX):
    """
    transform text to embeddings
    """
    def preprocess_text(raw_text, model=EMBEDDINGS_INDEX):
        """
        text preprocessing
        """
        raw_text = raw_text.split()
        return list(filter(lambda x: x in EMBEDDINGS_INDEX.keys(), raw_text))

    tokens = preprocess_text(text)

    if not tokens:
        return np.zeros(300)

    vec = [EMBEDDINGS_INDEX[i] for i in tokens]
    text_vector = np.mean(vec, axis=0)
    return np.array(text_vector)


LE = joblib.load('./pkl_objects/labelencoder.pkl')
CLF = joblib.load('./pkl_objects/clf.pkl')


def find_num(sub):
    """
    extract transaction id from email subject
    """
    nums = []
    res = ''
    for word in sub.split():
        try:
            nums.append(float(word))
        except ValueError:
            pass
    if not nums:
        res = '000000'
    else:
        res = int(nums[0])
    return str(res)


def inp(emailto, emailfrom, subj, bod):
    """
    returns predicted class, transaction id from subject
    """
    text = subj + " " + bod
    tid = str(find_num(text))
    text = get_only_chars(text)
    x_test_mean = np.array([transform_sentence(text, EMBEDDINGS_INDEX)])

    y_pred = CLF.predict(x_test_mean)
    out = LE.inverse_transform(y_pred)
    return out[0], tid


k, ID = inp("fvf", "defrfg", "payment processed 123456",
            "hi, the payment for acc 1234 for usd 3456 was paid successfully.")
print(k, type(k), ID, type(ID))