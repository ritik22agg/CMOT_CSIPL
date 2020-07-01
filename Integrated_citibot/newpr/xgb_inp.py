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

MY_STOP = ['\'d', '\'ll', '\'m', '\'re', '\'s', 'a','cc','subject','http', 'gbp', 'usd', 'eur', 'inr', 'cad',
          'thanks', 'acc', 'id', 'account', 'regards', 'hi', 'hello', 'thank you', 'greetings', 'about','above',
          'across','after','afterwards','alone','along','among', 'amongst','amount','an','and','another','any',
          'anyhow','anyone','anything','anyway','anywhere','around','as', 'at','because','before','beforehand',
          'behind','below', 'beside','besides','between','both','bottom','but','by','ca','call','can','could',
          'did', 'do', 'does', 'doing', 'down', 'due', 'during', 'each', 'eight', 'either', 'eleven', 'else',
          'elsewhere', 'everyone', 'everything', 'everywhere', 'fifteen', 'fifty', 'first', 'five', 'for',
          'former', 'formerly', 'forty', 'four', 'from', 'front', 'further', 'he', 'hence', 'her', 'here',
          'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how','hundred',
          'if', 'indeed', 'into', 'it', 'its', 'itself', 'just', 'keep', 'last', 'latter', 'latterly', 'least',
          'less', 'many', 'may', 'me', 'meanwhile', 'might', 'mine', 'more', 'moreover', 'much', 'must', 'my',
          'myself', 'name', 'namely', 'neither', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'now', 'nowhere',
          'of', 'off', 'often', 'on', 'one', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves',
          'out', 'own', 'per', 'perhaps', 'please', 'quite', 'rather', 're', 'really', 'regarding', 'same','she',
          'side', 'since', 'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'somewhere',
          'such', 'ten', 'that', 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter',
          'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'third', 'this', 'those', 'three',
          'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve',
          'twenty', 'two', 'under', 'up', 'upon', 'us', 'using', 'various', 'via', 'we', 'well', 'whatever',
          'whence', 'whenever', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether',
          'which', 'while', 'whither', 'whoever', 'whole', 'whom', 'whose', 'with', 'within', 'yet', 'you', 'your',
          'yours', 'yourself', 'yourselves', '\'m', '\'re', '’s']


def get_only_chars(text):
    """
    cleaning of text
    remove white spaces, tabs, newlines, non alphabets
    convert to lower case
    replace financial abbreviations with full form
    remove punctuation and stopwords
    """
    text = text.replace("-", " ")  # replace hyphens with spaces
    text = text.replace("\t", " ")
    text = text.replace("\n", " ")
    text = text.replace("n't", " not")
    text = text.replace("l've", "l have")
    text = text.replace("d've", "d have")

    text = NLP(text)
    text = " ".join(token.orth_ for token in text if not token.is_punct | token.is_space)
    t = ""

    for i in text.lower().split():
        if func(i) is not None:
            t += func(i) + " "
        else:
            t += i + " "

    t = t.rstrip()
    text = " ".join([i for i in t.lower().split() if i not in MY_STOP])
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = " ".join([i for i in text.split() if len(i) != 1])

    return text


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
CLF = joblib.load('./pkl_objects/clf_40new.pkl')


def find_id(sub):
    """
    extract transaction id from email (subject + body)
    """
    nums = []
    res = ''
    text = re.sub(r'[^0-9]', ' ', sub)
    sub = sub.lower()
    for t in text.split():
        try:
            nums.append(t)
        except ValueError:
            pass
    if not nums:
        res = None
        return res

    def func(sub, nums):
        for i in nums:
            start_idx = sub.find(i)
            if "trans id" in sub[max(0, start_idx - 10): start_idx]:
                return i, True
            elif "transaction id" in sub[max(0, start_idx - 20): start_idx]:
                return i, True
            elif "trans id no." in sub[max(0, start_idx - 25): start_idx]:
                return i, True
            elif "transaction" in sub[max(0, start_idx - 16): start_idx]:
                return i, True
            elif "number" in sub[max(0, start_idx - 8): start_idx]:
                return i, True
            elif "no." in sub[max(0, start_idx - 10): start_idx]:
                return i, True
            elif "num" in sub[max(0, start_idx - 10): start_idx]:
                return i, True
            elif "id" in sub[max(0, start_idx - 10): start_idx]:
                return i, True
        return "", False

    num_str, boolean = func(sub, nums)
    if boolean is True:
        return num_str
    return None


def find_amt(s):
    """
    extract transaction amount from email (subject + body)
    """
    nums = []
    res = ''
    text = re.sub(r'[^0-9]', ' ', s)
    s = s.lower()
    for t in text.split():
        t= ' ' + t + ' '
        try:
            nums.append(t)
        except ValueError:
            pass
    if not nums:
        res = None
        return res

    def func(sub, nums):
        for i in nums:
            start_idx = sub.find(i)
            end_idx = start_idx + len(i)-2
            if " usd " in sub[max(0,start_idx - 10): start_idx+1] or " usd " in sub[end_idx: min(len(s)-1,end_idx + 10)]:
                return i, True
            elif " cad " in sub[max(0,start_idx - 10): start_idx+1] or " cad " in sub[end_idx: min(len(s)-1,end_idx + 10)]:
                return i, True
            elif " inr " in sub[max(0,start_idx - 10): start_idx+1] or " inr " in sub[end_idx: min(len(s)-1,end_idx + 10)]:
                return i, True
            elif " gbp " in sub[max(0,start_idx - 10): start_idx+1] or " gbp " in sub[end_idx: min(len(s)-1,end_idx + 10)]:
                return i, True
            elif " usd " in sub[max(0,start_idx - 10): start_idx+1] or " usd " in sub[end_idx: min(len(s)-1,end_idx + 10)]:
                return i, True
            elif " rs " in sub[max(0,start_idx - 8): start_idx+1] or " rs " in sub[end_idx: min(len(s)-1,end_idx + 8)]:
                return i, True
            elif " rupees " in sub[max(0,start_idx - 8): start_idx+1] or " rupees " in sub[end_idx: min(len(s)-1, end_idx + 8)]:
                return i, True
        return "", False

    num_str, boolean = func(s, nums)
    if boolean is True:
        return num_str
    return None


def common_func(text):
    text = get_only_chars(text)
    X_test_mean = np.array([transform_sentence(text, EMBEDDINGS_INDEX)])
    return X_test_mean


def is_empty_sent(subj, body):
    text = subj + " " + body
    X_test_mean = common_func(text)
    all_zeros = not X_test_mean.any()
    return all_zeros

def inp(emailto, emailfrom, subj, bod):
    """
    returns predicted class, transaction id from subject
    """
    text = subj + " " + bod
    t_id = find_id(text)
    t_amt = find_amt(text)

    X_test_mean = common_func(text)

    y_pred = CLF.predict(X_test_mean)

    out = LE.inverse_transform(y_pred)
    return out[0], t_id, t_amt


k, ID, amt = inp("fvf", "defrfg", "payment processed",
            "hi, the payment for amt. of usd 34535 was paid successfully for trans no. 21879244.")
print(k, type(k), ID, type(ID), amt, type(amt))