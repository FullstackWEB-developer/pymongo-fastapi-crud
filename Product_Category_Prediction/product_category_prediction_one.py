import re
import pickle
import os
import operator
# # from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
from keras.utils.data_utils import pad_sequences
# from nltk.corpus import stopwords
# from nltk.stem.snowball import SnowballStemmer
from keras.models import load_model

# import sys
# import imutils

# Args
# input_path = sys.argv[1]
# output_path = sys.argv[2]
cwd = os.path.dirname(os.path.realpath(__file__)) + "\\"

H5_FILE = cwd + "model-neural-net-only-name.h5"
PICKLE_FILE = cwd + "tokenizer-only-name.pickle"

# Utility function for data cleaning, natural language processing concepts

def decontract(sentence):
    sentence = str(sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can\'t", "can not", sentence)
    sentence = re.sub(r"n\'t", " not", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'s", " is", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'t", " not", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'m", " am", sentence)
    return sentence

def cleanPunc(sentence): 
    sentence = str(sentence)
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned

def keepAlpha(sentence):
    sentence = str(sentence)
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', '', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

# def removeStopWords(sentence):
#     sentence = str(sentence)
#     global re_stop_words
#     return re_stop_words.sub("", sentence)

# stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
#             "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
#             'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
#             'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
#             'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
#             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
#             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
#             'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
#             'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
#             'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
#             's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
#             've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
#             "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
#             "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
#             'won', "won't", 'wouldn', "wouldn't"])

# re_stop_words = re.compile(r"\b(" + "|".join(stopwords) + ")\\W", re.I)

# stemmer = SnowballStemmer("english")
# def stemming(sentence):
#     sentence = str(sentence)
#     stemSentence = ""
#     for word in sentence.split():
#         stem = stemmer.stem(word)
#         stemSentence += stem
#         stemSentence += " "
#     stemSentence = stemSentence.strip()
#     return stemSentence

classes = ['Automotive',
    'Baby Care',
    'Bags, Wallets & Belts',
    'Beauty and Personal Care',
    'Cameras & Accessories',
    'Clothing',
    'Computers',
    'Footwear',
    'Furniture',
    'Home Decor & Festive Needs',
    'Home Furnishing',
    'Home Improvement',
    'Jewellery',
    'Kitchen & Dining',
    'Mobiles & Accessories',
    'Pens & Stationery',
    'Sports & Fitness',
    'Tools & Hardware',
    'Toys & School Supplies',
    'Watches',
    'Others']
def findCategory(name,
                description='',
                tokenizer_file=PICKLE_FILE,
                model_file=H5_FILE
                ):
    name = name.lower()
    name = decontract(name)
    name = cleanPunc(name)
    name = keepAlpha(name)
    # name = removeStopWords(name)
    # name = stemming(name)
    print("~ file: routes.py ~ line 69 ~ name", name, description)

    # name = name.split(' ')[-1]

    # description = description.lower()
    # description = decontract(description)
    # description = cleanPunc(description)
    # description = keepAlpha(description)
    # description = removeStopWords(description)
    # description = stemming(description)

    # information = brand + name + description
    information = name + description

    # loading
    tokenizer = 0
    with open(tokenizer_file, 'rb') as handle:
        tokenizer = pickle.load(handle)

    sequences = tokenizer.texts_to_sequences([information])
    x = pad_sequences(sequences, maxlen=500)

    # joblib_model = joblib.load(joblib_file)
    model = load_model(model_file)
    prediction = model.predict(x)
    pred_scores = [score for pred in prediction for score in pred]
    pred_dict = {}
    for cla,score in zip(classes,pred_scores):
        pred_dict[cla] = score

    sorted_dic = dict(sorted(pred_dict.items(),
                key=operator.itemgetter(1), reverse=True)[:10])
    val = dict(filter(lambda elem: elem[1] > 0.5, sorted_dic.items()))
    # only select first element
    print("~ file: routes.py ~ line 96 ~ val", val)
    if len(val) > 0:
        if (predict_category := list(val.keys())[0]) is not None:
            print(predict_category)
            return predict_category
    return ""

# new function multi for loop
def find_multi_Category(data):
    categories = []
    args = data.split('|')
    # data = args.map(lambda x: categories.append(findCategory(x)))
    # print("~ file: routes.py ~ data", data)
    for arg in args: # for each product
        # with findCategory(arg) as executor:
        # executor = findCategory(arg)
        # print("~ file: routes.py ~ executor", executor)
        categories.append(findCategory(arg))
    print("~ file: routes.py ~ categories", categories)
    return categories


# output_path = findCategory(input_path,'')
# output_path = findCategory('Alberta Side Table','')
# prediction
# print(output_path)
# sys.stdout.flush()