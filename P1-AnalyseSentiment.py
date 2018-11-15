import re
import nltk
import sklearn
from sklearn.datasets import load_files
from nltk.tokenize import word_tokenize
from nltk.corpus import sentiwordnet as swn


#nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords as sw
from sklearn.metrics import average_precision_score
import string

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from nltk import sent_tokenize
from nltk import pos_tag
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from sklearn.linear_model import LogisticRegression


ps = PorterStemmer()
bookdir = r'Book'
# loading all files as training data.
book_train = load_files(bookdir, shuffle=True)
#print(book_train.data)
# target names ("classes") are automatically generated from subfolder names
#print(book_train.target_names)
#print(book_train.filenames)

#nltk.download('sentiwordnet')

stopwd  = set(sw.words('english'))

lemmatizer = WordNetLemmatizer()

def lemmatize_texte( token, tag, normalize):
    tag = {
        'N': wn.NOUN,
        'V': wn.VERB,
        'R': wn.ADV,
        'J': wn.ADJ
    }.get(tag[0])

    if tag is None:
        return None
    else:

        try:
            res_negpos = swn.senti_synset(token + "." + tag + ".01")
            if normalize == 1:
                return lemmatizer.lemmatize(token, tag), res_negpos
            else:
                if normalize == 2:
                    return ps.stem(token), res_negpos
                else:
                    return token, res_negpos
        except:
            return None


def Tranforme_texte(texte_st, normalize):


    resultat=[]

    # Converting to Lowercase
    try:
        document_tok = sent_tokenize(str(texte_st).lower())
    except:
        texte_st = "This is a goob bad book"
        document_tok = sent_tokenize(str(texte_st).lower())
    doc_res = []
    for sent in document_tok:
        # Break the sentence into part of speech tagged tokens
        set_res = []
        for token, tag in pos_tag(word_tokenize(sent)):
            # Apply preprocessing to the token
            token = token.lower()

            # Si stopword, ignorer le token et continuer
            if token in stopwd:
                continue

            # Lemmatize the token and yield
            lemma = lemmatize_texte(token, tag,normalize)
            if lemma is None:
                continue
            set_res.append(lemma[0])
            # Ajout des comptes des poids positif et negatif
            if (lemma[1].pos_score() > lemma[1].neg_score()):
                set_res.append("mot_positif")
            else:
                if (lemma[1].pos_score() < lemma[1].neg_score()):
                    set_res.append("mot_negatif")
        set_res = ' '.join(set_res)

        doc_res.append(set_res)
    doc_res = ' '.join(doc_res)
    return doc_res

def Transform_documents(texte_doc_list,normalize):
    documents = []
    for sen in range(0, len(texte_doc_list)):
        documents.append(Tranforme_texte(texte_doc_list[sen],normalize))
    return documents

X, y = book_train.data, book_train.target
m_normalize = 1#
vectorizer = CountVectorizer(min_df=4, stop_words=stopwd)
New_X = Transform_documents(X,m_normalize)

X = vectorizer.fit_transform(New_X).toarray()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
clf = MultinomialNB().fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(sklearn.metrics.accuracy_score(y_test, y_pred))




average_precision = average_precision_score(y_test, y_pred)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))


print("-------------------------------------------------")
print("Logistic Reg")
logreg = LogisticRegression().fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(sklearn.metrics.accuracy_score(y_test, y_pred))
average_precision = average_precision_score(y_test, y_pred)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
print("-------------------------------------------------")


#reviews_tt = [0,1,0,0,1,1,0,1,1,1,0,1,1]
#reviews_new = ['This book was bad', 'Absolute joy ride',
#            'Jean  was terrible', 'Steven Seagal shined through.',
#              'This was certainly a movie', 'Two thumbs up', 'I fell asleep halfway through',
#              "We can't wait for the sequel!!", '!', '?', 'I cannot recommend this highly enough',
#              'instant classic.', 'Steven Seagal was amazing. His performance was Oscar-worthy.']
#
#
#
#
#reviews_new_counts = vectorizer.transform(Transform_documents(reviews_new,m_normalize))
#
#pred = clf.predict(reviews_new_counts)
#for review, category in zip(Transform_documents(reviews_new,m_normalize), pred):
#    print('%r => %s' % (review, book_train.target_names[category]))
#print (pred)
#print(sklearn.metrics.accuracy_score(reviews_tt, pred))
#
#
#pred = logreg.predict(reviews_new_counts)
#for review, category in zip(Transform_documents(reviews_new,m_normalize), pred):
#    print('%r => %s' % (review, book_train.target_names[category]))
#print (pred)
#print(sklearn.metrics.accuracy_score(reviews_tt, pred))