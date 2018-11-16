# -*- coding: utf-8 -*-
# ########################################################################################################################
#                                                                                                                      #
# AUTEUR: CHIHEB EL OUEKDI
#  IFT - 7022
#  TRAVAIL-2
# DATE: 13 Novembre 2018                                                                                                #
########################################################################################################################

import re
import nltk
import sklearn
from sklearn.datasets import load_files
from nltk.tokenize import word_tokenize
from nltk.corpus import sentiwordnet as swn
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
import sys
from sklearn.metrics import recall_score


########################################################################################################################
#                                                                                                                      #
#               Preparation de la structure de calcul NLTK et SCIKIT LEARN
#                                                                                                                      #
########################################################################################################################

ps = PorterStemmer()
bookdir = r'Book'
book_train = load_files(bookdir, shuffle=True)

# Liste des stop words
stopwd  = set(sw.words('english'))


#affichage des resultat dans un fichier
#f = open('A SOUINONavec-stemming1.txt','w')
#stdout_old = sys.stdout
#sys.stdout = f


#fonction pour:
#             - éliminer les mot qui ne sont pas des nom, verbe, adjectif ou adverbe
#             - faire le stemming d'un mot (ou non) selon la variable "normalize"
#             - calcul le poid positif ou négatif et insert le mot "mot_positif/mot_negatif en conséquence
def Stemming_texte( token, tag, normalize):
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
                return ps.stem(token), res_negpos
            else:
                return token, res_negpos
        except:
            return None

#Fonction pour préparer la liste d'attribut d'une phrase
def Tranforme_texte(texte_st, normalize):


    resultat=[]

    # Converting to Lowercase
    try:
        document_tok = sent_tokenize(str(texte_st).lower())
    except:
        texte_st = "This is a good bad book"
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

            # Lem#matize the token and yield
            lemma = Stemming_texte(token, tag,normalize)
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

#Fonction pour préparer la liste d'attribut d'un texte/paragraphe
def Transform_documents(texte_doc_list,normalize):
    documents = []
    for sen in range(0, len(texte_doc_list)):
        documents.append(Tranforme_texte(texte_doc_list[sen],normalize))
    return documents


#Fonction pour afficher les faits saillants de l'entrainement
def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print ("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))

X, y = book_train.data, book_train.target
# 1 => Stemming
m_normalize = 1
# 2 => Pas de Stemming
#m_normalize = 2

New_X = Transform_documents(X,m_normalize)


########################################################################################################################
#                                                                                                                      #
#               Afficher les résultat en format CSV (délimitauer ;) pour faciliter l'importation dans Excel
#                                                                                                                      #
########################################################################################################################


print("NAIVE BAYES;-----;-----;LOGISTIC REGRESSION;-----;-----")
print("accuracy;precision;recall;accuracy;precision;recall")

#variation des comptes minimaux <a considérer
for mindf in range(1,51):

    #Bag of word
    vectorizer = CountVectorizer(min_df=mindf, stop_words=stopwd)
    X = vectorizer.fit_transform(New_X).toarray()
    # 90% entrainement 10% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    # NAIVE BAYES
    clf = MultinomialNB().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    average_precision_nb = average_precision_score(y_test, y_pred)
    accuracy_score_nb=sklearn.metrics.accuracy_score(y_test, y_pred)
    recall_nb = recall_score(y_test, y_pred)
    #show_most_informative_features(vectorizer,clf)

    #LOGISTIC REGRESSION
    logreg = LogisticRegression().fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    average_precision_lr = average_precision_score(y_test, y_pred)
    recall_lg = recall_score(y_test, y_pred)
    accuracy_score_lr = sklearn.metrics.accuracy_score(y_test, y_pred)


    print(accuracy_score_nb.__str__()+";"+average_precision_nb.__str__()+";"+recall_nb.__str__()+";"+accuracy_score_lr.__str__()+";"+average_precision_lr.__str__()+";"+recall_lg.__str__())



print ("All done")