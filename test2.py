# -*- coding: utf-8 -*-
# ########################################################################################################################
#                                                                                                                      #
# AUTEUR: CHIHEB EL OUEKDI
#  IFT - 7022
#  TRAVAIL-2
# DATE:                                                                                                 #
########################################################################################################################

import nltk
from nltk.probability import *
from nltk.classify import NaiveBayesClassifier
from nltk.classify import MaxentClassifier
import re
import os

import sys





#Lecture du contenu du fichier d'entrainement
file = open('identification_langue/corpus_entrainement/english-training.txt', 'rb')
contenu_english_train = file.read()
file.close()

########################################################################################################################
#                                                                                                                      #
#               Preparation de la structure de calcul de probabilite avec NLTK
#                                                                                                                      #
########################################################################################################################


def Attributs_phrase(phrase,nbgrm):
    nbgrm = 1
    attribut = {}
    phrase = phrase.lower()
    ngram_train = list(nltk.ngrams(phrase,nbgrm))
    freq_dist_ngram = FreqDist(ngram_train)

    for t in ngram_train:
        attribut["count({})".format(t)] = freq_dist_ngram[t]

    return attribut


def Reconnaitre_langue(fichier, classificateur):

    count_langue = {"en":0,"es":0,"fr":0, "pt":0}
    f = open(fichier, 'rb')

    testset=[]
    txt_lect = [ligne for ligne in f]
    file.close()
    for ln in range(0, len(txt_lect), NB_LIGNES):
        if NB_LIGNES > 1:
            x = txt_lect[ln:ln + (NB_LIGNES - 1)]
            x = " ".join(x.__str__())
        else:
            x = txt_lect[ln]
        testset.append(Attributs_phrase(x, 3))

    for atrb in testset:
        resultat = classificateur.classify(atrb)
        count_langue[resultat] +=1
        langue_texte = "en"
        nb_phrase_langue =0
    for x in count_langue:
        if count_langue[x] > nb_phrase_langue:
            langue_texte = x
            nb_phrase_langue = count_langue[x]
    return langue_texte


NB_LIGNES=10

file = open('identification_langue/corpus_entrainement/english-training.txt', 'rb')

txt_lect = [ligne for ligne in file]
file.close()
train_english=[]
for ln in range(0,len(txt_lect),NB_LIGNES):
    if NB_LIGNES >1:
        x= txt_lect[ln:ln+(NB_LIGNES-1)]
        x = ' '.join(map(str, x))
    else:
        x = txt_lect[ln]
    train_english.append((Attributs_phrase(x, 3), "en"))






file = open('identification_langue/corpus_entrainement/espanol-training.txt', 'rb')

txt_lect = [ligne for ligne in file]
file.close()
train_espagnol=[]
for ln in range(0,len(txt_lect),NB_LIGNES):
    if NB_LIGNES >1:
        x= txt_lect[ln:ln+(NB_LIGNES-1)]
        x = ' '.join(map(str, x))
    else:
        x = txt_lect[ln]
    train_espagnol.append((Attributs_phrase(x, 3), "es"))


file = open('identification_langue/corpus_entrainement/french-training.txt', 'rb')
txt_lect = [ligne for ligne in file]
file.close()
train_french=[]
for ln in range(0,len(txt_lect),NB_LIGNES):
    if NB_LIGNES >1:
        x= txt_lect[ln:ln+(NB_LIGNES-1)]
        x = ' '.join(map(str, x))
    else:
        x = txt_lect[ln]
    train_french.append((Attributs_phrase(x, 3), "fr"))

file = open('identification_langue/corpus_entrainement/portuguese-training.txt', 'rb')
txt_lect = [ligne for ligne in file]
file.close()
train_portuguese=[]
for ln in range(0,len(txt_lect),NB_LIGNES):
    if NB_LIGNES >1:
        x= txt_lect[ln:ln+(NB_LIGNES-1)]
        x = ' '.join(map(str, x))
    else:
        x = txt_lect[ln]
    train_portuguese.append((Attributs_phrase(x, 3), "pt"))


trainset = train_english + train_espagnol+train_french+train_portuguese





classifier = NaiveBayesClassifier.train(trainset)

rep_test = "identification_langue/corpus_test1/"


p1 = re.compile(r'.*(fr|en|es|pt)\.txt')

testset=[]
accuracy = 0
for fabc in os.listdir(rep_test):
    m1 = p1.search(fabc)
    if m1 is not None:
        file = open(rep_test+fabc, 'rb')
        txt_lect = [ligne for ligne in file]
        file.close()
        for ln in range(0, len(txt_lect), NB_LIGNES):
            if NB_LIGNES > 1:
                x = txt_lect[ln:ln + (NB_LIGNES - 1)]
                x = ' '.join(map(str, x))
            else:
                x = txt_lect[ln]
            testset.append((Attributs_phrase(x, 3), m1.group(1)))

        if Reconnaitre_langue(rep_test + fabc, classifier) == m1.group(1):
            accuracy +=1

print (nltk.classify.util.accuracy(classifier, testset))

print (float(accuracy))

classifier.show_most_informative_features()



