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
########################################################################################################################
#                                                                                                                      #
#               Preparation de la structure de calcul de probabilite avec NLTK
#                                                                                                                      #
########################################################################################################################


def Attributs_phrases(phrase,nbgrm):

    attribut = {}
    phrase = phrase.lower()
    ngram_train = list(nltk.ngrams(phrase,nbgrm))
    freq_dist_ngram = FreqDist(ngram_train)

    for t in ngram_train:
        attribut["count({})".format(t)] = freq_dist_ngram[t]

    return attribut


def Preparer_Attribut_List(chemin_fichier, nb_ligne, langue, ngramme):

    file = open(chemin_fichier, 'rb')
    txt_lect = [ligne for ligne in file]
    file.close()

    Attributs_list = []
    for ln in range(0, len(txt_lect), nb_ligne):
        if nb_ligne > 1:
            x=txt_lect[ln:ln + (nb_ligne - 1)]
            x =' '.join(map(str, x))
        else:
            x = txt_lect[ln]
        Attributs_list.append((Attributs_phrases(x.__str__(), ngramme), langue))

    return Attributs_list



def Reconnaitre_langue(fichier, classificateur,nb_ligne, ngramme):

    count_langue = {"en":0,"es":0,"fr":0, "pt":0}
    f = open(fichier, 'rb')

    testset=[]
    txt_lect = [ligne for ligne in f]
    f.close()
    for ln in range(0, len(txt_lect), nb_ligne):
        if NB_LIGNES > 1:
            x = txt_lect[ln:ln + (nb_ligne - 1)]
            x = ' '.join(map(str, x))
        else:
            x = txt_lect[ln]
        testset.append(Attributs_phrases(x, ngramme))

    for atrb in testset:
        resultat = classificateur.classify(atrb)
        count_langue[resultat] +=1
        langue_texte = "en"
        nb_phrase_langue =0
    for x in count_langue:
        if count_langue[x] > nb_phrase_langue:
            langue_texte = x
            nb_phrase_langue = count_langue[x]
    print (count_langue)
    return langue_texte


NB_LIGNES=3
N_GRAMMES =4
Chemin_fich_en = 'identification_langue/corpus_entrainement/english-training.txt'
Chemin_fich_es = 'identification_langue/corpus_entrainement/espanol-training.txt'
Chemin_fich_fr = 'identification_langue/corpus_entrainement/french-training.txt'
Chemin_fich_pt = 'identification_langue/corpus_entrainement/portuguese-training.txt'


trainset = Preparer_Attribut_List(Chemin_fich_en,NB_LIGNES,"en",N_GRAMMES) \
           +  Preparer_Attribut_List(Chemin_fich_es,NB_LIGNES,"es",N_GRAMMES)\
           +  Preparer_Attribut_List(Chemin_fich_fr,NB_LIGNES,"fr",N_GRAMMES)\
           + Preparer_Attribut_List(Chemin_fich_pt,NB_LIGNES,"pt",N_GRAMMES)


classifier = NaiveBayesClassifier.train(trainset)

Chemin_fich_test = "identification_langue/corpus_test1/"


p1 = re.compile(r'.*(fr|en|es|pt)\.txt')

testset=[]
accuracy = 0
for fabc in os.listdir(Chemin_fich_test):
    m1 = p1.search(fabc)
    if m1 is not None:

        testset += Preparer_Attribut_List(Chemin_fich_test+fabc, NB_LIGNES, m1.group(1), N_GRAMMES)

        print(fabc + "::" + Reconnaitre_langue(Chemin_fich_test + fabc, classifier,NB_LIGNES,N_GRAMMES))
        if Reconnaitre_langue(Chemin_fich_test + fabc, classifier,NB_LIGNES,N_GRAMMES) == m1.group(1):

            accuracy +=1

print (nltk.classify.util.accuracy(classifier, testset))

print (float(accuracy))

classifier.show_most_informative_features()



