import barebones_enc_dec_batched as bedb
import dynet as dy
import sys
import readData
from collections import defaultdict
import numpy as np

modelFile=str(sys.argv[1])
outFile1Name=str(sys.argv[2])
outFile2Name=str(sys.argv[3])

wids_de=defaultdict(int)
wids_en=defaultdict(int)

model=dy.Model()
encoder_params={}
decoder_params={}
(encoder,revcoder,decoder,encoder_params["lookup"],decoder_params["lookup"],decoder_params["R"],decoder_params["bias"])=model.load(modelFile)

train_sentences_en=readData.read_corpus(wids_en,mode="train",update_dict=True,min_frequency=bedb.MIN_EN_FREQUENCY,language="en")
train_sentences_de=readData.read_corpus(wids_de,mode="train",update_dict=True,min_frequency=bedb.MIN_DE_FREQUENCY,language="de")

reverse_wids_de=bedb.reverseDictionary(wids_de)
reverse_wids_en=bedb.reverseDictionary(wids_en)

train_sentences_en=None
train_sentences_de=None

test_sentences_de=readData.read_corpus(wids_de,mode="test",update_dict=False,min_frequency=bedb.MIN_DE_FREQUENCY,language="de")
blind_sentences_de=readData.read_corpus(wids_de,mode="blind",update_dict=False,min_frequency=bedb.MIN_DE_FREQUENCY,language="de")

K=4


outFile1=open(outFile1Name,"w")
for testIndex,test_sentence_de in enumerate(test_sentences_de):
    sentence_en_hat,interpreted_sentence_en_hat,loss=bedb.beamDecode(model,encoder,revcoder,decoder,encoder_params,decoder_params,test_sentence_de,reverse_wids_en,k=K,concatenative=True)
    outFile1.write(interpreted_sentence_en_hat+"\n")
    #print interpreted_sentence_en_hat
    if testIndex%10==0:
        print testIndex
        print interpreted_sentence_en_hat

outFile1.close()


outFile2=open(outFile2Name,"w")
for blindIndex,blind_sentence_de in enumerate(blind_sentences_de):
    sentence_en_hat,interpreted_sentence_en_hat,loss=bedb.beamDecode(model,encoder,revcoder,decoder,encoder_params,decoder_params,blind_sentence_de,reverse_wids_en,k=K,concatenative=True)
    outFile2.write(interpreted_sentence_en_hat+"\n")
    #print interpreted_sentence_en_hat
    if blindIndex%10==0:
        print blindIndex
        print interpreted_sentence_en_hat

outFile2.close()




