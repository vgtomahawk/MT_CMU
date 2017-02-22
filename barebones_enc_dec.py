import sys
import math
import random
import dynet as dy
import readData
from collections import defaultdict
import argparse
import numpy as np
import datetime
import nltk

#Config Definition
EMB_SIZE=128
LAYER_DEPTH=1
HIDDEN_SIZE=128
NUM_EPOCHS=20
START=0
UNK=1
STOP=2
GARBAGE=3
MIN_EN_FREQUENCY=1
MIN_DE_FREQUENCY=1

def attend(encoder_outputs,state_factor_matrix):
    miniBatchLength=state_factor_matrix.npvalue().shape[1]
    encoderOutputLength=state_factor_matrix.npvalue().shape[0]
    hiddenSize=encoder_outputs[0].npvalue().shape[0]

    factor_Products=[state_factor_matrix[l] for l in range(encoderOutputLength)]
    factor_Products=dy.esum([dy.cmult(encoder_outputs[l],dy.concatenate([state_factor_matrix[l]]*hiddenSize)) for l in range(encoderOutputLength)])
    
    return factor_Products

def attend_vector(encoder_outputs,state_factor_vector):
    encoderOutputLength=state_factor_vector.npvalue().shape[0]
    hiddenSize=encoder_outputs[0].npvalue().shape[0]
    
    factor_Products=[dy.cmult(dy.concatenate([state_factor_vector[l]]*hiddenSize),encoder_outputs[l]) for l in range(encoderOutputLength)]
   
    factor_Products=dy.esum(factor_Products)
    return factor_Products


def do_one_example(model,encoder,revcoder,decoder,encoder_params,decoder_params,sentence_de,sentence_en):
    dy.renew_cg()
    total_words=len(sentence_en)
    encoder_lookup=encoder_params["lookup"]
    decoder_lookup=decoder_params["lookup"]
    R=dy.parameter(decoder_params["R"])
    bias=dy.parameter(decoder_params["bias"])

    sentence_de_forward=[START,]+sentence_de
    sentence_de_reverse=sentence_de[::-1]+[START,]

    s=encoder.initial_state()
    inputs=[dy.lookup(encoder_lookup,de) for de in sentence_de_forward]
    states=s.add_inputs(inputs)
    encoder_outputs=[s.output() for s in states]

    s_reverse=revcoder.initial_state()
    inputs=[dy.lookup(encoder_lookup,de) for de in sentence_de_reverse]
    states_reverse=s_reverse.add_inputs(inputs)
    revcoder_outputs=[s.output() for s in states_reverse]

    final_coding_output=encoder_outputs[-1]+revcoder_outputs[-1]
    final_state=states[-1].s()
    final_state_reverse=states_reverse[-1].s()
    final_coding_state=((final_state_reverse[0]+final_state[0]),(final_state_reverse[1]+final_state[1]))
    final_combined_outputs=[revcoder_output+encoder_output for revcoder_output,encoder_output in zip(revcoder_outputs[::-1],encoder_outputs)]

    s_init=decoder.initial_state().set_s(final_state_reverse)
    o_init=s_init.output() 
    alpha_init=dy.softmax(dy.concatenate([dy.dot_product(o_init,final_combined_output) for final_combined_output in final_combined_outputs]))
    c_init=attend_vector(final_combined_outputs,alpha_init)

    
    s_0=s_init
    o_0=o_init
    alpha_0=alpha_init
    c_0=c_init
    

    losses=[]
    
    for en in sentence_en:
        #Calculate loss and append to the losses array
        scores=R*o_0+bias
        loss=dy.pickneglogsoftmax(scores,en)
        losses.append(loss)

        #Take in input
        i_t=dy.concatenate([dy.lookup(decoder_lookup,en),c_0])
        s_t=s_0.add_input(i_t)
        o_t=s_t.output()
        alpha_t=dy.softmax(dy.concatenate([dy.dot_product(o_t,final_combined_output) for final_combined_output in final_combined_outputs]))
        c_t=attend_vector(final_combined_outputs,alpha_t)
        
        #Prepare for the next iteration
        s_0=s_t
        o_0=o_t
        c_0=c_t
        alpha_0=alpha_t

    total_loss=dy.esum(losses)
    return total_loss,total_words

   
"""
def train_batch(model,encoder,revcoder,decoder,encoder_params,decoder_params,train_sentences,valid_sentences,NUM_EPOCHS,modelFile):
    trainer=dy.AdamTrainer(model)
    train_sentences.sort(key = lambda x: len(x[0]))
    train_order=[x*BATCH_SIZE for x in range((len(train_sentences)-1)/BATCH_SIZE)]
    valid_sentences.sort(key = lambda x: len(x[0]))
    valid_order=[x*BATCH_SIZE for x in range((len(valid_sentences)-1)/BATCH_SIZE)]


    for epochId in xrange(NUM_EPOCHS):
        random.shuffle(train_order)
        for iter,tidx in enumerate(train_order):
            if (iter*BATCH_SIZE)%3200==0:
                print "Points Covered,",iter*BATCH_SIZE,"Time,",datetime.datetime.now()
            batch=train_sentences[tidx:tidx+BATCH_SIZE]
            batch_de,batch_de_reverse,batch_en=pad_batch_train(batch)
            do_one_batch(model,encoder,revcoder,decoder,encoder_params,decoder_params,batch_de,batch_de_reverse,batch_en,HIDDEN_SIZE,train=True,trainer=trainer)
        print "Epoch:",epochId
        trainer.update_epoch(1.0)

        print "Computing Per-Sentence Log Probability"
        totalLoss=0.0
        random.shuffle(valid_order)
        for iter,tidx in enumerate(valid_order):
            batch=valid_sentences[tidx:tidx+BATCH_SIZE]
            batch_de,batch_de_reverse,batch_en=pad_batch_train(batch)
            batch_loss=do_one_batch(model,encoder,revcoder,decoder,encoder_params,decoder_params,batch_de,batch_de_reverse,batch_en,HIDDEN_SIZE,train=False,trainer=None)
            totalLoss+=sum([x for x in batch_loss.npvalue()])

        totalLoss/=len(valid_sentences)
        print "Total Loss",totalLoss

                
        references=[]
        hypotheses=[]
        dumpFile=open("dump.txt","w")

        for valid_sentence_de,valid_sentence_en in valid_sentences:
            sentence_de,sentence_de_reverse=pad_sentence_test(valid_sentence_de)
            valid_sentence_en_hat,final_encoding_state,totalLoss=translate_greedy(model,encoder,revcoder,decoder,encoder_params,decoder_params,sentence_de,sentence_de_reverse)
            references.append(strip_stops_and_unks(valid_sentence_en))
            hypotheses.append(strip_stops_and_unks(valid_sentence_en_hat))
            dumpFile.write("Source:"+str(valid_sentence_de)+"\n")
            dumpFile.write("Ref:"+str(valid_sentence_en)+"\n")
            dumpFile.write("Hyp:"+str(valid_sentence_en_hat)+"\n")
            dumpFile.write("Final Encoding State:"+str(final_encoding_state.npvalue())+"\n")


        dumpFile.close()

        BLEU=compute_BLEU(references,hypotheses)
        print "Corpus Bleu:",BLEU
        
        #Save Model
        print "Saving to File"
        model.save(modelFile,[encoder,encoder_params["lookup"],decoder,decoder_params["lookup"],decoder_params["R"],decoder_params["bias"]])

        #Translate test sentences

        #Translate blind sentences
"""

# Read in data
wids_en=defaultdict(lambda: len(wids_en))
wids_de=defaultdict(lambda: len(wids_de))

train_sentences_en=readData.read_corpus(wids_en,mode="train",update_dict=True,min_frequency=MIN_EN_FREQUENCY,language="en")
train_sentences_de=readData.read_corpus(wids_de,mode="train",update_dict=True,min_frequency=MIN_DE_FREQUENCY,language="de")

dicFile=open("dictionary_en.txt","w")
print len(wids_en)
for key in wids_en:
    dicFile.write(key+","+str(wids_en[key])+"\n")
dicFile.close()
print "Writing EN"

dicFile=open("dictionary_de.txt","w")
print len(wids_de)
for key in wids_en:
    dicFile.write(key+","+str(wids_de[key])+"\n")
dicFile.close()
print "Writing DE"

valid_sentences_en=readData.read_corpus(wids_en,mode="valid",update_dict=False,min_frequency=MIN_EN_FREQUENCY,language="en")
valid_sentences_de=readData.read_corpus(wids_de,mode="valid",update_dict=False,min_frequency=MIN_DE_FREQUENCY,language="de")

train_sentences=zip(train_sentences_de,train_sentences_en)
valid_sentences=zip(valid_sentences_de,valid_sentences_en)

train_sentences=train_sentences[:5000]
valid_sentences=valid_sentences

VOCAB_SIZE_EN=len(wids_en)
VOCAB_SIZE_DE=len(wids_de)

#Specify model
model=dy.Model()

encoder=dy.LSTMBuilder(LAYER_DEPTH,EMB_SIZE,HIDDEN_SIZE,model)
revcoder=dy.LSTMBuilder(LAYER_DEPTH,EMB_SIZE,HIDDEN_SIZE,model)
decoder=dy.LSTMBuilder(LAYER_DEPTH,EMB_SIZE+HIDDEN_SIZE,HIDDEN_SIZE,model)

encoder_params={}
encoder_params["lookup"]=model.add_lookup_parameters((VOCAB_SIZE_DE,EMB_SIZE))

decoder_params={}
decoder_params["lookup"]=model.add_lookup_parameters((VOCAB_SIZE_EN,EMB_SIZE))
decoder_params["R"]=model.add_parameters((VOCAB_SIZE_EN,HIDDEN_SIZE))
decoder_params["bias"]=model.add_parameters((VOCAB_SIZE_EN))

trainer=dy.AdamTrainer(model)

totalSentences=0
for epochId in xrange(NUM_EPOCHS):    
    random.shuffle(train_sentences)
    for sentenceId,sentence in enumerate(train_sentences):
        totalSentences+=1
        sentence_de=sentence[0]
        sentence_en=sentence[1]
        loss,words=do_one_example(model,encoder,revcoder,decoder,encoder_params,decoder_params,sentence_de,sentence_en)
        loss.value()
        loss.backward()
        trainer.update()
        if totalSentences%1000==0:
            random.shuffle(valid_sentences)
            perplexity=0.0
            totalLoss=0.0
            totalWords=0.0
            for valid_sentence in valid_sentences:
                valid_sentence_de=valid_sentence[0]
                valid_sentence_en=valid_sentence[1]
                validLoss,words=do_one_example(model,encoder,revcoder,decoder,encoder_params,decoder_params,valid_sentence_de,valid_sentence_en)
                totalLoss+=float(validLoss.value())
                totalWords+=words
            print totalLoss
            print totalWords
            perplexity=math.exp(totalLoss/totalWords)
            print "Validation perplexity after epoch:",epochId,"sentenceId:",sentenceId,"Perplexity:",perplexity,"Time:",datetime.datetime.now()             
    trainer.update_epoch(1.0)

#train_batch(model,encoder,revcoder,decoder,encoder_params,decoder_params,train_sentences,valid_sentences,NUM_EPOCHS,"Models/"+"Attentional"+"_"+str(HIDDEN_SIZE)+"_"+"Uni")
