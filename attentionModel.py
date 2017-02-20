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
EMB_SIZE=100
LAYER_DEPTH=1
BATCH_SIZE=16
HIDDEN_SIZE=300
NUM_EPOCHS=20
START=0
UNK=1
STOP=2
GARBAGE=3
MIN_EN_FREQUENCY=5
MIN_DE_FREQUENCY=3


def padBatch(batch):
    batch_length_de=max([len(x[0]) for x in batch])
    batch_length_en=max([len(x[1]) for x in batch])
    batch_de=[x[0] for x in batch]
    batch_en=[x[1] for x in batch]
    lengths_de=[len(x) for x in batch_de]
    lengths_en=[len(x) for x in batch_en]
    #print lengths_en
    batch_de=[[GARBAGE]*(batch_length_de-lengths_de[i])+[START,]+batch_de[i] for i in range(len(batch_de))]
    batch_en=[batch_en[i]+[STOP]*(batch_length_en-lengths_en[i]) for i in range(len(batch_en))]
    #masks_de=[[1]*lengths_de[i]+[0]*(batch_length_de-lengths_de[i]) for i in range(len(batch_de))]
    #masks_en=[[1]*lengths_en[i]+[0]*(batch_length_en-lengths_en[i]) for i in range(len(batch_en))]
    batch_de=np.array(batch_de).T
    batch_en=np.array(batch_en).T
    return batch_de,batch_en

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
   

    #print len(factor_Products)
    #print factor_Products[0].npvalue().shape
    factor_Products=dy.esum(factor_Products)
    return factor_Products

def do_one_batch_attention(model,encoder,decoder,encoder_params,decoder_params,batch_de,batch_en,HIDDEN_SIZE):
    dy.renew_cg()
    encoder_lookup=encoder_params["lookup"]
    decoder_lookup=decoder_params["lookup"]
    R=dy.parameter(decoder_params["R"])
    bias=dy.parameter(decoder_params["bias"])



    s=encoder.initial_state()
    
    inputs=[dy.lookup_batch(encoder_lookup,ys) for ys in batch_de]
    states=s.add_inputs(inputs)
    encoder_outputs=[s.output() for s in states]

    final_encoding_state=encoder_outputs[-1]

    s_init=decoder.initial_state(states[-1].s())
    c_init=dy.vecInput(HIDDEN_SIZE)
    c_init.set(np.zeros(HIDDEN_SIZE))
    
    #print "170"

    s_0=s_init.add_input(dy.concatenate([dy.lookup_batch(decoder_lookup,[START]*batch_en.shape[1]),c_init]))
    o_0=s_0.output()
    alpha_0=dy.softmax(dy.concatenate([dy.dot_product(o_0,encoder_output) for encoder_output in encoder_outputs]))
    c_0=attend(encoder_outputs,alpha_0)


    losses=[]
    
    for ys in batch_en:
        #Calculate loss and append to the losses array TODO: Add context vector in softmax computation too
        scores=R*o_0+bias
        loss=dy.pickneglogsoftmax_batch(scores,ys)
        losses.append(loss)

        #Take in input
        i_t=dy.lookup_batch(decoder_lookup,ys)
        
        s_t=s_0.add_input(dy.concatenate([dy.lookup_batch(decoder_lookup,ys),c_0]))
        o_t=s_t.output()
        alpha_t=dy.softmax(dy.concatenate([dy.dot_product(o_t,encoder_output) for encoder_output in encoder_outputs]))
        c_t=attend(encoder_outputs,alpha_t)

        #Prepare for the next iteration
        s_0=s_t
        o_0=o_t
        c_0=c_t
        alpha_0=alpha_t


    batch_loss=dy.esum(losses)
    final_loss=dy.sum_batches(batch_loss)
    return final_loss

def sample(probs):
    rnd=random.random()
    for i,p in enumerate(probs):
        rnd-=p
        if rnd<=0:
            break
    return i

def greedy_decode_attention(model,encoder,decoder,encoder_params,decoder_params,sentence_de,HIDDEN_SIZE):
    dy.renew_cg()
    encoder_lookup=encoder_params["lookup"]
    decoder_lookup=decoder_params["lookup"]
    R=dy.parameter(decoder_params["R"])
    bias=dy.parameter(decoder_params["bias"])



    s=encoder.initial_state()
    
    inputs=[encoder_lookup[ys] for ys in sentence_de]
    states=s.add_inputs(inputs)
    encoder_outputs=[s.output() for s in states]

    final_encoding_state=encoder_outputs[-1]

    s_init=decoder.initial_state(states[-1].s())
    c_init=dy.vecInput(HIDDEN_SIZE)
    c_init.set(np.zeros(HIDDEN_SIZE))
    
    #print "170"

    s_0=s_init.add_input(dy.concatenate([decoder_lookup[START],c_init]))
    o_0=s_0.output()
    alpha_0=dy.softmax(dy.concatenate([dy.dot_product(o_0,encoder_output) for encoder_output in encoder_outputs]))
    c_0=attend_vector(encoder_outputs,alpha_0)


    totalLoss=0.0
    current_sequence=[]
    current_token=START
    
    while current_token!=STOP and len(current_sequence)<len(sentence_de)+40:
        #Calculate loss and append to the losses array TODO: Add context vector in softmax computation too
        scores=R*o_0+bias
        ys=np.argmax(scores.npvalue())
        loss=dy.pickneglogsoftmax(scores,ys)
        totalLoss+=loss.value()
    
        current_token=ys
        current_sequence.append(current_token)

        #Take in input
        i_t=dy.lookup(decoder_lookup,current_token)
        
        s_t=s_0.add_input(dy.concatenate([decoder_lookup[current_token],c_0]))
        o_t=s_t.output()
        alpha_t=dy.softmax(dy.concatenate([dy.dot_product(o_t,encoder_output) for encoder_output in encoder_outputs]))
        c_t=attend_vector(encoder_outputs,alpha_t)

        #Prepare for the next iteration
        s_0=s_t
        o_0=o_t
        c_0=c_t
        alpha_0=alpha_t


    return current_sequence,final_encoding_state,totalLoss



def strip_stops_and_unks(sequence):
    newSequence=[]
    for word in sequence:
        if word==STOP:
            continue
        else:
            newSequence.append(word)

    return newSequence


def compute_BLEU(references,hypotheses):
    #lengths=[len(x) for x in references]
    #hypotheses=[hypotheses[i] if lengths[i]>=4 for i in range(len(lengths))]
    #references=[[references[i]] if lengths[i]>=4 for i in range(len(lengths))]

    references=[[reference] for reference in references]
    return nltk.translate.bleu_score.corpus_bleu(references,hypotheses)
    

def train_batch(model,encoder,decoder,encoder_params,decoder_params,train_sentences,valid_sentences,NUM_EPOCHS,modelFile):
    trainer=dy.AdamTrainer(model)
    train_sentences.sort(key = lambda x: len(x[1]))
    #print [len(x[1]) for x in train_sentences]
    train_order=[x*BATCH_SIZE for x in range((len(train_sentences)-1)/BATCH_SIZE)]
    valid_sentences.sort(key = lambda x: len(x[1]))
    valid_order=[x*BATCH_SIZE for x in range((len(valid_sentences)-1)/BATCH_SIZE)]


    for epochId in xrange(NUM_EPOCHS):
        random.shuffle(train_order)
        for iter,tidx in enumerate(train_order):
            if (iter*BATCH_SIZE)%3200==0:
                print "Points Covered,",iter*BATCH_SIZE,"Time,",datetime.datetime.now()
            batch=train_sentences[tidx:tidx+BATCH_SIZE]
            batch_de,batch_en=padBatch(batch)
            batch_loss=do_one_batch_attention(model,encoder,decoder,encoder_params,decoder_params,batch_de,batch_en,HIDDEN_SIZE)
            batch_loss.backward()
            trainer.update()
        print "Epoch:",epochId
        trainer.update_epoch(1.0)

        print "Computing Per-Sentence Log Probability"
        totalLoss=0.0
        random.shuffle(valid_order)
        for iter,tidx in enumerate(valid_order):
            batch=valid_sentences[tidx:tidx+BATCH_SIZE]
            batch_de,batch_en=padBatch(batch)
            batch_loss=do_one_batch_attention(model,encoder,decoder,encoder_params,decoder_params,batch_de,batch_en,HIDDEN_SIZE)
            totalLoss+=-sum([x.value() for x in batch_loss])

        totalLoss/=len(valid_sentences)

        print "Total Loss",totalLoss

                
        references=[]
        hypotheses=[]
        dumpFile=open("dump.txt","w")

        for valid_sentence_de,valid_sentence_en in valid_sentences:
            if len(valid_sentence_en)>=5 and len(valid_sentence_de)>=5:
                valid_sentence_en_hat,final_encoding_state,totalLoss=greedy_decode_attention(model,encoder,decoder,encoder_params,decoder_params,valid_sentence_de,HIDDEN_SIZE)
                references.append(strip_stops_and_unks(valid_sentence_en))
                hypotheses.append(strip_stops_and_unks(valid_sentence_en_hat))
                dumpFile.write("Source:"+str(valid_sentence_de)+"\n")
                dumpFile.write("Ref:"+str(valid_sentence_en)+"\n")
                dumpFile.write("Hyp:"+str(valid_sentence_en_hat)+"\n")
                dumpFile.write("Negative Log Loss:"+str(totalLoss)+"\n")
                dumpFile.write("Final Encoding State:"+str(final_encoding_state.npvalue())+"\n")


        dumpFile.close()

        BLEU=compute_BLEU(references,hypotheses)
        print "Corpus Bleu:",BLEU
        

        print "Saving to File"
        model.save(modelFile,[encoder,encoder_params["lookup"],decoder,decoder_params["lookup"],decoder_params["R"],decoder_params["bias"]])

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


train_sentences=train_sentences[:10000]
valid_sentences=valid_sentences[:20000]

VOCAB_SIZE_EN=len(wids_en)
VOCAB_SIZE_DE=len(wids_de)

#Specify model
model=dy.Model()

encoder=dy.LSTMBuilder(LAYER_DEPTH,EMB_SIZE,HIDDEN_SIZE,model)
decoder=dy.LSTMBuilder(LAYER_DEPTH,EMB_SIZE+HIDDEN_SIZE,HIDDEN_SIZE,model)

encoder_params={}
encoder_params["lookup"]=model.add_lookup_parameters((VOCAB_SIZE_DE,EMB_SIZE))

decoder_params={}
decoder_params["lookup"]=model.add_lookup_parameters((VOCAB_SIZE_EN,EMB_SIZE))
decoder_params["R"]=model.add_parameters((VOCAB_SIZE_EN,HIDDEN_SIZE))
#decoder_params["A"]=model.add_parameters((VOCAB_SIZE_EN,HIDDEN_SIZE))
decoder_params["bias"]=model.add_parameters((VOCAB_SIZE_EN))

train_batch(model,encoder,decoder,encoder_params,decoder_params,train_sentences,valid_sentences,NUM_EPOCHS,"Models/"+"Attentional"+"_"+str(HIDDEN_SIZE)+"_"+"Uni")


