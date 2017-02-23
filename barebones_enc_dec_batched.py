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
import pickle

#Config Definition
EMB_SIZE=256
LAYER_DEPTH=1
BATCH_SIZE=32
HIDDEN_SIZE=400
NUM_EPOCHS=5
START=0
UNK=1
STOP=2
GARBAGE=3
MIN_EN_FREQUENCY=1
MIN_DE_FREQUENCY=1
MAX_TRAIN_SENTENCES=40000

def greedyDecode(model,encoder,revcoder,decoder,encoder_params,decoder_params,sentence_de,reverse_dict):
    dy.renew_cg()
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
    sentence_en=[]
    currentToken=None

    while currentToken!=STOP and len(sentence_en)<2*len(sentence_de)+10:
        #Calculate loss and append to the losses array
        #scores=R*o_0+bias
        scores=dy.affine_transform([bias,R,o_0])
        currentToken=np.argmax(scores.npvalue())
        loss=dy.pickneglogsoftmax(scores,currentToken)
        losses.append(loss)
        sentence_en.append(currentToken)

        #Take in input
        i_t=dy.concatenate([dy.lookup(decoder_lookup,currentToken),c_0])
        s_t=s_0.add_input(i_t)
        o_t=s_t.output()
        alpha_t=dy.softmax(dy.concatenate([dy.dot_product(o_t,final_combined_output) for final_combined_output in final_combined_outputs]))
        c_t=attend_vector(final_combined_outputs,alpha_t)
        
        #Prepare for the next iteration
        s_0=s_t
        o_0=o_t
        c_0=c_t
        alpha_0=alpha_t

    loss=dy.esum(losses)

    interpreted_sentence_en=" ".join([reverse_dict[en] for en in sentence_en])
    return sentence_en,interpreted_sentence_en,loss

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
        #scores=R*o_0+bias
        scores=dy.affine_transform([bias,R,o_0])

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

def do_one_batch(model,encoder,revcoder,decoder,encoder_params,decoder_params,sentence_de,sentence_en):
    dy.renew_cg()
    
    #Assumption is that all given sentences have the same source length
    sentence_de_forward=[[START,]+x for x in sentence_de]
    sentence_de_reverse=[x[::-1]+[START,] for x in sentence_de]
    sentence_de_forward=np.array(sentence_de_forward)
    sentence_de_reverse=np.array(sentence_de_reverse)
    sentence_de_forward=sentence_de_forward.T
    sentence_de_reverse=sentence_de_reverse.T

    #print sentence_de_forward.shape
    #print sentence_de_reverse.shape

    length_en=[len(x) for x in sentence_en]
    max_length_en=max(length_en)
    sentence_en=[sentence_en[i]+[STOP,]*(max_length_en-length_en[i]) for i in range(len(length_en))]
    mask_en=[[1.0,]*length_en[i]+[0.0,]*(max_length_en-length_en[i]) for i in range(len(length_en))]
    sentence_en=np.array(sentence_en)
    mask_en=np.array(mask_en)
    sentence_en=sentence_en.T
    mask_en=mask_en.T

    #print sentence_en.shape
    #print mask_en.shape

    total_words=np.sum(mask_en)
    CURRENT_BATCH_SIZE=mask_en.shape[1]

    encoder_lookup=encoder_params["lookup"]
    decoder_lookup=decoder_params["lookup"]
    R=dy.parameter(decoder_params["R"])
    bias=dy.parameter(decoder_params["bias"])


    s=encoder.initial_state()
    inputs=[dy.lookup_batch(encoder_lookup,de) for de in sentence_de_forward]
    states=s.add_inputs(inputs)
    encoder_outputs=[s.output() for s in states]

    s_reverse=revcoder.initial_state()
    inputs=[dy.lookup_batch(encoder_lookup,de) for de in sentence_de_reverse]
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
    c_init=attend(final_combined_outputs,alpha_init)

    
    s_0=s_init
    o_0=o_init
    alpha_0=alpha_init
    c_0=c_init
    

    losses=[]
    
    for en,mask in zip(sentence_en,mask_en):
        #Calculate loss and append to the losses array
        #scores=R*o_0+bias
        #scores=dy.affine_transform([bias,R,o_0])
        scores=dy.affine_transform([bias,R,o_0])

        loss=dy.pickneglogsoftmax_batch(scores,en)
        
        mask_expr=dy.inputVector(mask)
        mask_expr=dy.reshape(mask_expr,(1,),CURRENT_BATCH_SIZE)

        #mask=mask.reshape((1,CURRENT_BATCH_SIZE))

        losses.append(loss*mask_expr)
        
        #Reshaping debug
        #print loss.npvalue().shape
        #print reshapedMaskInput.npvalue().shape        
        
        #Take in input
        i_t=dy.concatenate([dy.lookup_batch(decoder_lookup,en),c_0])
        s_t=s_0.add_input(i_t)
        o_t=s_t.output()
        alpha_t=dy.softmax(dy.concatenate([dy.dot_product(o_t,final_combined_output) for final_combined_output in final_combined_outputs]))
        c_t=attend(final_combined_outputs,alpha_t)
        
        #Prepare for the next iteration
        s_0=s_t
        o_0=o_t
        c_0=c_t
        alpha_0=alpha_t

    
    total_loss=dy.sum_batches(dy.esum(losses))
    return total_loss,total_words

   
def main():
    # Read in data
    wids_en=defaultdict(lambda: len(wids_en))
    wids_de=defaultdict(lambda: len(wids_de))

    train_sentences_en=readData.read_corpus(wids_en,mode="train",update_dict=True,min_frequency=MIN_EN_FREQUENCY,language="en")
    train_sentences_de=readData.read_corpus(wids_de,mode="train",update_dict=True,min_frequency=MIN_DE_FREQUENCY,language="de")

    enDictionaryFile="Models/"+"en-dict_"+str(MIN_EN_FREQUENCY)+".txt" 
    deDictionaryFile="Models/"+"de-dict_"+str(MIN_DE_FREQUENCY)+".txt"

    dicFile=open(enDictionaryFile,"w")
    print len(wids_en)
    for key in wids_en:
        dicFile.write(key+" "+str(wids_en[key])+"\n")
    dicFile.close()
    print "Writing EN"

    dicFile=open(deDictionaryFile,"w")
    print len(wids_de)
    for key in wids_en:
        dicFile.write(key+" "+str(wids_de[key])+"\n")
    dicFile.close()
    print "Writing DE"


    valid_sentences_en=readData.read_corpus(wids_en,mode="valid",update_dict=False,min_frequency=MIN_EN_FREQUENCY,language="en")
    valid_sentences_de=readData.read_corpus(wids_de,mode="valid",update_dict=False,min_frequency=MIN_DE_FREQUENCY,language="de")

    train_sentences=zip(train_sentences_de,train_sentences_en)
    valid_sentences=zip(valid_sentences_de,valid_sentences_en)

    train_sentences=train_sentences[:MAX_TRAIN_SENTENCES]
    valid_sentences=valid_sentences

    print "Number of Training Sentences:",len(train_sentences)
    print "Number of Validation Sentences:",len(valid_sentences)


    VOCAB_SIZE_EN=len(wids_en)
    VOCAB_SIZE_DE=len(wids_de)

    random.shuffle(train_sentences)
    random.shuffle(valid_sentences)

    #Prepare batches
    lengthMap={}
    for x in train_sentences:
        if len(x[0]) not in lengthMap:
            lengthMap[len(x[0])]=[]
        lengthMap[len(x[0])].append(x)

    print "Number of Different Lengths:",len(lengthMap)

    train_batches=[]

    for megaBatch in lengthMap.values():
        index=0
        while index<len(megaBatch):
            if index%BATCH_SIZE==0:
                batch=megaBatch[index:min(index+BATCH_SIZE,len(megaBatch))]
                train_batches.append(batch)
                index+=BATCH_SIZE

    print [len(batch) for batch in train_batches]
    print sum([len(batch) for batch in train_batches])

    #Free some memory.Dump useless references
    train_sentences=None
    train_sentences_en=None
    train_sentences_de=None

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
    sentencesCovered=totalSentences/3200

    startTime=datetime.datetime.now()
    print "Start Time",startTime
    for epochId in xrange(NUM_EPOCHS):    
        random.shuffle(train_batches)
        for batchId,batch in enumerate(train_batches):
            if len(batch)>1:
                totalSentences+=len(batch)
                if totalSentences/3200>sentencesCovered:
                    sentencesCovered=totalSentences/3200
                    print "Sentences covered:",totalSentences,"Current Time",datetime.datetime.now()
                sentence_de=[sentence[0] for sentence in batch]
                sentence_en=[sentence[1] for sentence in batch]
                loss,words=do_one_batch(model,encoder,revcoder,decoder,encoder_params,decoder_params,sentence_de,sentence_en)
                loss.value()
                loss.backward()
                trainer.update()
            else:
                totalSentences+=1
                #print "Sentences covered:",totalSentences
                sentence=batch[0]
                sentence_de=sentence[0]
                sentence_en=sentence[1]
                loss,words=do_one_example(model,encoder,revcoder,decoder,encoder_params,decoder_params,sentence_de,sentence_en)
                loss.value()
                loss.backward()
                trainer.update()
            #if totalSentences%1000<20:
            #    print "Total Sentences Covered:",totalSentences

        
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
        print "Validation perplexity after epoch:",epochId,"Perplexity:",perplexity,"Time:",datetime.datetime.now()             
        
        trainer.update_epoch(1.0)
        
        #Save Model
        modelFile="Models/"+"barebones_enc_dec_batched"+"_"+str(datetime.datetime.now())+"_"+str(EMB_SIZE)+"_"+str(LAYER_DEPTH)+"_"+str(HIDDEN_SIZE)+"_"+str(MIN_EN_FREQUENCY)+"_"+str(MIN_DE_FREQUENCY)
        model.save(modelFile,[encoder,revcoder,decoder,encoder_params["lookup"],decoder_params["lookup"],decoder_params["R"],decoder_params["bias"]])

    return wids_de,wids_en,modelFile

def reverseDictionary(dictionary):
    reverse_dictionary={}
    for key,value in dictionary.items():
        reverse_dictionary[value]=key
    return reverse_dictionary

def computePerplexity(model,encoder,revcoder,decoder,encoder_params,decoder_params,valid_sentences):
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
    return perplexity 

def metaMain(modelFile=None,wids_de=None,wids_en=None):

    if modelFile==None:
        wids_de,wids_en,modelFile=main()
    
    model=dy.Model()
    encoder_params={}
    decoder_params={}
    (encoder,revcoder,decoder,encoder_params["lookup"],decoder_params["lookup"],decoder_params["R"],decoder_params["bias"])=model.load(modelFile)

    print "Reversing dictionaries"
    #Reverse dictionaries
    reverse_wids_en=reverseDictionary(wids_en)
    reverse_wids_de=reverseDictionary(wids_de)

    print "Reading Test Data"
    test_sentences_en=readData.read_corpus(wids_en,mode="test",update_dict=False,min_frequency=MIN_EN_FREQUENCY,language="en")
    test_sentences_de=readData.read_corpus(wids_de,mode="test",update_dict=False,min_frequency=MIN_DE_FREQUENCY,language="de")
    test_sentences=zip(test_sentences_de,test_sentences_en)

    print "Reading blind German"
    blind_sentences_de=readData.read_corpus(wids_de,mode="blind",update_dict=False,min_frequency=MIN_EN_FREQUENCY,language="de")

    testPerplexity=computePerplexity(model,encoder,revcoder,decoder,encoder_params,decoder_params,test_sentences)
    print "Test perplexity,",testPerplexity
    
    outFileName=modelFile+"_testOutput"+"_"+str(datetime.datetime.now())
    refFileName=modelFile+"_testRef"+"_"+str(datetime.datetime.now())
    outFile=open(outFileName,"w")
    refFile=open(refFileName,"w")
    bleuOutputFile=modelFile+"_BLEU"

    print "Decoding Test Sentences"
    for test_sentence in test_sentences:
        sentence_en_hat,interpreted_test_sentence_en_hat,loss=greedyDecode(model,encoder,revcoder,decoder,encoder_params,decoder_params,test_sentence[0],reverse_wids_en)
        #print interpreted_test_sentence_en_hat
        outFile.write(interpreted_test_sentence_en_hat+"\n")
        interpreted_test_sentence_en=" ".join([reverse_wids_en[x] for x in test_sentence[1]])
        refFile.write(interpreted_test_sentence_en+"\n")

    outFile.close()
    refFile.close()

    print "wrote Data"
    print "Computing perplexity"
    #import shlex
    #import subprocess
    #subprocess.call(["perl","multi-bleu.perl","-lc",refFileName,"<",outFileName],stdout=stdout)
    print "Over"
    return modelFile

if __name__=="__main__":
    modelFile=None

    if modelFile!=None:
        wids_en=defaultdict(lambda: len(wids_en))
        wids_de=defaultdict(lambda: len(wids_en))

        train_sentences_en=readData.read_corpus(wids_en,mode="train",update_dict=True,min_frequency=MIN_EN_FREQUENCY,language="en")
        train_sentences_de=readData.read_corpus(wids_de,mode="train",update_dict=True,min_frequency=MIN_DE_FREQUENCY,language="de")
        train_sentences_en=None
        train_sentences_de=None
        metaMain(modelFile=modelFile,wids_de=wids_de,wids_en=wids_en)

    else:
        modelFile=metaMain()
        print "Model File Name:",modelFile
