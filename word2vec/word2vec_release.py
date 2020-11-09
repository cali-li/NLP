import os,sys,re,csv
import pickle
from collections import Counter, defaultdict
import numpy as np
import scipy
import math
import random
import nltk
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
from numba import jit



#... (1) First load in the data source and tokenize into one-hot vectors.
#... Since one-hot vectors are 0 everywhere except for one index, we only need to know that index.


#... (2) Prepare a negative sampling distribution table to draw negative samples from.
#... Consistent with the original word2vec paper, this distribution should be exponentiated.


#... (3) Run a training function for a number of epochs to learn the weights of the hidden layer.
#... This training will occur through backpropagation from the context words down to the source word.

#... (4) Test your model. Compare cosine similarities between learned word vectors.

#... (5) Feel free make minor change to the structure or input of the skeleton code (P.S you still need to finish all the task and implemenent word2vec correctly)










#.................................................................................
#... global variables
#.................................................................................


random.seed(10)
np.random.seed(10)
randcounter = 10
np_randcounter = 10


vocab_size = 0
hidden_size = 100
uniqueWords = [""]                      #... list of all unique tokens
wordcodes = {}                          #... dictionary mapping of words to indices in uniqueWords
wordcounts = Counter()                  #... how many times each token occurs
samplingTable = []                      #... table to draw negative samples from





#.................................................................................
#... load in the data and convert tokens to one-hot indices
#.................................................................................



def loadData(filename):
    global uniqueWords, wordcodes, wordcounts
    override = False
    if override:
        #... for debugging purposes, reloading input file and tokenizing is quite slow
        #...  >> simply reload the completed objects. Instantaneous.
        fullrec = pickle.load(open("w2v_fullrec.p","rb"))
        wordcodes = pickle.load( open("w2v_wordcodes.p","rb"))
        uniqueWords= pickle.load(open("w2v_uniqueWords.p","rb"))
        wordcounts = pickle.load(open("w2v_wordcounts.p","rb"))
        return fullrec

    # ... load in first 15,000 rows of unlabeled data file.  You can load in
    # more if you want later (and should do this for the final homework)
    handle = open(filename, "r", encoding="utf8")
    fullconts = handle.read().split("\n")
    #fullconts = fullconts[1:1500]  # (TASK) Use all the data for the final submission
    #... apply simple tokenization (whitespace and lowercase)
    fullconts = [" ".join(fullconts).lower()]




    print ("Generating token stream...")
    ##############################################################################
    #... (TASK) populate fullrec as one-dimension array of all tokens in the order they appear.
    #... ignore stopwords in this process
    #... for simplicity, you may use nltk.word_tokenize() to split fullconts.
    #... keep track of the frequency counts of tokens in origcounts.

    #fullrec = []
    words = nltk.word_tokenize(fullconts[0])
    fullrec = np.array([word for word in words if word.isalnum()])

    min_count = 50
    origcounts = Counter(fullrec)
    ##############################################################################




    print ("Performing minimum thresholding..")
    
    ##############################################################################
    #... (TASK) populate array fullrec_filtered to include terms as-is that appeared at least min_count times
    #... replace other terms with <UNK> token.
    #... update frequency count of each token in dict wordcounts where: wordcounts[token] = freq(token)
    fullrec_filtered = []
    for i in origcounts:
        if origcounts[i] >= min_count:
            fullrec_filtered.append(i)
        else:
            fullrec_filtered.append("<UNK>")
    fullrec_filtered = np.array(fullrec_filtered)#... fill in
    count = 0
    wordcounts = dict()
    for i in fullrec_filtered:
        if i != "<UNK>":
            wordcounts[i] = origcounts[i]/len(fullrec)
            count += origcounts[i]
    wordcounts["<UNK>"] = (len(fullrec) - count)/len(fullrec)
    ##############################################################################
    #... after filling in fullrec_filtered, replace the original fullrec with this one.
    fullrec = fullrec_filtered


    print ("Producing one-hot indicies")
    ##############################################################################
    #... (TASK) sort the unique tokens into array uniqueWords
    #... produce their one-hot indices in dict wordcodes where wordcodes[token] = onehot_index(token)
    #... replace all word tokens in fullrec with their corresponding one-hot indices.
    uniqueWords = np.unique(fullrec)#... fill in
    wordcodes = dict()#... fill in
    fullrec_onehot = np.empty([len(fullrec),len(uniqueWords)])
    for i in range(len(uniqueWords)):
        wordcodes[uniqueWords[i]] = np.zeros(len(uniqueWords))
        wordcodes[uniqueWords[i]][i] = 1
    for i in range(len(fullrec)):
        fullrec_onehot[i] = wordcodes[fullrec[i]]
    fullrec = fullrec_onehot
    ##############################################################################


    #... close input file handle
    handle.close()



    #... store these objects for later.
    #... for debugging, don't keep re-tokenizing same data in same way.
    #... just reload the already-processed input data with pickles.
    #... NOTE: you have to reload data from scratch if you change the min_count, tokenization or number of input rows
    
    pickle.dump(fullrec, open("w2v_fullrec.p","wb+"))
    pickle.dump(wordcodes, open("w2v_wordcodes.p","wb+"))
    pickle.dump(uniqueWords, open("w2v_uniqueWords.p","wb+"))
    pickle.dump(dict(wordcounts), open("w2v_wordcounts.p","wb+"))


    #... output fullrec should be sequence of tokens, each represented as their one-hot index from wordcodes.
    return fullrec






#.................................................................................
#... compute sigmoid value
#.................................................................................
#@jit
def sigmoid(x):
    return 1.0/(1+np.exp(-x))









#.................................................................................
#... generate a table of cumulative distribution of words
#.................................................................................


def negativeSampleTable(uniqueWords, wordcounts, exp_power=0.75):
    #global wordcounts
    #... stores the normalizing denominator (count of all tokens, each count raised to exp_power)
    max_exp_count = 0


    print ("Generating exponentiated count vectors")
    ##############################################################################
    #... (TASK) for each uniqueWord, compute the frequency of that word to the power of exp_power
    #... store results in exp_count_array.
    exp_count_array = np.array([math.pow(math.exp(wordcounts[i]), exp_power) for i in uniqueWords])#... fill in
    ##############################################################################
    max_exp_count = sum(exp_count_array)



    print ("Generating distribution")
    ##############################################################################
    #... (TASK) compute the normalized probabilities of each term.
    #... using exp_count_array, normalize each value by the total value max_exp_count so that
    #... they all add up to 1. Store this corresponding array in prob_dist
    prob_dist = np.array([exp_count_array[i]/max_exp_count for i in range(len(exp_count_array))])#... fill in
    ##############################################################################



    print ("Filling up sampling table")
    ##############################################################################
    #... (TASK) create a dict of size table_size where each key is a sequential number and its value is a one-hot index
    #... the number of sequential keys containing the same one-hot index should be proportional to its prob_dist value
    #... multiplied by table_size. This table should be stored in cumulative_dict.
    #... we do this for much faster lookup later on when sampling from this table.
    cumulative_dict = dict()#... fill in
    table_size = 1e7
    for i in range(len(prob_dist)):
        repeat = int(prob_dist[i] * table_size)
        length_dict = len(cumulative_dict)
        for j in range(repeat):
            #cumulative_dict[j+length_dict] = np.zeros(len(uniqueWords))
            #cumulative_dict[j+length_dict][i] = 1
            cumulative_dict[j+length_dict] = i
    ##############################################################################


    return cumulative_dict

#.................................................................................
#... generate a specific number of negative samples
#.................................................................................

def generateSamples(context_idx, num_samples):
    global samplingTable, uniqueWords, randcounter
    results = []
    ##############################################################################
    #... (TASK) randomly sample num_samples token indices from samplingTable.
    #... don't allow the chosen token to be context_idx.
    #... append the chosen indices to results
    sample = random.sample(samplingTable.keys(), num_samples)
    results=[wordcodes[uniqueWords[samplingTable[i]]] for i in sample]
    ##############################################################################

    return results



#@jit(nopython=True)
def performDescent(num_samples,center_token,context_index,W1,W2,negative_indices,learning_rate=0.05):
    # sequence chars was generated from the mapped sequence in the core code
    nll_new = 0
    
    ##############################################################################
        #... (TASK) implement gradient descent. Find the current context token from context_words
        #... and the associated negative samples from negative_indices. Run gradient descent on both
        #... weight matrices W1 and W2.
        #... compute the total negative log-likelihood and store this in nll_new.
        #... You don't have to use all the input list above, feel free to change them
    v = np.vstack((context_index ,negative_indices))
    t = np.zeros(num_samples + 1)
    t[0] = 1
    
    tw1 = W1.transpose()
    tw2 = W2.transpose()
    vword = np.dot(tw1,center_token)
    vj = np.dot(tw2,v.transpose()).transpose()
    
    l=[]
    for i in range(len(v)):
        l.append((sigmoid(np.dot(np.dot(tw2,v[i]), vword))-t[i])*np.dot(tw2,v[i]))
    vword += -learning_rate*sum(l)
    #vi += -learning_rate*sum([(sigmoid(sum(np.dot(np.dot(tw2,v[i]), vi)))-t[i])*np.dot(tw2,v[i]) for i in range(len(v))])
    #vi += sum(np.dot(np.dot(tw2,v[0]), vi))
    #vi +=np.dot(tw1,vi)
    for j in range(len(v)):
        vj[j] += -np.multiply(learning_rate*(sigmoid(np.dot(np.dot(tw2,v[j]), vword))-t[j]),vword)
    vj = np.array(vj)
    t1 = -np.ones(num_samples+1)
    t1[0] = 1
    nll_new = -math.log(sum(sigmoid(t1*np.dot(vj,vword))))
    ##############################################################################

    return (nll_new,vword,vj)








#.................................................................................
#... learn the weights for the input-hidden and hidden-output matrices
#.................................................................................


def trainer(curW1 = None, curW2=None):
    global uniqueWords, wordcodes, fullsequence, vocab_size, hidden_size,np_randcounter, randcounter
    vocab_size = len(uniqueWords)           #... unique characters
    hidden_size = 100                       #... number of hidden neurons
    context_window = [-2,-1,1,2]            #... specifies which context indices are output. Indices relative to target word. Don't include index 0 itself.
    nll_results = []                        #... keep array of negative log-likelihood after every 1000 iterations


    #... determine how much of the full sequence we can use while still accommodating the context window
    start_point = int(math.fabs(min(context_window)))
    end_point = len(fullsequence)-(max(max(context_window),0))
    mapped_sequence = fullsequence



    #... initialize the weight matrices. W1 is from input->hidden and W2 is from hidden->output.
    if curW1==None:
        np_randcounter += 1
        W1 = np.random.uniform(-.5, .5, size=(vocab_size, hidden_size))
        W2 = np.random.uniform(-.5, .5, size=(vocab_size, hidden_size))
    else:
        #... initialized from pre-loaded file
        W1 = curW1
        W2 = curW2



    #... set the training parameters
    epochs = 5
    num_samples = 2
    learning_rate = 0.05
    nll = 0
    iternum = 0




    #... Begin actual training
    for j in range(0,epochs):
        print ("Epoch: ", j)
        prevmark = 0

        #... For each epoch, redo the whole sequence...
        for i in range(start_point,end_point):

            if (float(i)/len(mapped_sequence))>=(prevmark+0.1):
                print ("Progress: ", round(prevmark+0.1,1))
                prevmark += 0.1
            if iternum%10000==0:
                print ("Negative likelihood: ", nll)				
                nll_results.append(nll)
                nll = 0

            ###################################################################
            #... (TASK) determine which token is our current input. Remember that we're looping through mapped_sequence
            center_token = fullsequence[i]#... fill in
            if all(center_token == wordcodes["<UNK>"]):
                continue
            else:
                context_index = np.append(fullsequence[i-2:i],fullsequence[i+1:i+3],axis=0)
            #... (TASK) don't allow the center_token to be <UNK>. move to next iteration if you found <UNK>.
            ###################################################################

            iternum += 1
            #... now propagate to each of the context outputs
            #for k in range(0, len(context_window)):
            for k in context_index:

                #... (TASK) Use context_window to find one-hot index of the current context token.
                #context_index = wordcodes[context_words[k]]#[wordcodes[i] for i in context_words]#... fill in



                #... construct some negative samples
                negative_idx = generateSamples(k, num_samples)

                #... (TASK) You have your context token and your negative samples.
                #... Perform gradient descent on both weight matrices.
                #... Also keep track of the negative log-likelihood in variable nll.
                
                ###################################################################
                result = performDescent(num_samples, center_token, k,W1,W2,negative_idx)
                nll = result[0]
                j,=np.where(uniqueWords == center_token)
                W1[j] = result[1]
                j_0,=np.where(k == 1)
                W2[j_0[0]] = result[2][0]
                j_1,=np.where(negative_idx[0] == 1)
                W2[j_1[0]] = result[2][1]
                j_2,=np.where(negative_idx[1] == 1)
                W2[j_2[0]] = result[2][2]
                ###################################################################



    for nll_res in nll_results:
        print (nll_res)
    return [W1,W2]


#.................................................................................
#... Load in a previously-saved model. Loaded model's hidden and vocab size must match current model.
#.................................................................................

def load_model():
    handle = open("saved_W1.data","rb")
    W1 = np.load(handle)
    handle.close()
    handle = open("saved_W2.data","rb")
    W2 = np.load(handle)
    handle.close()
    return [W1,W2]






#.................................................................................
#... Save the current results to an output file. Useful when computation is taking a long time.
#.................................................................................

def save_model(W1,W2):
    handle = open("saved_W1.data","wb+")
    np.save(handle, W1, allow_pickle=False)
    handle.close()

    handle = open("saved_W2.data","wb+")
    np.save(handle, W2, allow_pickle=False)
    handle.close()

word_embeddings = []
proj_embeddings = []



def train_vectors(preload=False):
    global word_embeddings, proj_embeddings
    if preload:
        [curW1, curW2] = load_model()
    else:
        curW1 = None
        curW2 = None
    [word_embeddings, proj_embeddings] = trainer(curW1,curW2)
    save_model(word_embeddings, proj_embeddings)








#.................................................................................
#... for the averaged morphological vector combo, estimate the new form of the target word
#.................................................................................

def morphology(word_seq):
    global word_embeddings, uniqueWords, wordcodes
    embeddings = word_embeddings
    vectors = [word_seq[0], 
               np.dot(embeddings.transpose(), wordcodes[word_seq[1]])]
    vector_math = vectors[0]+vectors[1]
    outputs = []
    value_list = []
    ##############################################################################
    #... find whichever vector is closest to vector_math
    #... (TASK) Use the same approach you used in function prediction() to construct a list
    #... of top 10 most similar words to vector_math. Return this list.
    for i in uniqueWords:
        value = np.dot(word_embeddings.transpose(), wordcodes[i])
        value_list.append(abs(cosine(value, vector_math)))
    top_ten = np.argsort(np.array(value_list))[::-1][:10]
    for i in top_ten:
        outputs.append(uniqueWords[i])
    return(outputs)
    ##############################################################################




#.................................................................................
#... for the triplet (A,B,C) find D such that the analogy A is to B as C is to D is most likely
#.................................................................................

def analogy(word_seq):
    global word_embeddings, proj_embeddings, uniqueWords, wordcodes
    embeddings = word_embeddings
    vectors = [np.dot(word_embeddings.transpose(), wordcodes[word_seq[0]]),
               np.dot(word_embeddings.transpose(), wordcodes[word_seq[1]]),
               np.dot(word_embeddings.transpose(), wordcodes[word_seq[2]])]
    vector_math = -vectors[0] + vectors[1] - vectors[2] # + vectors[3] = 0
    top_ten = []
    value_list =[]
    ##############################################################################
    #... find whichever vector is closest to vector_math
    #... (TASK) Use the same approach you used in function prediction() to construct a list
    #... of top 10 most similar words to vector_math. Return this list.
    for i in uniqueWords:
        value = np.dot(word_embeddings.transpose(), wordcodes[i])
        value_list.append(abs(cosine(value, vector_math)))
    top_ten = np.argsort(np.array(value_list))[::-1][:10]
    for i in top_ten:
        outputs.append(uniqueWords[i])
    return(outputs)
    ##############################################################################







#.................................................................................
#... find top 10 most similar words to a target word
#.................................................................................


def get_neighbors(target_word):
    global word_embeddings, uniqueWords, wordcodes
    #targets = [target_word]
    outputs = []
    value_list = []
    ##############################################################################
    #... (TASK) search through all uniqueWords and for each token, compute its similarity to target_word.
    #... you will compute this using the absolute cosine similarity of the word_embeddings for the word pairs.
    #... Note that the cosine() function from scipy.spatial.distance computes a DISTANCE so you need to convert that to a similarity.
    #... return a list of top 10 most similar words in the form of dicts,
    #... each dict having format: {"word":<token_name>, "score":<cosine_similarity>}
    target_value = np.dot(word_embeddings.transpose(), wordcodes[target_word])
    for i in uniqueWords:
        value = np.dot(word_embeddings.transpose(), wordcodes[i])
        value_list.append(abs(cosine(value, target_value)))
    top_ten = np.argsort(np.array(value_list))[::-1][:10]
    for i in top_ten:
        dic=dict()
        dic["word"] = uniqueWords[i]
        dic["score"] = value_list[i]
        outputs.append(dic)
    return(outputs)
    ##############################################################################








if __name__ == '__main__':
    if len(sys.argv)==2:
        filename = sys.argv[1]  # feel free to raed the file in a different way
        #... load in the file, tokenize it and assign each token an index.
        #... the full sequence of characters is encoded in terms of their one-hot positions

        fullsequence= loadData(filename)
        print ("Full sequence loaded...")
        #print(uniqueWords)
        #print (len(uniqueWords))



        #... now generate the negative sampling table
        print ("Total unique words: ", len(uniqueWords))
        print("Preparing negative sampling table")
        samplingTable = negativeSampleTable(uniqueWords, wordcounts)


        #... we've got the word indices and the sampling table. Begin the training.
        #... NOTE: If you have already trained a model earlier, preload the results (set preload=True) (This would save you a lot of unnecessary time)
        #... If you just want to load an earlier model and NOT perform further training, comment out the train_vectors() line
        #... ... and uncomment the load_model() line

        train_vectors(preload=False)
        #[word_embeddings, proj_embeddings] = load_model()



        #... we've got the trained weight matrices. Now we can do some predictions
        #...pick ten words you choose
        targets = ["good", "bad", "food", "apple",'tasteful','unbelievably','uncle','tool','think']
        for targ in targets:
            print("Target: ", targ)
            bestpreds= (get_neighbors(targ))
            for pred in bestpreds:
                print (pred["word"],":",pred["score"])
            print ("\n")



        #... try an analogy task. The array should have three entries, A,B,C of the format: A is to B as C is to ?
        print (analogy(["apple", "fruit", "banana"]))



        #... try morphological task. Input is averages of vector combinations that use some morphological change.
        #... see how well it predicts the expected target word when using word_embeddings vs proj_embeddings in
        #... the morphology() function.
        #... this is the optional task, if you don't want to finish it, common lines from 545 to 556

        s_suffix = [np.dot(embeddings.transpose(), wordcodes["banana"]) - np.dot(embeddings.transpose(), wordcodes["bananas"])]
        others = [["apples", "apple"],["values", "value"]]
        for rec in others:
            s_suffix.append(np.dot(embeddings.transpose(), wordcodes[word_seq[0]]) - np.dot(embeddings.transpose(), wordcodes[word_seq[1]]))
        s_suffix = np.mean(s_suffix, axis=0)
        print (morphology([s_suffix, "apples"]))
        print (morphology([s_suffix, "pears"]))





        

    else:
        print ("Please provide a valid input filename")
        sys.exit()
