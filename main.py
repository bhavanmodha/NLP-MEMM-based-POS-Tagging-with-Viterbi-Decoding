import sys

import nltk
from nltk.corpus import brown
import numpy
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
# Load the Brown corpus with Universal Dependencies tags
# proportion is a float
# Returns a tuple of lists (sents, tags)
def load_training_corpus(proportion=1.0):
    brown_sentences = brown.tagged_sents(tagset='universal')
    num_used = int(proportion * len(brown_sentences))

    corpus_sents, corpus_tags = [None] * num_used, [None] * num_used
    for i in range(num_used):
        corpus_sents[i], corpus_tags[i] = zip(*brown_sentences[i])
    print("Train Load Done")
    return (corpus_sents, corpus_tags)


# Generate word n-gram features
# words is a list of strings
# i is an int
# Returns a list of strings
def get_ngram_features(words, i):
    n = len(words)-1

    prevbigram = 'prevbigram-'+("<s>" if (i-1)<0 else words[i-1])
    nextbigram = 'nextbigram-'+("</s>" if (i+1)>n else words[i+1])
    prevskip = 'prevskip-'+("<s>" if (i-2)<0 else words[i-2])
    nextskip = 'nextskip-'+("</s>" if (i+2)>n else words[i+2])
    prevtrigram = 'prevtrigram-'+("<s>" if (i-1)<0 else words[i-1])+'-'+("<s>" if (i-2)<0 else words[i-2])
    nexttrigram = 'nexttrigram-'+("</s>" if (i+1)>n else words[i+1])+'-'+("</s>" if (i+2)>n else words[i+2])
    centertrigram = 'centertrigram-'+("<s>" if (i-1)<0 else words[i-1])+'-'+("</s>" if (i+1)>n else words[i+1])
    print("get_ngram")

    return [prevbigram,nextbigram,prevskip,nextskip,prevtrigram,nexttrigram,centertrigram]


# Generate word-based features
# word is a string
# returns a list of strings
def get_word_features(word):
    ans = []
    capital = num = hyphen = 0
    all_caps = 1
    c=0
    wordshape=""
    s_wordshape = ""
    last = ""
    for i in word:
        if i.isupper():
            capital = 1
            c+=1
            wordshape+="X"
            if last != "X":
                s_wordshape += "X"
            last = "X"
        elif i.islower():
            c+=1
            all_caps = 0
            wordshape+="x"
            if last != "x":
                s_wordshape += "x"
            last = "x"
        elif i.isdigit():
            c+=1
            num = 1
            wordshape+='d'
            if last != "d":
                s_wordshape += "d"
            last = "d"
        else:
            if i == '-':
                hyphen = 1 
            wordshape+=i
            if last!=i:
                s_wordshape+=i
            last = i
    ans.append('word-'+word)
    if capital==1:
        ans.append("capital")
    if word.isupper():
        ans.append("allcaps") 
    if c>0:   
        ans.append("wordshape-"+wordshape)
        ans.append("short-wordshape-"+s_wordshape)
    if num ==1 :
        ans.append("number")
    if hyphen==1:
        ans.append("hyphen")
    
    if len(word)>=4:
        n=4
    else:
        n=len(word)

    for i in range(n):
        ans.append("prefix"+ str(i+1) +"-" + word[0:i+1])
    for i in range(n):
        ans.append("suffix"+ str(i+1) +"-" + word[-(i+1):])

    return ans
    



# Wrapper function for get_ngram_features and get_word_features
# words is a list of strings
# i is an int
# prevtag is a string
# Returns a list of strings
def get_features(words, i, prevtag):
    ngram_features = get_ngram_features(words, i)
    word_features = get_word_features(words[i])
    ngram_features = [word.lower() for word in ngram_features]
    for i in range(len(word_features)):
        if word_features[i].startswith("wordshape") or word_features[i].startswith("short"):
            continue
        word_features[i] = word_features[i].lower()
        print('get_features')
    return ngram_features+word_features+['tagbigram-'+prevtag.lower()]


# Remove features that occur fewer than a given threshold number of time
# corpus_features is a list of lists, where each sublist corresponds to a sentence and has elements that are lists of strings (feature names)
# threshold is an int
# Returns a tuple (corpus_features, common_features)
def remove_rare_features(corpus_features, threshold=5):
    features_count = {}
    new = [[] for _ in range(len(corpus_features))]
    for i in corpus_features:
        for j in i:
            for k in j:
                if k in features_count:
                    features_count[k]+=1
                else:
                    features_count[k]=1
    
    for i in range(len(corpus_features)):
        for j in range(len(corpus_features[i])):
            new[i].append([])
            for k in range(len(corpus_features[i][j])):
                if features_count[corpus_features[i][j][k]] >= threshold:
                    new[i][j].append(corpus_features[i][j][k])
    
    common_features = [features for features in features_count if features_count[features]>=threshold]
    print('remove rare')
    return (new,common_features)




# Build feature and tag dictionaries
# common_features is a set of strings
# corpus_tags is a list of lists of strings (tags)
# Returns a tuple (feature_dict, tag_dict)
def get_feature_and_label_dictionaries(common_features, corpus_tags):
    iter = 0
    feature_dict = {}
    tag_dict = {}
    for feature in common_features:
        feature_dict[feature] = iter
        iter += 1
    iter = 0
    for sent in corpus_tags:
        for tag in sent:
            if tag not in tag_dict:
                tag_dict[tag] = iter
                iter += 1
                if iter == 12:
                    return(feature_dict, tag_dict)
    print('get_dict')
    return(feature_dict, tag_dict)

# Build the label vector Y
# corpus_tags is a list of lists of strings (tags)
# tag_dict is a dictionary {string: int}
# Returns a Numpy array
def build_Y(corpus_tags, tag_dict):
    result = []
    for sent in corpus_tags:
        for tag in sent:
            result.append(tag_dict[tag])
    print('build_Y')
    return numpy.array(result)

# Build a sparse input matrix X
# corpus_features is a list of lists, where each sublist corresponds to a sentence and has elements that are lists of strings (feature names)
# feature_dict is a dictionary {string: int}
# Returns a Scipy.sparse csr_matrix
def build_X(corpus_features, feature_dict):
    rows = []
    cols = []
    iter = 0
    for i in range(len(corpus_features)):
        for j in range(len(corpus_features[i])):
            for k in range(len(corpus_features[i][j])):
                if corpus_features[i][j][k] in feature_dict:
                    rows.append(iter)
                    cols.append(feature_dict[corpus_features[i][j][k]])
            iter += 1
    val = numpy.ones(len(cols))
    rows = numpy.array(rows)
    cols = numpy.array(cols)
    return csr_matrix((val, (rows, cols)),shape = (iter, len(feature_dict)))



# Train an MEMM tagger on the Brown corpus
# proportion is a float
# Returns a tuple (model, feature_dict, tag_dict)
def train(proportion=1.0):
    corpus_sents, tags = load_training_corpus(0.1)
    features = [[] for _ in range(len(corpus_sents))]
    
    for i in range(len(corpus_sents)):
        for j in range(len(corpus_sents[i])):
            prev = tags[i][j-1] if (j-1)>=0 else "<s>"
            features[i].append(get_features(corpus_sents[i], j, prev))
    
    corpus_features, common_features = remove_rare_features(features)
    feature_dict, tag_dict = get_feature_and_label_dictionaries(common_features, tags)
    X = build_X(corpus_features, feature_dict)
    Y = build_Y(tags, tag_dict)
 
    lr = LogisticRegression(class_weight="balanced", solver="saga", multi_class="multinomial")
    lr.fit(X,Y)
    print('Train')
    return (lr, feature_dict, tag_dict)

# Load the test set
# corpus_path is a string
# Returns a list of lists of strings (words)
def load_test_corpus(corpus_path):
    with open(corpus_path) as inf:
        lines = [line.strip().split() for line in inf]
    print('Load Test')
    return [line for line in lines if len(line) > 0]


# Predict tags for a test sentence
# test_sent is a list containing a single list of strings
# model is a trained LogisticRegression
# feature_dict is a dictionary {string: int}
# reverse_tag_dict is a dictionary {int: string}
# Returns a tuple (Y_start, Y_pred)
def get_predictions(test_sent, model, feature_dict, reverse_tag_dict):
    y_pred = numpy.empty(((len(test_sent[0])-1),len(reverse_tag_dict),len(reverse_tag_dict)))
    # print(y_pred.shape)
    for i in range(1, len(test_sent[0])):
        features = []
        for prev_tag in reverse_tag_dict.values(): 
            features.append(get_features(test_sent[0],i,prev_tag))
        X = build_X([features],feature_dict)
        temp_y =  model.predict_log_proba(X)
        y_pred[i-1] = temp_y
    first_word = get_features(test_sent[0],0,"<s>")
    X = build_X([[first_word]], feature_dict)
    y_start = numpy.array(model.predict_log_proba(X))
    return(y_start,y_pred)


# Perform Viterbi decoding using predicted log probabilities
# Y_start is a Numpy array of size (1, T)
# Y_pred is a Numpy array of size (n-1, T, T)
# Returns a list of strings (tags)
def viterbi(Y_start, Y_pred):
    n = len(Y_pred)+1
    t = len(Y_pred[0])
    V = numpy.empty((n,t))
    BP = numpy.empty((n,t), dtype=int)

    V[0] = Y_start[0] 

    for i in range(1,n):
        for tag in range(t):
                V[i, tag] = max([V[i-1, prev_tag] + Y_pred[i-1, prev_tag, tag] for prev_tag in range(t)])
                BP[i,tag] = int(numpy.argmax(numpy.array([V[i-1, prev_tag] + Y_pred[i-1, prev_tag, tag] for prev_tag in range(t)])))

    ans=[0]*n
    ans[-1] = int(numpy.argmax(V[-1]))

    for i in range(n-2,-1,-1):
        ans[i] = BP[i+1, ans[i+1]]

    # print("V: ", V.shape)
    # print("BP: ", BP.shape)
    # print(len(ans))
    return ans
    

    
# Predict tags for a test corpus using a trained model
# corpus_path is a string
# model is a trained LogisticRegression
# feature_dict is a dictionary {string: int}
# tag_dict is a dictionary {string: int}
# Returns a list of lists of strings (tags)
def predict(corpus_path, model, feature_dict, tag_dict):
    test_sents = load_test_corpus(corpus_path)
    reverse_tag_dict = {}
    ans = []
    for tag, index in tag_dict.items():
        reverse_tag_dict[index] = tag
    for sent in test_sents:
        y_start, y_pred = get_predictions([sent], model, feature_dict, reverse_tag_dict)
        vit = viterbi(y_start, y_pred)
        # print("vit: ", vit)
        vit_ans = []
        for i in vit:
            vit_ans.append(reverse_tag_dict[i])
        # print("vit_ans: ", vit_ans)
        ans.append(vit_ans)
    return ans

def main(args):
    model, feature_dict, tag_dict = train(0.09)

    predictions = predict('test.txt', model, feature_dict, tag_dict)
    for test_sent in predictions:
        print(test_sent)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
