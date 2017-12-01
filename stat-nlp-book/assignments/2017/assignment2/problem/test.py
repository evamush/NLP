#! SETUP 1 - DO NOT CHANGE, MOVE NOR COPY
import sys, os
_snlp_book_dir = "../../../../"
sys.path.append(_snlp_book_dir) 
import math
from collections import defaultdict
import statnlpbook.bio as bio

#! SETUP 2 - DO NOT CHANGE, MOVE NOR COPY
train_path = _snlp_book_dir + "data/bionlp/train"
event_corpus = bio.load_assignment2_training_data(train_path)
event_train = event_corpus[:len(event_corpus)//4 * 3]
event_dev = event_corpus[len(event_corpus)//4 * 3:]
assert(len(event_train)==53988)


"""
event_candidate, label = event_corpus[0]
event_candidate.sent
event_candidate.trigger_index
event_candidate.sent.tokens[0]
event_candidate.sent.dependencies[:5]
event_candidate.sent.mentions
event_candidate.sent.parents[0], event_candidate.sent.children[0] 
event_candidate.sent.is_protein[3], event_candidate.sent.is_protein[7]

event_candidate.argument_candidate_spans[:4]
bio.render_event(event_candidate)
bio.render_dependencies(event_candidate.sent)
{y for _,y in event_corpus}
"""
example = event_corpus[398][0]
bio.render_dependencies(example.sent)

#count the words in training dataset

def cal_term_count(dataset):
    term_count = {}
    for event,_ in dataset:
        for token in event.sent.tokens:
            word = token['stem']
            if word not in term_count.keys():
                term_count[word] = 1
            else:
                term_count[word] += 1
    return term_count

#count_dict = cal_term_count(event_train)
#x = sorted(count_dict.items(),key=lambda d:d[1], reverse = True)
#meaningless_words = np.asarray(x)[:5,0]


def add_dependency_child_feats(result, event):
    """
    Append to the `result` dictionary features based on the syntactic dependencies of the event trigger word of
    `event`. The feature keys should have the form "Child: [label]->[word]" where "[label]" is the syntactic label
    of the syntatic child (e.g. "det" in the case above), and "[word]" is the word of the syntactic child (e.g. "The" 
    in the case above).
    Args:
        result: a defaultdict that returns `0.0` by default. 
        event: the event for which we want to populate the `result` dictionary with dependency features.
    Returns:
        Nothing, but populates the `result` dictionary. 
    """
    index = event.trigger_index # You will need to change this 
    for child,label in event.sent.children[index]:
        word = example.sent.tokens[child]['word']
        result["Child: " + label + "->" + word ] += 1.0 
    return result

#! ASSESSMENT 1 - DO NOT CHANGE, MOVE NOR COPY
result = defaultdict(float)
add_dependency_child_feats(result, example)

check_1 = len(result) == 2
check_2 = result['Child: det->The'] == 1.0
check_3 = result['Child: nn->PCR'] == 1.0
print(check_1, check_2, check_3)


from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

result = defaultdict(float)
for x,_ in event_train:
    result['trigger_word=' + x.sent.tokens[x.trigger_index]['word']] += 1.0
#print (result) 

label_encoder = LabelEncoder()
vectorizer = DictVectorizer()
vectorizer.fit_transform([result])[0,10]
#print(len(result)) ---2165
#print(label_encoder.fit_transform([y for _,y in event_train]).max())---9
#print(vectorizer.fit_transform([result]).shape) ---(1, 2165)
#vectorizer.fit_transform([result]).toarray()[:,:10]
#result

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNetCV
import numpy as np
from collections import defaultdict

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

# converts labels into integers, and vice versa, needed by scikit-learn.
label_encoder = LabelEncoder()

# encodes feature dictionaries as numpy vectors, needed by scikit-learn.
vectorizer = DictVectorizer()

def event_feat(event):
    """
    This feature function returns a dictionary representation of the event candidate. You can improve the model 
    by improving this feature function.
    Args:
        event: the `EventCandidate` object to produce a feature dictionary for.
    Returns:
        a dictionary with feature keys/indices mapped to feature counts.
    """
    result = defaultdict(float)
    trigger = event.trigger_index
    tokens =  event.sent.tokens
    children = event.sent.children
    parents = event.sent.parents
    mentions = event.sent.mentions

    result['trigger_word=' + tokens[trigger]['word']] += 1.0
    result['trigger_word=' + tokens[trigger]['stem']] += 1.0
    result['trigger_word=' + tokens[trigger]['pos']] += 1.0

    for child,label in children[trigger]:
        word = tokens[child]['word']
        result["Child: " + label + "->" + word ] += 1.0 
        for grandchild,grandlabel in children[child]:
            grandword = tokens[grandchild]['word']
            result["GrandChild: " + grandlabel + "->" + grandword ] += 1.0 
            
    for parent,label in parents[trigger]:
        word = tokens[parent]['word']
        result["Parents: " + label + "->" + word ] += 1.0 
        for grandparent,grandlabel in parents[parent]:
            grandword = tokens[grandparent]['word']
            result["GrandParent: " + grandlabel + "->" + grandword ] += 1.0 
    

    if tokens[trigger]["pos"]!="IN" :
        word = tokens[trigger]["word"]
        result["trigger is not IN"+word] += 1.0
    elif tokens[trigger]["pos"]=="IN" :
        word = tokens[trigger]["word"]
        result["trigger is IN"+word] += 1.0

    if tokens[trigger+1]["pos"]!="IN" :
        word = tokens[trigger]["word"] + tokens[trigger+1]["word"]
        result["beside trigger is not IN"+word] += 1.0
    elif tokens[trigger+1]["pos"]=="IN" :
        word = tokens[trigger]["word"] + tokens[trigger+1]["word"]
        result["beside trigger is IN"+word] += 1.0

    if tokens[trigger-1]["pos"]!="IN" :
        word = tokens[trigger]["word"] + tokens[trigger-1]["word"]
        result["beside trigger is not IN"+word] += 1.0
    elif tokens[trigger-1]["pos"]=="IN" :
        word = tokens[trigger]["word"] + tokens[trigger-1]["word"]
        result["beside trigger is IN"+word] += 1.0



    
    result["Number of Proteins=" + str(len(mentions))] += 1.0
    min_distance = float("inf")
    for i in range(len(mentions)):
        begin = mentions[i]["begin"]
        end = mentions[i]["end"]
        label = mentions[i]["label"]
        word = tokens[trigger]['stem']
        if (end == trigger) or (end == trigger - 1) or (begin == trigger + 1) or (begin == trigger + 2):
            result["Protein lies near "+ word] += 1.0
        else:
            result["No protein lies near" + word] += 1.0

        if begin > trigger:
            if begin - trigger  <= min_distance:
                min_distance = begin - trigger 
       

        for j in range(begin,end):
            for parent,label in parents[j]:
                if parent == trigger:
                    result["Protein's parent is trigger:" + label] += 1.0
                else:
                    result["Protein's parent is not a trigger:" + label] += 1.0


            for child,label in children[j]:
                if child == trigger:
                    result["Protein's child is trigger:" + label] += 1.0
                else:
                    result["Protein's child is not a trigger:" + label] += 1.0

    result["minimum distance between protein and trigger = " + str(min_distance)] += 1.0
    return result
    


# We convert the event candidates and their labels into vectors and integers, respectively.
train_event_x = vectorizer.fit_transform([event_feat(x) for x,_ in event_train])
train_event_y = label_encoder.fit_transform([y for _,y in event_train])
print(train_event_x.shape)

# Create and train the model. Feel free to experiment with other parameters and learners.


def predict_event_labels(event_candidates):
    """
    This function receives a list of `bio.EventCandidate` objects and predicts their labels. 
    It is currently implemented using scikit-learn, but you are free to replace it with any other
    implementation as long as you fulfil its contract.
    Args:
        event_candidates: A list of `EventCandidate` objects to label.
    Returns:
        a list of event labels, where the i-th label belongs to the i-th event candidate in the input.
    """
    event_x = vectorizer.transform([event_feat(e) for e in event_candidates])
    event_y = label_encoder.inverse_transform(lr.predict(event_x))
    return event_y


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

for i in np.arange(1,5):
    lr = LogisticRegression(C=i,class_weight='balanced')
    #lr = LinearDiscriminantAnalysis()
    lr.fit(train_event_x, train_event_y)



#! ASSESSMENT 2 - DO NOT CHANGE, MOVE NOR COPY
    _snlp_event_test = event_dev # This line will be changed by us after submission to point to a test set.
    _snlp_event_test_guess = predict_event_labels([x for x,_ in _snlp_event_test[:]])
    _snlp_cm_test = bio.create_confusion_matrix(_snlp_event_test,_snlp_event_test_guess)  
    print(i, bio.evaluate(_snlp_cm_test)[2]) # This is the F1 score




#vectorizer.fit_transform([result])[0,10]
#print(len(result))
#result
#label_encoder.fit_transform([y for _,y in event_train]).max()
#print(vectorizer.fit_transform([result]))