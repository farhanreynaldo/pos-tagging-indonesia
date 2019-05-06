import re
from sklearn_crfsuite import CRF
from joblib import dump, load
from sklearn.model_selection import train_test_split 

def prepare_train_test(tagged_sentences, seed, path):
    train_set, test_set = train_test_split(tagged_sentences, test_size=0.2, random_state=seed)
    prefix, suffix = select_params(path)
    X_train, y_train = prepare_data(train_set, prefix=prefix, suffix=suffix)
    X_test, y_test = prepare_data(test_set, prefix=prefix, suffix=suffix)
    return X_train, y_train, X_test, y_test

def features(sentence, index, prefix=True, suffix=True):
    ### sentence is of the form [w1,w2,w3,..], index is the position of the word in the sentence
    ### and entriestoRemove type is tuple
    d = {
        'is_first_capital':int(sentence[index][0].isupper()),
        'is_first_word': int(index == 0),
        'is_last_word':int(index == len(sentence)-1),
        'is_complete_capital': int(sentence[index].upper() == sentence[index]),
        'prev_word':'' if index == 0 else sentence[index-1],
        'next_word':'' if index == len(sentence)-1 else sentence[index+1],
        'is_numeric':int(sentence[index].isdigit()),
        'is_alphanumeric': int(bool((re.match('^(?=.*[0-9]$)(?=.*[a-zA-Z])', sentence[index])))),
        'word_has_hyphen': 1 if '-' in sentence[index] else 0 
         }
    
    if prefix:
        d.update({'prefix_1':sentence[index][0],
                 'prefix_2': sentence[index][:2],
                 'prefix_3':sentence[index][:3],
                 'prefix_4':sentence[index][:4]})
    
    if suffix:
        d.update({'suffix_1':sentence[index][-1],
                  'suffix_2':sentence[index][-2:],
                  'suffix_3':sentence[index][-3:],
                  'suffix_4':sentence[index][-4:]})
        
    return d
    
def untag(sentence):
      return [word for word, tag in sentence]

def prepare_data(tagged_sentences, prefix, suffix):
    X, y = [], []
    for sentences in tagged_sentences:
        X.append([features(untag(sentences), index, prefix, suffix) for index in range(len(sentences))])
        y.append([tag for word, tag in sentences])
    return X, y

def fit_and_dump(X_train, y_train, path):
    crf = CRF(
        algorithm='lbfgs',
        c1=0.01,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
        )
    crf.fit(X_train, y_train)
    dump(crf, path)
    return crf

def select_params(path):
    if 'baseline' in str(path):
        prefix, suffix = False, False
    elif 'prefix' in str(path):
        prefix, suffix = True, False
    elif 'suffix' in str(path):
        prefix, suffix = False, True
    else:
        prefix, suffix = True, True
    return prefix, suffix