import re

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

def prepare_data(tagged_sentences, prefix=True, suffix=True):
    X, y = [], []
    for sentences in tagged_sentences:
        X.append([features(untag(sentences), index, prefix, suffix) for index in range(len(sentences))])
        y.append([tag for word, tag in sentences])
    return X, y