from itertools import chain

def read_data(data_path):
    data = open(data_path, 'r').read()
    data = data.split('\n\n')
    tagged_sentences = []

    for sentence in data:
        tagged_sentences.append(list(tuple(word_tag.split('\t')) for word_tag in sentence.split('\n')))
    
    return tagged_sentences

def flatten(y):
    return list(chain.from_iterable(y))