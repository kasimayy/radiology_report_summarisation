import numpy as np
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import random
import itertools

class Vectoriser():
    def __init__(self, data_output_dir, load_dicts=False, dicts_dir=None):
        self.output_dir = data_output_dir
        self.load_dicts = load_dicts
        self.dicts_dir = dicts_dir
        
        if self.load_dicts:
            self.word_to_id, self.id_to_word = load_sentence_dicts(self.dicts_dir)
            self.ent_to_id, self.id_to_ent = load_entities_dicts(self.dicts_dir)
        else:
            self.word_to_id = {}
            self.id_to_word = {}
            self.ent_to_id = {}
            self.id_to_ent = {}
            
    def preprocess_entities(self, entities):
        ''' 1. Removing words that appear <5 times
        '''
        
        # determine vocab length and remove words that appear less than 5 times
        all_words = Counter()
        for ents in entities:
#             if isinstance(ents, str):
#                 all_words.update(ents)
#             else:
#                 sen = [s for s in ents]
            all_words.update(ents)

        vocab = [k for k, v in all_words.items() if v >= 5]
        print('Total vocab length: {0}\nVocab length of words>=5: {1}'.format(len(all_words), len(vocab)))
        reduced_ents = []
        for ents in entities:
            reduced_ents.append([word for word in ents if word in vocab])
            
        return reduced_ents
    
    def entities_to_vectors(self, entities, save=False):
#         if self.load_dicts:
#             print('Creating list of entity ids from loaded dictionaries')
        if not self.load_dicts:
            print('Creating new dictionaries and new list of entity ids')
            all_words = Counter()
            for ents in entities:
                all_words.update(ents)

            vocab = set(all_words)
            print('Entities vocab length: {}'.format(len(vocab)))

            # create dictionary of entity to idx and vice versa
            self.ent_to_id = {ent: idx for idx, ent in enumerate(vocab)}

            with open(self.output_dir + 'ent_to_id.pickle', 'wb') as handle:
                pickle.dump(self.ent_to_id, handle, protocol=pickle.HIGHEST_PROTOCOL)

            self.id_to_ent = {idx: ent for idx, ent in enumerate(vocab)}

            with open(self.output_dir + 'id_to_ent.pickle', 'wb') as handle:
                pickle.dump(self.id_to_ent, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # convert entities to list of ids
        ents_ids = [[self.ent_to_id[ent] for ent in ents] for ents in entities]

        self.ents_ids_array = np.array(ents_ids)
        if save:
            np.save(self.output_dir + 'ents_ids_array.npy', self.ents_ids_array)
        else:
            return self.ents_ids_array
    
    def preprocess_sentences(self, text, max_sentence_length=30, padtok='.', stopwords_filepath='stopwords.txt'):
        '''Preprocesses a list of paragraphs
        1. Basic negation removal of sentences which contain words in [' no ', 'not', ' negative ']
        2. Lowercasing and removing punctuation
        3. Removing words that appear <3 times
        4. Removing stopwords
        5. Removing empty strings
        6. Cropping/padding sentences to max_sentence_length with padtok

        Parameters
        ----------
        text: list of paragraphs, where each paragraph is a list of sentences and each sentence is a str
        output_dir: filepath to dir for storing intermediate results
        max_sentence_length: int, max length of sentence vector
        padtok: str, token to pad the sentences to the same length
        stopwords_filename: path to .txt file of stopwords
        '''
        
        # perform basic negation removal
        sentences = []
        neg_list = [' no ', 'not', ' negative ']
        num_sentences = []
        for p in text:
            new_sentences = []
            _sentences = p.split('.')
            for s in _sentences:
                if  all(word not in s.lower() for word in neg_list):
                    new_sentences.append(''.join(s))
            sentences.append(''.join(new_sentences)) # keep sentences separate
            num_sentences.append(len(new_sentences))
            
        print('Avg. number of sentences per report after negation removal: {}'.format(np.mean(num_sentences)))
        
        # tokenize sentences and calculate basic stats
        tok_sentences = [s.split(' ') for s in sentences]
        lengths = [len(s.split(' ')) for s in sentences]
        print('Max sentence length: {}\nMin sentence length: {}\nAvg sentence length: {}\nStd: {}'.format(np.max(lengths),
                                                                                                       np.min(lengths),
                                                                                                       np.mean(lengths),
                                                                                                       np.std(lengths)))

        # lowercase all words, remove punctuation
        l_tok_sentences = []
        for sen in tok_sentences:
            l_tok_sentences.append([w.lower().replace('.','').replace(',','') for w in sen])

        # determine vocab length and remove words that appear less than 3 times
        all_words = Counter()
        for sen in l_tok_sentences:
            sen = [s for s in sen]
            all_words.update(sen)

        vocab = [k for k, v in all_words.items() if v >= 5]
        print('Total vocab length: {0}\nVocab length of words>=5: {1}'.format(len(all_words), len(vocab)))

        # remove stopwords from vocab
        with open(stopwords_filepath, 'r') as file:
            stopwords = file.readlines()

        stopwords = [w.replace('\n', '') for w in stopwords]
        vocab = [w for w in vocab if w not in stopwords]

        # remove tokens in sentences if not in vocab
        r_tok_sentences = []
        for sen in l_tok_sentences:
            r_tok_sentences.append([word for word in sen if word in vocab])

        # remove empty strings
        n_tok_sentences = []
        for sen in r_tok_sentences:
            n_tok_sentences.append(list(filter(None, sen)))

        # new lengths
        new_lengths = [len(s) for s in n_tok_sentences]
        print('Max sentence length: {}\nMin sentence length: {}\nAvg sentence length: {}\nStd: {}'.format(np.max(new_lengths),
                                                                                                           np.min(new_lengths),
                                                                                                           np.mean(new_lengths),
                                                                                                           np.std(new_lengths)))

        # pad sentences
        vocab.append(padtok)
        tok_sentences_padded = [pad_sentence(s, max_sentence_length, padtok) for s in n_tok_sentences]

        self.vocab_len = len(vocab)
        print('Sentences vocab length after stopwords removal: {}'.format(self.vocab_len))
        
        return tok_sentences_padded
    
    def sentences_to_vectors(self, tok_sentences_padded):
        '''Converts tokenised sentences to a list of vectors of word indices. Output is saved as .npy

        Parameters
        ----------
        sentences: list of sentences, where each sentences is a str
        output_dir: filepath to dir for storing intermediate results
        '''
        
        if self.load_dicts:
            print('Creating list of word ids from loaded dictionaries')
        else:
            print('Creating new dictionaries and new list of word ids')
            all_words = Counter()
            for sen in tok_sentences_padded:
                all_words.update(sen)

            vocab = set(all_words)
            print('Sentences vocab length: {}'.format(len(vocab)))
            
            # create dictionary of token to idx and vice versa
            self.word_to_id = {token: idx for idx, token in enumerate(vocab)}

            with open(self.output_dir + 'word_to_id.pickle', 'wb') as handle:
                pickle.dump(self.word_to_id, handle, protocol=pickle.HIGHEST_PROTOCOL)

            self.id_to_word = {idx: token for idx, token in enumerate(vocab)}

            with open(self.output_dir + 'id_to_word.pickle', 'wb') as handle:
                pickle.dump(self.id_to_word, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # convert tokenised sentences to list of ids
        token_ids = [[self.word_to_id[token] for token in tok_sen] for tok_sen in tok_sentences_padded]
        self.token_ids_array = np.array(token_ids)
        self.vocab_len = np.amax(self.token_ids_array)+1
        self.max_sen_len = self.token_ids_array.shape[1]
        np.save(self.output_dir + 'token_ids_array.npy', self.token_ids_array)

    def sample_output(self, samples, padtok='.'):
        for i, (_sentence, _true_ents, _pred_ents) in enumerate(samples):
            print('Sample {}'.format(i))
            sen_str = vector_to_sentence(_sentence, self.id_to_word, padtok)
            print('Sentence: ' + ' '.join(sen_str))
            true_ents_str = vector_to_entities(_true_ents, self.id_to_ent)
            print('True entities: {}'.format(true_ents_str))
            pred_ents_ = np.array([_pred_ents > 0.5])*1.0
            pred_ents_str = vector_to_entities(_pred_ents, self.id_to_ent)
            print('Predicted entities: {}'.format(pred_ents_str))
            print('')

def augment_text(text):
    '''Random shuffles sentences in text
    '''
    sentences = text.split('.')
    random.shuffle(sentences)
    return '.'.join(sentences)
    
            
def load_sentence_dicts(data_dir):
    '''Loads token_ids_array and word-index dictionaries
    '''
    with open(data_dir + 'word_to_id.pickle', 'rb') as handle:
        word_to_id = pickle.load(handle)

    with open(data_dir + 'id_to_word.pickle', 'rb') as handle:
        id_to_word = pickle.load(handle)

    return word_to_id, id_to_word

def load_entities_dicts(data_dir):
    '''Loads ents_ids_array and ent-index dictionaries
    '''
    with open(data_dir + 'ent_to_id.pickle', 'rb') as handle:
        ent_to_id = pickle.load(handle)

    with open(data_dir + 'id_to_ent.pickle', 'rb') as handle:
        id_to_ent = pickle.load(handle)

    return ent_to_id, id_to_ent

def pad_entities(entity, max_len, end_token='end'):
    ent = [end_token for x in range(max_len)]

    if len(entity) > max_len-1:
        cropped_ent = entity[:max_len-1]
        ent[:-1] = cropped_ent
    elif len(entity) < max_len-1:
        ent[:len(entity)] = entity
    else:
        ent[:-1] = entity
    ent=ent[:max_len]
    return ent

def pad_sentence(sentence, max_len, padtok='.'):
    '''Crops and pads sentences to max_len with token 'padtok'
    '''
    sen = [padtok for x in range(max_len)]
    if len(sentence) > max_len:
        sen = sentence[:max_len]
    elif len(sentence) < max_len:
        sen[:len(sentence)] = sentence
    else:
        sen = sentence
    return sen

def one_hot_encode(list_of_idxs, vocab_len):
    '''One-hot-encodes an array of indices into shape (samples, vocab_length)
    '''
    onehot = np.zeros((len(list_of_idxs), vocab_len),dtype='float32')
    for i, idxs in enumerate(list_of_idxs):
        for idx in idxs:
            onehot[i, idx] = 1
    return onehot

def one_hot_sequence(list_of_idxs, vocab_len):
    '''One-hot-encodes an array of indices into shape (samples, seq_length, vocab_length)
    '''
    onehot = np.zeros((list_of_idxs.shape[0], list_of_idxs.shape[1], vocab_len), dtype='float32')
    for i, idxs in enumerate(list_of_idxs):
        for j, idx in enumerate(idxs):
            onehot[i, j, idx] = 1
    return onehot

def vector_to_sentence(sentence_vector, id_to_word, padtok='.'):
    '''Converts a vector of word ids to a vector of strings
    '''
    sentence = []
    for idx in sentence_vector:
        word = id_to_word[idx]
        if word is not padtok:
            sentence.append(word)
    return sentence

def vector_to_entities(entities_vector, id_to_ent):
    '''Converts a vector of one-hot entities to a vector of strings
    '''
    entities = []
    idxs = np.where(entities_vector==1)[0]
    for idx in idxs:
        ent = id_to_ent[idx]
        entities.append(ent)
    return entities

def class_balanced_sample(df, vectoriser, pathology_ids, ent_col, index_col):
    total_classes = len(pathology_ids)
    sample_df = df.sample(2)
    df_entities = list(sample_df[ent_col])
    vectoriser.entities_to_vectors(df_entities)
    
    all_ids=sum(list(vectoriser.ents_ids_array),[])
    all_pathology_ids = [idx for idx in all_ids if idx in pathology_ids]
    
    counter=0
    unique_ids = set(all_pathology_ids)
    old_instances_per_class = 0
    while len(unique_ids)!=total_classes:
        counter+=1
        sample = df.sample(1)
        if int(sample.index.values) in sample_df[index_col]:
            continue

        sample_entities = list(sample[ent_col])
        vectoriser.entities_to_vectors(sample_entities)
        sample_ids = vectoriser.ents_ids_array[0].tolist()
        sample_pathology_ids = [idx for idx in sample_ids if idx in pathology_ids]
        
        try:
            all_pathology_ids.extend(sample_pathology_ids)
        except:
            continue
        unique_ids = set(all_pathology_ids)
        
        if len(unique_ids) == old_instances_per_class:
            continue

        sample_df = sample_df.append(sample)
        old_instances_per_class = len(unique_ids)
    print('total passes: {}'.format(counter))
    
    return sample_df

def batch_generator_seq2seq(sentences, svl, entities, entities_shifted, evl, batch_size):
    '''Yields batches of one-hot-encoded sentences, one-hot-encoded entities and shifted one-hot-encoded-entities
    in the shape (samples, sequence_length, vocab_length)
    
    Parameters
    ----------
    sentences: np.array of sentences as token ids, shape (samples, sequence_length)
    svl: int, sentence vocab length
    entities: np.array of entities as token ids, shape (samples, sequence_length)
    entities_shifted: np.array of time shifted (t-1) entities
    evl: int, entity vocab length
    batch_size: int
    '''
    
    sentence_vocab_length = svl
    entity_vocab_length = evl
    while True:
        for batch in range(0, len(sentences), batch_size):
            yield (one_hot_sequence(sentences[batch:batch+batch_size], sentence_vocab_length),
                  one_hot_sequence(entities[batch:batch+batch_size], entity_vocab_length),
                  one_hot_sequence(entities_shifted[batch:batch+batch_size], entity_vocab_length))
            
def batch_generator2(vec, token_ids_array, entities, batch_size, max_sentence_length):
    '''Yields batches of one-hot encoded sentences and vectorised entities

    Parameters
    ----------
    vec: OneHotEncoder object from sklearn.preprocessing
    token_ids_array: sentences encoded as lists of word ids
    entities: vectorised entities
    batch_size: int
    max_sentence_length: int
    '''
    for cbatch in range(0, len(sentences), batch_size):
        yield (vec.fit_transform(token_ids_array[:batch_size].reshape(batch_size,-1))\
               .toarray().reshape(batch_size, max_sentence_length, -1), 
               entities[cbatch:(cbatch + batch_size)])

def batch_generator3(token_ids_array, entities, batch_size):
    '''Yields batches of sentences encoded as token ids and vectorised entities
    '''
    for cbatch in range(0, len(sentences), batch_size):
        yield (token_ids_array[:batch_size].reshape(batch_size,-1), 
               entities[cbatch:(cbatch + batch_size)])