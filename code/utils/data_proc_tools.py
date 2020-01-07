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
            
def preprocess_reports(text_reports, stopwords_filepath='stopwords.txt'):
    '''Preprocesses a list of reports, where each report is a list of sentences
    1. Basic negation removal of sentences which contain words in [' no ', 'not', ' negative ']
    2. Lowercasing and removing punctuation
    3. Removing words that appear outside 99th percentile
    4. Removing stopwords

    Parameters
    ----------
    text_reports: list of reports where each report is a list of sentences
    stopwords_filename: path to .txt file of stopwords
    '''

    # perform basic negation removal
    _text_reports = []
    neg_list = ['no ', 'not ', 'negative ']
    num_sentences_before_neg = []
    num_sentences_after_neg = []
    for p in text_reports:
        new_sentences = []
        num_sentences_before_neg.append(len(p))
        for s in p:
            if  all(word not in s.lower() for word in neg_list):
                new_sentences.append(s)
        _text_reports.append(new_sentences) # keep sentences separate
        num_sentences_after_neg.append(len(new_sentences))
    print('Avg. number of sentences per report before negation removal: {}'.format(np.mean(num_sentences_before_neg)))
    print('Avg. number of sentences per report after negation removal: {}'.format(np.mean(num_sentences_after_neg)))
    print('Min sentence length: {}\nMax sentence length: {}'.format(np.min(num_sentences_after_neg), np.max(num_sentences_after_neg)))

    # split by words and join sentences with '.'
    s_text_reports = []
    num_words = []
    for p in _text_reports:
        new_p = ' . '.join(p).split(' ')
        num_words.append(len(new_p))
        s_text_reports.append(new_p)
    
    # determine vocab length and word stats
    all_words = Counter()
    for p in s_text_reports:
        all_words.update(p)
        
    # determine 99th percentile cut-off
    cut_off = 0
    cc = 0
    total = sum(list(all_words.values()))
    for k, v in all_words.most_common():
        cc+=v
        percent = cc/total*100
        if percent>99:
            cut_off = v
            break

    vocab = [k for k, v in all_words.items() if v >= cut_off]

    print('Total vocab length: {}\nVocab length of words>={}: {}\n'.format(len(all_words), cut_off, len(vocab)))

    # remove stopwords from vocab
    with open(stopwords_filepath, 'r') as file:
        stopwords = file.readlines()

    stopwords = [w.replace('\n', '') for w in stopwords]
    vocab = [w for w in vocab if w not in stopwords]

    # remove tokens in sentences if not in vocab
    r_text_reports = []
    for p in s_text_reports:
        r_text_reports.append([word for word in p if word in vocab])

    # word + sentence stats after processing
    final_num_words = []
    num_sentences = []
    words_per_sentence = []
    for p in r_text_reports:
        final_num_words.append(len(p))
        report = ' '.join(p)
        sentences = report.split('.')
        num_sentences.append(len(sentences))
        for sen in sentences:
            words = sen.split(' ')
            words_per_sentence.append(len(words))

    print('Average number of sentences per report: {}'.format(np.mean(num_sentences)))
    print('STD number of sentences per report: {}\n'.format(np.std(num_sentences)))
    
    print('Average number of words per sentence: {}'.format(np.mean(words_per_sentence)))
    print('Max number of words per sentence: {}'.format(np.max(words_per_sentence)))
    print('STD number of words per sentence: {}\n'.format(np.std(words_per_sentence))) 
    
    print('Average number of words per exam report: {}'.format(np.mean(final_num_words)))
    print('STD number of words per exam report: {}\n'.format(np.std(final_num_words)))
    
    print('Vocab length after stopwords removal: {}'.format(len(vocab)))

    return r_text_reports

def reports_to_vectors(text_reports, dicts_dir, load_dicts=False, save=False, output_dir=None, unknown_token='**unknown**'):
    '''Converts a list of tokenised reports to a list of vectors of word indices. Output is saved as .npy

    Parameters
    ----------
    reports: list of reports, where each report is a list of str
    load_dicts: load word_to_id and id_to_word dicts if the already exist
    dicts_dir: directory of dicts if they exist
    output_dir: filepath to dir for storing dicts if storing new dicts
    '''

    if load_dicts:
        #print('Creating list of word ids from loaded dictionaries')
        word_to_id, id_to_word = load_report_dicts(dicts_dir)
    else:
        #print('Creating new dictionaries and new list of word ids')
        all_words = Counter()
        for p in text_reports:
            all_words.update(p)

        vocab = list(set(all_words))
        print('Sentences vocab length: {}'.format(len(vocab)))
        
        # create token for words not in dict
        vocab.append(unknown_token)

        # create dictionary of token to idx and vice versa
        word_to_id = {token: idx for idx, token in enumerate(vocab)}

        with open(dicts_dir + 'word_to_id.pickle', 'wb') as handle:
            pickle.dump(word_to_id, handle, protocol=pickle.HIGHEST_PROTOCOL)

        id_to_word = {idx: token for idx, token in enumerate(vocab)}

        with open(dicts_dir + 'id_to_word.pickle', 'wb') as handle:
            pickle.dump(id_to_word, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    # convert tokenised report to list of ids
    token_ids = []
    for p in text_reports:
        p_ids = []
        for token in p:
            if token in word_to_id.keys():
                p_ids.append(word_to_id[token])
            else:
                p_ids.append(word_to_id[unknown_token])
        token_ids.append(p_ids)
    
#     token_ids = [[word_to_id[token] for token in p if token in word_to_id.keys() else word_to_id['**unknown**']] for p in text_reports]
    token_ids_array = np.array(token_ids)
    if save:
        np.save(output_dir + 'token_ids_array.npy', token_ids_array)
    else:
        return token_ids_array

def preprocess_mesh(mesh_captions):
    ''' Calculates caption stats.
    Removes words falling outside 99th percentile.
    '''

    # determine vocab and caption stats
    all_words = Counter()
    num_captions = [] # number of captions per exam
    cap_lengths = [] # word lengths of individual captions
    all_caps_lengths = [] # total word lengths of all captions per exam
    mul_mesh = 0
    normal_cases = 0
    for caption in mesh_captions:
        if 'normal' in caption[0]:
            normal_cases+=1
        if len(caption)>1:
            mul_mesh+=1
        num_captions.append(len(caption))
        all_caps = list(itertools.chain(*caption))
        all_caps_lengths.append(len(all_caps))
        for c in caption:
            sen = [s for s in c]
            cap_lengths.append(len(sen))
            all_words.update(sen)

    print('Stats prior to vocab reduction')
    print('Average number of captions per exam: {}'.format(np.mean(num_captions)))
    print('Average number of terms per caption: {}'.format(np.mean(cap_lengths)))
    print('Average number of terms per exam: {}'.format(np.mean(all_caps_lengths)))
    print('STD of terms per exam: {}'.format(np.std(all_caps_lengths)))
    print('Exams with >1 MeSH annotation: {}'.format(mul_mesh))
    print('Normal vs abnormal cases: normal: {} abnormal: {}'.format(normal_cases, len(mesh_captions)-normal_cases))

    # determine 99th percentile cut-off
    cut_off = 0
    cc = 0
    total = sum(list(all_words.values()))
    for k, v in all_words.most_common():
        cc+=v
        percent = cc/total*100
        if percent>99:
            cut_off = v
            break

    vocab = [k for k, v in all_words.items() if v >= cut_off]

    print('Total vocab length: {}\nVocab length of words>={}: {}'.format(len(all_words), cut_off, len(vocab)))

    reduced_mesh = []
    for caption in mesh_captions:
        red = []
        for c in caption:
            red.append([word for word in c if word in vocab])
        reduced_mesh.append(red)
        
    # determine vocab and caption stats
    all_words = Counter()
    num_captions = [] # number of captions per exam
    cap_lengths = [] # word lengths of individual captions
    all_caps_lengths = [] # total word lengths of all captions per exam
    mul_mesh = 0
    normal_cases = 0
    for caption in reduced_mesh:
        if 'normal' in caption[0]:
            normal_cases+=1
        if len(caption)>1:
            mul_mesh+=1
        num_captions.append(len(caption))
        all_caps = list(itertools.chain(*caption))
        all_caps_lengths.append(len(all_caps))
        for c in caption:
            sen = [s for s in c]
            cap_lengths.append(len(sen))
            all_words.update(sen)

    print('\nStats after to vocab reduction')
    print('Average number of captions per exam: {}'.format(np.mean(num_captions)))
    print('Average number of terms per caption: {}'.format(np.mean(cap_lengths)))
    print('Average number of terms per exam: {}'.format(np.mean(all_caps_lengths)))
    print('STD of terms per exam: {}'.format(np.std(all_caps_lengths)))
    print('Exams with >1 MeSH annotation: {}'.format(mul_mesh))
    print('Normal vs abnormal cases: normal: {} abnormal: {}'.format(normal_cases, len(mesh_captions)-normal_cases))

    # combine mesh into one list of mesh terms
#     _reduced_mesh = []
#     for caption in reduced_mesh:
#         new_caption = list(itertools.chain(*caption))
#         _reduced_mesh.append(new_caption)
    
    return reduced_mesh
 
def mesh_to_vectors(mesh_captions, dicts_dir, load_dicts=False, save=False, output_dir=None, unknown_token='**unknown**'):
    if load_dicts:
        #print('Creating list of mesh ids from loaded dictionaries')
        mesh_to_id, id_to_mesh = load_mesh_dicts(dicts_dir)
    else:
        print('Creating new dictionaries and new list of mesh ids')
        all_words = Counter()
        for caption in mesh_captions:
            all_words.update(caption)

        vocab = list(set(all_words))
        print('MeSH vocab length: {}'.format(len(vocab)))
        
        # create token for mesh terms not in dict
        vocab.append(unknown_token)

        # create dictionary of entity to idx and vice versa
        mesh_to_id = {mesh: idx for idx, mesh in enumerate(vocab)}

        with open(dicts_dir + 'mesh_to_id.pickle', 'wb') as handle:
            pickle.dump(mesh_to_id, handle, protocol=pickle.HIGHEST_PROTOCOL)

        id_to_mesh = {idx: mesh for idx, mesh in enumerate(vocab)}

        with open(dicts_dir + 'id_to_mesh.pickle', 'wb') as handle:
            pickle.dump(id_to_mesh, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # convert mesh captions to list of ids
    
    mesh_ids = []
    for caption in mesh_captions:
        caption_ids = []
        for mesh in caption:
            if mesh in mesh_to_id.keys():
                caption_ids.append(mesh_to_id[mesh])
            else:
                caption_ids.append(mesh_to_id[unknown_token])
        mesh_ids.append(caption_ids)
#     mesh_ids = [[mesh_to_id[mesh] for mesh in caption] for caption in mesh_captions]

    mesh_ids_array = np.array(mesh_ids)
    if save:
        np.save(output_dir + 'mesh_ids_array.npy', mesh_ids_array)
    else:
        return mesh_ids_array

def shuffle_text1(text):
    '''Random shuffles sentences in text
    
    Parameters
    ----------
    text: list of sentences
    '''
    random.shuffle(text)
    return '.'.join(text)

def shuffle_text2(text):
    '''Random shuffles sentences in text
    
    Parameters
    ----------
    text: list of words/tokens and '.'
    '''
    
    s = ' '.join(text).split('.')
    random.shuffle(s)
    new_s = ' .'.join(s).split(' ')
    _new_s = [s for s in new_s if s !='']
    return _new_s
            
def load_report_dicts(data_dir):
    '''Loads word-index dictionaries
    '''
    with open(data_dir + 'word_to_id.pickle', 'rb') as handle:
        word_to_id = pickle.load(handle)

    with open(data_dir + 'id_to_word.pickle', 'rb') as handle:
        id_to_word = pickle.load(handle)

    return word_to_id, id_to_word

def load_mesh_dicts(data_dir):
    '''Loads mesh-index dictionaries
    '''
    with open(data_dir + 'mesh_to_id.pickle', 'rb') as handle:
        mesh_to_id = pickle.load(handle)

    with open(data_dir + 'id_to_mesh.pickle', 'rb') as handle:
        id_to_mesh = pickle.load(handle)

    return mesh_to_id, id_to_mesh

def pad_sequence(sequence, max_len, start_token='start', end_token='end'):
    '''Prepend start_token to sequence, pad sequence to max_len with end_token
    If sequence is more than max_len, crop to max_len-2 and prepend+append start_token+end_token
    '''
    seq = [end_token for x in range(max_len)]
    seq[0] = start_token

    if sequence:
        if len(sequence) > max_len-2:
            seq[1:-1] = sequence[:max_len-2]
        elif len(sequence) <= max_len-2:
            seq[1:len(sequence)+1] = sequence
#     else:
#         print('Sequence empty')
    return seq

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

def vector_to_sentence(sentence_vector, id_to_word, start_token='start', end_token='end', unknown_token='**unknown**'):
    '''Converts a vector of word ids to a vector of strings
    '''
    sentence = []
    for idx in sentence_vector:
        if idx in id_to_word.keys():
            word = id_to_word[idx]
        else:
            word = unknown_token
        if word is not end_token:
            sentence.append(word)
    return sentence

def vector_to_entities(entities_vector, id_to_ent, start_token='start', end_token='end', unknown_token='**unknown**'):
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