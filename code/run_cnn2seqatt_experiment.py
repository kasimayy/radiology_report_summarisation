import os
import numpy as np
import pandas as pd
from collections import Counter
import itertools
import json
import sys
# sys.path.append("..")
from utils import data_proc_tools as dpt
from utils import plot_tools as pt
from utils.custom_metrics import recall, precision, binary_accuracy
from utils.custom_metrics import recall_np, precision_np, binary_accuracy_np, multilabel_confusion_matrix
from utils.text_sum_models import CNN2SeqAtt
import random
random.seed(42)
random_state=1000

dir = '/vol/medic02/users/ag6516/radiology_report_summarisation/'
data_dir = dir + 'data/'

aug = 'aug'
model_name = 'cnn2seq_att'
model_output_dir = dir + 'trained_models/{}/'.format(model_name)

train_df = pd.read_pickle(data_dir + 'train/{}_train.pkl'.format(aug))
val_df = pd.read_pickle(data_dir + 'val/val.pkl')

# prepend and append start and end tokens to mesh captions and text reports
start_token = 'start'
end_token = 'end'
unknown_token = '**unknown**'
max_mesh_length = 13 # avg. + 1std. + start + end
max_report_length = 37 # avg. + 1std. + start + end

train_df['pad_mesh_caption'] = train_df.all_mesh.apply(lambda x: dpt.pad_sequence(x, max_mesh_length, start_token, end_token))
train_df['pad_text_report'] = train_df.report.apply(lambda x: dpt.pad_sequence(x, max_report_length, start_token, end_token))

val_df['pad_mesh_caption'] = val_df.all_mesh.apply(lambda x: dpt.pad_sequence(x, max_mesh_length, start_token, end_token))
val_df['pad_text_report'] = val_df.report.apply(lambda x: dpt.pad_sequence(x, max_report_length, start_token, end_token))

train_mesh = list(train_df.pad_mesh_caption)
train_reports = list(train_df.pad_text_report)

# vectorize mesh captions
dpt.mesh_to_vectors(train_mesh, dicts_dir=data_dir+'dicts/', 
                    load_dicts=True, save=True, output_dir=data_dir+'train/')

# vectorise reports
dpt.reports_to_vectors(train_reports, dicts_dir=data_dir+'dicts/', 
                       load_dicts=True, save=True, output_dir=data_dir+'train/')
                       
word_to_id, id_to_word = dpt.load_report_dicts(data_dir+'dicts/')
mesh_to_id, id_to_mesh = dpt.load_mesh_dicts(data_dir+'dicts/')

report_vocab_length = len(word_to_id)
mesh_vocab_length = len(mesh_to_id)

# Create arrays of indixes for input sentences, output entities and shifted output entities (t-1)
train_token_ids_array = np.load(data_dir + 'train/token_ids_array.npy')
train_mesh_ids_array = np.load(data_dir + 'train/mesh_ids_array.npy')
train_mesh_ids_array_shifted =[np.concatenate((mesh_to_id[start_token], t[:-1]), axis=None) for t in train_mesh_ids_array]
train_mesh_ids_array_shifted = np.asarray(train_mesh_ids_array_shifted)

val_token_ids_array = np.load(data_dir + 'val/token_ids_array.npy')
val_mesh_ids_array = np.load(data_dir + 'val/mesh_ids_array.npy')
val_mesh_ids_array_shifted = [np.concatenate((mesh_to_id[start_token], t[:-1]), axis=None) for t in val_mesh_ids_array]
val_mesh_ids_array_shifted = np.asarray(val_mesh_ids_array_shifted)

# one-hot-encode
#one_hot_reports_train = dpt.one_hot_sequence(train_token_ids_array, report_vocab_length)
one_hot_mesh_train = dpt.one_hot_sequence(train_mesh_ids_array, mesh_vocab_length)
one_hot_mesh_shifted_train = dpt.one_hot_sequence(train_mesh_ids_array_shifted, mesh_vocab_length)

#one_hot_reports_val = dpt.one_hot_sequence(val_token_ids_array, report_vocab_length)
one_hot_mesh_val = dpt.one_hot_sequence(val_mesh_ids_array, mesh_vocab_length)
one_hot_mesh_shifted_val = dpt.one_hot_sequence(val_mesh_ids_array_shifted, mesh_vocab_length)

input_dim = len(word_to_id)
output_dim = len(mesh_to_id)
embedding_dim = 1024
decoder_hidden_dim = 256
conv_dim = 64
conv1_kernel = 1
conv2_kernel = 3
conv3_kernel = 5
input_seq_length = max_report_length
output_seq_length = max_mesh_length
epochs = 50
optimizer = 'adam'
loss='categorical_crossentropy'
batch_size = 128

experiment = CNN2SeqAtt(epochs=epochs,
                           metrics=['accuracy', binary_accuracy,recall,precision],
                           optimizer=optimizer,
                           loss=loss,
                           batch_size=batch_size, 
                           input_dim=input_dim,
                           output_dim=output_dim,
                           embedding_dim=embedding_dim,
                           decoder_hidden_dim=decoder_hidden_dim,
                           conv_dim=conv_dim,
                           conv1_kernel=conv1_kernel,
                           conv2_kernel=conv2_kernel,
                           conv3_kernel=conv3_kernel,
                           input_seq_length=input_seq_length,
                           output_seq_length=output_seq_length,
                           verbose=True)

experiment.build_model()
experiment.model.summary()

# train
experiment.run_experiment(train_token_ids_array, one_hot_mesh_shifted_train, one_hot_mesh_train, 
                              val_token_ids_array, one_hot_mesh_shifted_val, one_hot_mesh_val)
                              
experiment.save_weights_history(model_output_dir)

# evaluate
# decode a one hot encoded string
def one_hot_decode(encoded_seq):
    return [np.argmax(vector) for vector in encoded_seq]
    
def strip_start_end(seq, start_token='start', end_token='end'):
    stripped_seq = []
    for s in seq:
        if s not in [start_token, end_token]:
            stripped_seq.append(s)
    return stripped_seq 

def predict_sequence(experiment, report_ids, max_seq_len, id_to_mesh, start_token='start', end_token='end'):
    
    in_text = []
    for t in range(max_seq_len-1):
        # pad input

        in_seq = dpt.pad_sequence(in_text, max_seq_len, 
                                  start_token=start_token, 
                                  end_token=end_token)
        #print(in_seq)
        # integer encoder
        in_seq_ids = dpt.mesh_to_vectors([in_seq], dicts_dir=data_dir+'dicts/', 
                       load_dicts=True, save=False)
        #print(in_seq_ids.shape)
        # one-hot encode
        in_seq_onehot = dpt.one_hot_sequence(in_seq_ids, mesh_vocab_length)
        #print(in_seq_onehot.shape)
        
        prediction = experiment.model.predict([report_ids, in_seq_onehot])
        #print(prediction.shape)
        
        prediction = np.argmax(prediction,axis=2)
        #print(prediction)
        mesh = id_to_mesh[prediction[0][t+1]]
        #print(mesh)
        if mesh == end_token:
            break
        in_text.append(mesh)
    return in_text
    
# BLEU
import nltk
from nltk.translate.bleu_score import sentence_bleu
from progress.bar import Bar

def evaluate_model(model, df, report_vocab_length):
    actual, predicted = list(), list()
    bleu1, bleu2, bleu3, bleu4 = list(), list(), list(), list()
    bar = Bar('Processing', max=len(df))
    
    for _, sample in df.iterrows():
        true_mesh_caption = sample.single_mesh
        sample_report = sample.pad_text_report

        sample_report_ids = []
        for token in sample_report:
            if token in word_to_id.keys():
                sample_report_ids.append(word_to_id[token])
            else:
                sample_report_ids.append(word_to_id[unknown_token])

        sample_report_ids = np.array(sample_report_ids).reshape(1, len(sample_report_ids))

        predicted_mesh = predict_sequence(model, 
                                  sample_report_ids, 
                                  max_mesh_length, 
                                  id_to_mesh,
                                  start_token=start_token,
                                  end_token=end_token)

        # sample_report = strip_start_end(sample_report)
        yhat = strip_start_end(predicted_mesh)
        reference = true_mesh_caption
        
        # calculate BLEU score
        bleu1.append(sentence_bleu([reference], yhat, weights=(1.0, 0, 0, 0)))
        bleu2.append(sentence_bleu([reference], yhat, weights=(0.5, 0.5, 0, 0)))
        bleu3.append(sentence_bleu([reference], yhat, weights=(0.3, 0.3, 0.3, 0)))
        bleu4.append(sentence_bleu([reference], yhat, weights=(0.25, 0.25, 0.25, 0.25)))
    
        # store actual and predicted
        actual.append(reference)
        predicted.append(yhat)
        
        bar.next()
        
    print('\nBLEU1: ', np.mean(bleu1)*100)
    print('BLEU2: ', np.mean(bleu2)*100)
    print('BLEU3: ', np.mean(bleu3)*100)
    print('BLEU4: ', np.mean(bleu4)*100)
    bar.finish()
    return actual, predicted
 
print('BLEU Train\n')
train_actual, train_predicted = evaluate_model(experiment, train_df.sample(1000), report_vocab_length)
print('BLEU Val\n')
val_actual, val_predicted = evaluate_model(experiment, val_df, report_vocab_length)

# ROUGE
import rouge
evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                       max_n=4,
                       limit_length=True,
                       length_limit=100,
                       length_limit_type='words',
                       apply_avg='Avg',
                       apply_best='Best',
                       alpha=0.5, # Default F1_score
                       weight_factor=1.2,
                       stemming=True)
                       
def prepare_results(p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)
    
train_hypotheses = [' '.join(p) for p in train_predicted]
train_references = [' '.join(a) for a in train_actual]

scores = evaluator.get_scores(train_hypotheses, train_references)

print('ROUGE Train\n')
for metric, results in sorted(scores.items(), key=lambda x: x[0]):
    print(prepare_results(results['p'], results['r'], results['f']))
    
val_hypotheses = [' '.join(p) for p in val_predicted]
val_references = [' '.join(a) for a in val_actual]

scores = evaluator.get_scores(val_hypotheses, val_references)

print('ROUGE Val\n')
for metric, results in sorted(scores.items(), key=lambda x: x[0]):
    print(prepare_results(results['p'], results['r'], results['f']))