''' Multi-label Multi-instance Text Classification Experiments

Script for running multiple text CNN multi-label classification experiments.

Dataset: Openi Chest X-ray raw reports + MeSH term annotations

Preprocessing steps: For text reports, lower-casing, punctuation removal, negated sentence removal, removal of words appearing <5 times, tokenization, padding/cropping, vectorising into list of word ids. For MeSH annotations, lower-casing, removal of words appearing <5 times, one-hot encoding.

Training: MultiLabelTextCNN2 model trained on sub-sampled report-MeSH annotated data, sample size = [100,500,1000,3000, 'all'],
validated on the same 300 validation set.

'''

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("..")
from utils import data_proc_tools as dpt
from utils import plot_tools as pt
from utils.custom_metrics import recall, precision, binary_accuracy
from utils.multi_label_text_models import MultiLabelTextCNN2

dir = '/vol/medic02/users/ag6516/radiology_image_report_generation/'
data_output_dir = dir + 'data/chestx/processed/'
dicts_dir = dir + 'data/chestx/processed/dicts/'

bce_weights = np.linspace(0.1,1,10)
recall_weights = np.linspace(0.1,1,10)
aug = True

total_experiments = len(bce_weights)*len(recall_weights)
counter=0
for sample_size in [100,500,1000,3000,'all']:

    for bce_weight in bce_weights:

        for recall_weight in recall_weights:
            counter+=1
            print('Running text cnn experiment {} of {}'.format(counter, total_experiments))
            print('Sample size: {}, bce_weight: {}, recall_weight: {}'.format(sample_size, bce_weight, recall_weight))
            model_output_dir = dir + 'trained_models/chestx/text_cnn_single_output/train_{}/'.format(sample_size)
            val_captions_df = pd.read_pickle(data_output_dir + 'val/val_300.pkl')
            if sample_size == 'all' and aug:
                train_captions_df = pd.read_pickle(data_output_dir + 'train_{0}/train_{0}_aug.pkl'.format(sample_size))
            else:
                train_captions_df = pd.read_pickle(data_output_dir + 'train_{0}/train_{0}.pkl'.format(sample_size))

            # extract tokenized reports and mesh captions from df
            train_tok_reports_padded = list(train_captions_df.tok_reports_padded)
            train_mesh_captions = list(train_captions_df.reduced_single_mesh)

            val_tok_reports_padded = list(val_captions_df.tok_reports_padded)
            val_mesh_captions = list(val_captions_df.reduced_single_mesh)

            print('Vectorising Sentences and Entities...')
            train_vectoriser = dpt.Vectoriser(data_output_dir+'train_{}/'.format(sample_size), load_dicts=True, dicts_dir=dicts_dir)
            val_vectoriser = dpt.Vectoriser(data_output_dir+'val/', load_dicts=True, dicts_dir=dicts_dir)

            train_vectoriser.entities_to_vectors(train_mesh_captions)
            val_vectoriser.entities_to_vectors(val_mesh_captions)

            train_vectoriser.sentences_to_vectors(train_tok_reports_padded)
            val_vectoriser.sentences_to_vectors(val_tok_reports_padded)

            # dictionaries
            word_to_id = train_vectoriser.word_to_id
            id_to_word = train_vectoriser.id_to_word

            mesh_to_id = train_vectoriser.ent_to_id
            id_to_mesh = train_vectoriser.id_to_ent

            report_vocab_length = len(word_to_id)
            mesh_vocab_length = len(mesh_to_id)

            # get token_ids_array
            train_token_ids_array = train_vectoriser.token_ids_array
            train_mesh_ids_array = train_vectoriser.ents_ids_array

            val_token_ids_array = val_vectoriser.token_ids_array
            val_mesh_ids_array = val_vectoriser.ents_ids_array

            train_mesh_vectors = dpt.one_hot_encode(train_mesh_ids_array, mesh_vocab_length)
            val_mesh_vectors = dpt.one_hot_encode(val_mesh_ids_array, mesh_vocab_length)

            reports_train = train_token_ids_array
            mesh_train = train_mesh_vectors

            reports_val = val_token_ids_array
            mesh_val = val_mesh_vectors

            print('Training size: {} \nVal size: {}'.format(reports_train.shape[0], reports_val.shape[0]))

            input_dim = train_vectoriser.vocab_len
            max_sentence_length = train_vectoriser.max_sen_len
            output_dim = mesh_train.shape[1]
            epochs = 100
            optimizer = 'adam'
            batch_size = 128
            embedding_dim = 512
            feature_maps = [512, 512, 512]
            kernel_sizes = [3,4,5]
            hidden_dim = 256

            new_experiment = MultiLabelTextCNN2(epochs=epochs,
                                           metrics=['accuracy',binary_accuracy,recall,precision],
                                           loss = 'combined_loss',
                                           bce_weight = bce_weight,
                                           recall_weight = recall_weight,
                                           optimizer=optimizer,
                                           batch_size=batch_size, 
                                           embedding_dim=embedding_dim,
                                           feature_maps=feature_maps,
                                           kernel_sizes=kernel_sizes,
                                           hidden_dim=hidden_dim,
                                           input_dim=input_dim, 
                                           max_sentence_length=max_sentence_length, 
                                           output_dim=output_dim, 
                                           verbose=True)
            new_experiment.run_experiment(reports_train, mesh_train, reports_val, mesh_val)
            new_experiment.save_weights_history(model_output_dir)