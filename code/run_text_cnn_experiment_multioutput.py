''' Multi-label Multi-instance Text Classification Experiments
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
from utils.custom_metrics import recall_np, precision_np, binary_accuracy_np, multilabel_confusion_matrix
from utils.multi_label_text_models import MultiLabelMultiOutputTextCNN
random_state=1000
# import tensorflow as tf
# tf.keras.backend.clear_session()
# tf.reset_default_graph()

dir = '/vol/medic02/users/ag6516/image_sentence_mapping/'
#dir = '/vol/biomedic2/users/ag6516/'
data_dir = 'data/chestx/'

ce_weights = np.linspace(0.1,1,10)
recall_weights = np.linspace(0.1,1,10)

total_experiments = len(ce_weights)*len(recall_weights)
counter=0
data_output_dir = dir + 'data/chestx/processed_balanced/'
dicts_dir = dir + 'data/chestx/processed_balanced/dicts/'
aug = True

PATHOLOGY_VOCAB_LEN = 41
ANATOMY_VOCAB_LEN = 27
POSITION_VOCAB_LEN = 6
SEVERITY_VOCAB_LEN = 28

for sample_size in ['all']: #[100, 500, 1000, 3000, 

    for ce_weight in [1.0]:#ce_weights:

        for recall_weight in [0.0]:#recall_weights:
            counter+=1
            print('Running text cnn experiment {} of {}'.format(counter, total_experiments))
            print('Sample size: {}, ce_weight: {}, recall_weight: {}'.format(sample_size, ce_weight, recall_weight))
            model_output_dir = dir + 'trained_models/chestx/text_cnn_multi_output/train_{}/'.format(sample_size)
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
            
            split_mesh_train = [mesh_train[:, 0:PATHOLOGY_VOCAB_LEN], 
                    mesh_train[:, PATHOLOGY_VOCAB_LEN:PATHOLOGY_VOCAB_LEN+ANATOMY_VOCAB_LEN],
                    mesh_train[:, PATHOLOGY_VOCAB_LEN+ANATOMY_VOCAB_LEN:PATHOLOGY_VOCAB_LEN+ANATOMY_VOCAB_LEN+POSITION_VOCAB_LEN],
                    mesh_train[:, PATHOLOGY_VOCAB_LEN+ANATOMY_VOCAB_LEN+POSITION_VOCAB_LEN:]]

            split_mesh_val = [mesh_val[:, 0:PATHOLOGY_VOCAB_LEN], 
                    mesh_val[:, PATHOLOGY_VOCAB_LEN:PATHOLOGY_VOCAB_LEN+ANATOMY_VOCAB_LEN],
                    mesh_val[:, PATHOLOGY_VOCAB_LEN+ANATOMY_VOCAB_LEN:PATHOLOGY_VOCAB_LEN+ANATOMY_VOCAB_LEN+POSITION_VOCAB_LEN],
                    mesh_val[:, PATHOLOGY_VOCAB_LEN+ANATOMY_VOCAB_LEN+POSITION_VOCAB_LEN:]]

            print('Training size: {} \nVal size: {}'.format(reports_train.shape[0], reports_val.shape[0]))

            input_dim = train_vectoriser.vocab_len
            max_sentence_length = train_vectoriser.max_sen_len
            #output_dim = mesh_train.shape[1]
            dense_output_dims = [PATHOLOGY_VOCAB_LEN, ANATOMY_VOCAB_LEN, POSITION_VOCAB_LEN, SEVERITY_VOCAB_LEN]
            loss_weights = {'pathology': 0.6, 'anatomy': 0.2, 'position': 0.1, 'severity': 0.1}
            output_dim = [124, 124, 124, 124]
            epochs = 100
            optimizer = 'adam'
            batch_size = 128
            embedding_dim = 512
            conv_dim1 = 512
            conv_dim2 = 0
            fc_dim = 0

            new_experiment = MultiLabelMultiOutputTextCNN(
                                           dense_output_dims=dense_output_dims,
                                           loss='categorical_crossentropy',
                                           loss_weights=loss_weights,
                                           epochs=epochs,
                                           metrics=['acc', binary_accuracy,recall,precision],
                                           optimizer=optimizer,
                                           batch_size=batch_size, 
                                           embedding_dim=embedding_dim, 
                                           conv_dim1=conv_dim1, 
                                           conv_dim2=conv_dim2,
                                           fc_dim=fc_dim,
                                           input_dim=input_dim, 
                                           max_sentence_length=max_sentence_length, 
                                           output_dim=output_dim, 
                                           verbose=True)
            new_experiment.run_experiment(reports_train, split_mesh_train, reports_val, split_mesh_val)          
            new_experiment.save_weights_history(model_output_dir)