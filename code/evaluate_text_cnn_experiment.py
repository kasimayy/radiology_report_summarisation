''' Evaluation of Multi-label Multi-instance Text Classification Experiments

Script to evaluate experiments made by run_text_cnn_experiment.py

1. Load sub-sampled report-MeSH annotated data of sample size = [100, 500, 1000, 3000, 'all']
2. Load single-output or multi-output model trained on specific sub-sample
3. Make MeSH predictions on remaining training set (unless loading model trained on 'all')
4. Evaluate and save training/validation accuracy, precision and recall over all MeSH labels
5. Evaluate and save training/validation accuracy, precision and recall split by pathology/anatomy/position/severity

'''

import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import sys
import math
sys.path.append("..")
from utils import data_proc_tools as dpt
from utils import plot_tools as pt
from utils.multi_label_text_models import MultiLabelTextCNN2
from utils.multi_label_text_models import MultiLabelMultiOutputTextCNN
from utils.custom_metrics import recall_np, precision_np, binary_accuracy_np, multilabel_confusion_matrix

import random
random.seed(42)
np.random.seed(42)
random_state=1000

dir = '/vol/medic02/users/ag6516/radiology_image_report_generation/'
data_dir = dir + 'data/chestx/'
data_type = 'processed_balanced'
data_output_dir = dir + 'data/chestx/processed/'
dicts_dir = dir + 'data/chestx/processed/dicts/'

aug = True
output_type = 'single' # multi or single

PATHOLOGY_VOCAB_LEN = 41
ANATOMY_VOCAB_LEN = 27
POSITION_VOCAB_LEN = 6
SEVERITY_VOCAB_LEN = 28

pvl = PATHOLOGY_VOCAB_LEN
avl = ANATOMY_VOCAB_LEN
povl = POSITION_VOCAB_LEN
svl = SEVERITY_VOCAB_LEN
    
for sample_size in [100, 500, 1000, 3000, 'all']: 
    
    print('Evaluating experiment for sample_size {}'.format(sample_size))
    model_output_dir = dir + 'trained_models/chestx/text_cnn_{}_output/train_{}/'.format(output_type, sample_size)

    if sample_size == 'all' and aug:
        train_df = pd.read_pickle(data_output_dir + 'train_{0}/train_{0}_aug.pkl'.format(sample_size))
    else:
        train_df = pd.read_pickle(data_output_dir + 'train_{0}/train_{0}.pkl'.format(sample_size))

    print('Evaluating on training size: {}'.format(train_df.index.nunique()))

    val_df = pd.read_pickle(data_output_dir + 'val/val_300.pkl')
    print('Evaluating on validation size: {}'.format(val_df.index.nunique()))

    # initialise vectoriser
    train_vectoriser = dpt.Vectoriser(data_output_dir+'train_{}/'.format(sample_size), load_dicts=True, dicts_dir=dicts_dir)
    val_vectoriser = dpt.Vectoriser(data_output_dir+'val/', load_dicts=True, dicts_dir=dicts_dir)
    test_vectoriser = dpt.Vectoriser(data_output_dir+'test/', load_dicts=True, dicts_dir=dicts_dir)
    
    # extract tokenized sentences and entities from df
    train_tok_reports_padded = list(train_df.tok_reports_padded)
    train_mesh_captions = list(train_df.reduced_single_mesh)

    val_tok_reports_padded = list(val_df.tok_reports_padded)
    val_mesh_captions = list(val_df.reduced_single_mesh)
    
    test_tok_reports_padded = list(test_df.tok_reports_padded)
    test_mesh_captions = list(test_df.reduced_single_mesh)
    
    # vectorize mesh captions
    train_vectoriser.entities_to_vectors(train_mesh_captions, save=True)
    val_vectoriser.entities_to_vectors(val_mesh_captions, save=True)
    test_vectoriser.entities_to_vectors(test_mesh_captions, save=True)

    # vectorise reports
    train_vectoriser.sentences_to_vectors(train_tok_reports_padded)
    val_vectoriser.sentences_to_vectors(val_tok_reports_padded)
    test_vectoriser.sentences_to_vectors(test_tok_reports_padded)
    
    # get token_ids_array and one-hot-encode mesh captions
    train_token_ids_array = train_vectoriser.token_ids_array
    train_mesh_ids_array = train_vectoriser.ents_ids_array

    val_token_ids_array = val_vectoriser.token_ids_array
    val_mesh_ids_array = val_vectoriser.ents_ids_array
    
    test_token_ids_array = test_vectoriser.token_ids_array
    test_mesh_ids_array = test_vectoriser.ents_ids_array

    word_to_id = train_vectoriser.word_to_id
    id_to_word = train_vectoriser.id_to_word

    mesh_to_id = train_vectoriser.ent_to_id
    id_to_mesh = train_vectoriser.id_to_ent

    train_mesh_vectors = dpt.one_hot_encode(train_mesh_ids_array, len(mesh_to_id))
    val_mesh_vectors = dpt.one_hot_encode(val_mesh_ids_array, len(mesh_to_id))
    test_mesh_vectors = dpt.one_hot_encode(test_mesh_ids_array, len(mesh_to_id))
    
    reports_train = train_token_ids_array
    mesh_train = train_mesh_vectors

    reports_val = val_token_ids_array
    mesh_val = val_mesh_vectors
    
    reports_test = test_token_ids_array
    mesh_test = test_mesh_vectors
    
    if output_type == 'multi':
        split_mesh_train = [mesh_train[:, 0:pvl], 
                    mesh_train[:, pvl:pvl+avl],
                    mesh_train[:, pvl+avl:pvl+avl+povl],
                    mesh_train[:, pvl+avl+povl:]]

        split_mesh_val = [mesh_val[:, 0:pvl], 
                  mesh_val[:, pvl:pvl+avl],
                  mesh_val[:, pvl+avl:pvl+avl+povl],
                  mesh_val[:, pvl+avl+povl:]]
        
        split_mesh_test = [mesh_test[:, 0:pvl], 
                  mesh_test[:, pvl:pvl+avl],
                  mesh_test[:, pvl+avl:pvl+avl+povl],
                  mesh_test[:, pvl+avl+povl:]]

    if output_type == 'multi':
        loss_name = 'categorical_crossentropy'
        dense_output_dims = [pvl, avl, povl, svl]
        loss_weights = {'pathology': 0.6, 'anatomy': 0.2, 'position': 0.1, 'severity': 0.1}
        output_dim = [124, 124, 124, 124]
        epochs = 100
        optimizer = 'adam'
        batch_size = 128
        embedding_dim = 512
        conv_dim1 = 512
        conv_dim2 = 0
        fc_dim = 0
        param_fn = 'param_cnn_{}_epochs_{}_convdim1_{}_convdim2_{}_fc_dim_{}_embeddingdim_{}_loss_weights.json'\
        .format(epochs, conv_dim1, conv_dim2, fc_dim, embedding_dim, [float(x) for x in loss_weights.values()])
        params = json.load(open(model_output_dir + param_fn, 'r'))
        old_experiment = MultiLabelMultiOutputTextCNN(**params)
    elif output_type == 'single':
        epochs = 100
        conv_dim = 512
        embedding_dim = 512
        loss_name = 'combined_loss'
        bce_weight = 0.5
        recall_weight = 0.4
        feature_maps = [512, 512, 512]
        kernel_sizes = [3,4,5]
        hidden_dim = 256
        param_fn = 'param_cnn_{}_epochs_{}_kernel_sizes_{}_feature_maps_{}_embeddingdim_{}_loss_{:.1f}_bce_{:.1f}_recall.json'\
        .format(epochs, kernel_sizes, feature_maps, embedding_dim, loss_name, bce_weight, recall_weight)
        params = json.load(open(model_output_dir + param_fn, 'r'))
        old_experiment = MultiLabelTextCNN2(**params)
    else:
        print('Set output type')

    old_experiment.build_model()
    old_experiment.load_weights_history(model_output_dir)

    # make predictions
    __pred_mesh_val = old_experiment.model.predict(reports_val)
    __pred_mesh_train = old_experiment.model.predict(reports_train)
    if output_type == 'multi':      
        _pred_mesh_val = np.concatenate(__pred_mesh_val, axis=1)
        _pred_mesh_train = np.concatenate(__pred_mesh_train, axis=1)
    elif output_type == 'single':
        _pred_mesh_val = __pred_mesh_val
        _pred_mesh_train = __pred_mesh_train
    pred_mesh_val = np.array([_pred_mesh_val > 0.5])*1.0
    pred_mesh_val = pred_mesh_val[0]
    pred_mesh_train = np.array([_pred_mesh_train > 0.5])*1.0
    pred_mesh_train = pred_mesh_train[0]

    # evaluate and save metrics
    metrics_table = pd.DataFrame()
    metrics_dict = {}
    metrics_dict['model'] = output_type + '_output'
    metrics_dict['dataset'] = 'chestx'
    metrics_dict['training_size'] = sample_size
    metrics_dict['epochs'] = epochs
    metrics_dict['loss_name'] = loss_name

    avg_binary_accuracy = binary_accuracy_np(mesh_train, pred_mesh_train)
    metrics_dict['train_bin_acc'] = avg_binary_accuracy*100
    avg_binary_accuracy = binary_accuracy_np(mesh_val, pred_mesh_val)
    metrics_dict['val_bin_acc'] = avg_binary_accuracy*100

    recall, avg_class, avg_sample = recall_np(mesh_train, pred_mesh_train)
    metrics_dict['train_recall'] = recall*100
    metrics_dict['train_recall_per_class'] = avg_class*100
    metrics_dict['train_recall_per_sample'] = avg_sample*100
    recall, avg_class, avg_sample = recall_np(mesh_val, pred_mesh_val)
    metrics_dict['val_recall'] = recall*100
    metrics_dict['val_recall_per_class'] = avg_class*100
    metrics_dict['val_recall_per_sample'] = avg_sample*100

    precision, avg_class, avg_sample = precision_np(mesh_train, pred_mesh_train)
    metrics_dict['train_prec'] = precision*100
    metrics_dict['train_prec_per_class'] = avg_class*100
    metrics_dict['train_prec_per_sample'] = avg_sample*100
    precision, avg_class, avg_sample = precision_np(mesh_val, pred_mesh_val)
    metrics_dict['val_prec'] = precision*100
    metrics_dict['val_prec_per_class'] = avg_class*100
    metrics_dict['val_prec_per_sample'] = avg_sample*100

    metrics_table = metrics_table.append(metrics_dict, ignore_index=True)
    
    # metrics for pathologies only
    metrics_dict_pathology = {}
    metrics_dict_pathology['model'] = output_type + '_output'
    metrics_dict_pathology['dataset'] = 'chestx_pathology'
    metrics_dict_pathology['training_size'] = sample_size
    metrics_dict_pathology['epochs'] = epochs
    metrics_dict_pathology['loss_name'] = loss_name
    mesh_train_p = mesh_train[:,0:pvl]
    pred_mesh_train_p = pred_mesh_train[:,0:pvl]
    mesh_val_p = mesh_val[:,0:pvl]
    pred_mesh_val_p = pred_mesh_val[:,0:pvl]

    avg_binary_accuracy = binary_accuracy_np(mesh_train_p, pred_mesh_train_p)
    metrics_dict_pathology['train_bin_acc'] = avg_binary_accuracy*100
    avg_binary_accuracy = binary_accuracy_np(mesh_val_p, pred_mesh_val_p)
    metrics_dict_pathology['val_bin_acc'] = avg_binary_accuracy*100

    recall, avg_class, avg_sample = recall_np(mesh_train_p, pred_mesh_train_p)
    metrics_dict_pathology['train_recall'] = recall*100
    metrics_dict_pathology['train_recall_per_class'] = avg_class*100
    metrics_dict_pathology['train_recall_per_sample'] = avg_sample*100
    recall, avg_class, avg_sample = recall_np(mesh_val_p, pred_mesh_val_p)
    metrics_dict_pathology['val_recall'] = recall*100
    metrics_dict_pathology['val_recall_per_class'] = avg_class*100
    metrics_dict_pathology['val_recall_per_sample'] = avg_sample*100

    precision, avg_class, avg_sample = precision_np(mesh_train_p, pred_mesh_train_p)
    metrics_dict_pathology['train_prec'] = precision*100
    metrics_dict_pathology['train_prec_per_class'] = avg_class*100
    metrics_dict_pathology['train_prec_per_sample'] = avg_sample*100
    precision, avg_class, avg_sample = precision_np(mesh_val_p, pred_mesh_val_p)
    metrics_dict_pathology['val_prec'] = precision*100
    metrics_dict_pathology['val_prec_per_class'] = avg_class*100
    metrics_dict_pathology['val_prec_per_sample'] = avg_sample*100
    
    metrics_table = metrics_table.append(metrics_dict_pathology, ignore_index=True)
    
    # metrics for anatomy only
    metrics_dict_anatomy = {}
    metrics_dict_anatomy['model'] = output_type + '_output'
    metrics_dict_anatomy['dataset'] = 'chestx_anatomy'
    metrics_dict_anatomy['training_size'] = sample_size
    metrics_dict_anatomy['epochs'] = epochs
    metrics_dict_anatomy['loss_name'] = loss_name
    mesh_train_p = mesh_train[:,pvl:pvl+avl]
    pred_mesh_train_p = pred_mesh_train[:,pvl:pvl+avl]
    mesh_val_p = mesh_val[:,pvl:pvl+avl]
    pred_mesh_val_p = pred_mesh_val[:,pvl:pvl+avl]
    
    avg_binary_accuracy = binary_accuracy_np(mesh_train_p, pred_mesh_train_p)
    metrics_dict_anatomy['train_bin_acc'] = avg_binary_accuracy*100
    avg_binary_accuracy = binary_accuracy_np(mesh_val_p, pred_mesh_val_p)
    metrics_dict_anatomy['val_bin_acc'] = avg_binary_accuracy*100

    recall, avg_class, avg_sample = recall_np(mesh_train_p, pred_mesh_train_p)
    metrics_dict_anatomy['train_recall'] = recall*100
    metrics_dict_anatomy['train_recall_per_class'] = avg_class*100
    metrics_dict_anatomy['train_recall_per_sample'] = avg_sample*100
    recall, avg_class, avg_sample = recall_np(mesh_val_p, pred_mesh_val_p)
    metrics_dict_anatomy['val_recall'] = recall*100
    metrics_dict_anatomy['val_recall_per_class'] = avg_class*100
    metrics_dict_anatomy['val_recall_per_sample'] = avg_sample*100

    precision, avg_class, avg_sample = precision_np(mesh_train_p, pred_mesh_train_p)
    metrics_dict_anatomy['train_prec'] = precision*100
    metrics_dict_anatomy['train_prec_per_class'] = avg_class*100
    metrics_dict_anatomy['train_prec_per_sample'] = avg_sample*100
    precision, avg_class, avg_sample = precision_np(mesh_val_p, pred_mesh_val_p)
    metrics_dict_anatomy['val_prec'] = precision*100
    metrics_dict_anatomy['val_prec_per_class'] = avg_class*100
    metrics_dict_anatomy['val_prec_per_sample'] = avg_sample*100
        
    metrics_table = metrics_table.append(metrics_dict_anatomy, ignore_index=True)

    # metrics for position only
    metrics_dict_position = {}
    metrics_dict_position['model'] =  output_type + '_output'
    metrics_dict_position['dataset'] = 'chestx_position'
    metrics_dict_position['training_size'] = sample_size
    metrics_dict_position['epochs'] = epochs
    metrics_dict_position['loss_name'] = loss_name
    mesh_train_p = mesh_train[:,pvl+avl:pvl+avl+povl]
    pred_mesh_train_p = pred_mesh_train[:,pvl+avl:pvl+avl+povl]
    mesh_val_p = mesh_val[:,pvl+avl:pvl+avl+povl]
    pred_mesh_val_p = pred_mesh_val[:,pvl+avl:pvl+avl+povl]
    
    avg_binary_accuracy = binary_accuracy_np(mesh_train_p, pred_mesh_train_p)
    metrics_dict_position['train_bin_acc'] = avg_binary_accuracy*100
    avg_binary_accuracy = binary_accuracy_np(mesh_val_p, pred_mesh_val_p)
    metrics_dict_position['val_bin_acc'] = avg_binary_accuracy*100

    recall, avg_class, avg_sample = recall_np(mesh_train_p, pred_mesh_train_p)
    metrics_dict_position['train_recall'] = recall*100
    metrics_dict_position['train_recall_per_class'] = avg_class*100
    metrics_dict_position['train_recall_per_sample'] = avg_sample*100
    recall, avg_class, avg_sample = recall_np(mesh_val_p, pred_mesh_val_p)
    metrics_dict_position['val_recall'] = recall*100
    metrics_dict_position['val_recall_per_class'] = avg_class*100
    metrics_dict_position['val_recall_per_sample'] = avg_sample*100

    precision, avg_class, avg_sample = precision_np(mesh_train_p, pred_mesh_train_p)
    metrics_dict_position['train_prec'] = precision*100
    metrics_dict_position['train_prec_per_class'] = avg_class*100
    metrics_dict_position['train_prec_per_sample'] = avg_sample*100
    precision, avg_class, avg_sample = precision_np(mesh_val_p, pred_mesh_val_p)
    metrics_dict_position['val_prec'] = precision*100
    metrics_dict_position['val_prec_per_class'] = avg_class*100
    metrics_dict_position['val_prec_per_sample'] = avg_sample*100
        
    metrics_table = metrics_table.append(metrics_dict_position, ignore_index=True)
    
    # metrics for severity only
    metrics_dict_severity = {}
    metrics_dict_severity['model'] = output_type + '_output'
    metrics_dict_severity['dataset'] = 'chestx_severity'
    metrics_dict_severity['training_size'] = sample_size
    metrics_dict_severity['epochs'] = epochs
    metrics_dict_severity['loss_name'] = loss_name
    mesh_train_p = mesh_train[:,pvl+avl+povl:]
    pred_mesh_train_p = pred_mesh_train[:,pvl+avl+povl:]
    mesh_val_p = mesh_val[:,pvl+avl+povl:]
    pred_mesh_val_p = pred_mesh_val[:,pvl+avl+povl:]
    
    avg_binary_accuracy = binary_accuracy_np(mesh_train_p, pred_mesh_train_p)
    metrics_dict_severity['train_bin_acc'] = avg_binary_accuracy*100
    avg_binary_accuracy = binary_accuracy_np(mesh_val_p, pred_mesh_val_p)
    metrics_dict_severity['val_bin_acc'] = avg_binary_accuracy*100

    recall, avg_class, avg_sample = recall_np(mesh_train_p, pred_mesh_train_p)
    metrics_dict_severity['train_recall'] = recall*100
    metrics_dict_severity['train_recall_per_class'] = avg_class*100
    metrics_dict_severity['train_recall_per_sample'] = avg_sample*100
    recall, avg_class, avg_sample = recall_np(mesh_val_p, pred_mesh_val_p)
    metrics_dict_severity['val_recall'] = recall*100
    metrics_dict_severity['val_recall_per_class'] = avg_class*100
    metrics_dict_severity['val_recall_per_sample'] = avg_sample*100

    precision, avg_class, avg_sample = precision_np(mesh_train_p, pred_mesh_train_p)
    metrics_dict_severity['train_prec'] = precision*100
    metrics_dict_severity['train_prec_per_class'] = avg_class*100
    metrics_dict_severity['train_prec_per_sample'] = avg_sample*100
    precision, avg_class, avg_sample = precision_np(mesh_val_p, pred_mesh_val_p)
    metrics_dict_severity['val_prec'] = precision*100
    metrics_dict_severity['val_prec_per_class'] = avg_class*100
    metrics_dict_severity['val_prec_per_sample'] = avg_sample*100
    
    metrics_table = metrics_table.append(metrics_dict_severity, ignore_index=True)
    
    metrics_table.to_csv(dir+'metric_results/chestx/{}_output/mesh_metrics_table_train_{}.csv'.format(output_type, sample_size), index=False)