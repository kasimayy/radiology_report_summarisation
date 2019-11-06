from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, SeparableConv1D
from keras.layers import Flatten, Dropout, Input, LSTM, BatchNormalization, Activation
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from utils.custom_losses import binary_recall_specificity_loss, combined_loss
import json
from keras.models import model_from_json
import os
import numpy as np

class MLPWordCounts(object):
    def __init__(self, **kwargs):
        default_params = {
            "epochs" : 50,
            "optimizer" : 'adam',
            "metrics" : ['accuracy'],
            "loss" : 'binary_crossentropy',
            "batch_size" : 128,
            "input_dim" : 100,
            "output_dim" : 100,
            "embedding_dim1" : 100,
            "embedding_dim2" : 100,
            "max_sentence_length" : 30
        }
        self.__dict__.update(default_params)
        self.__dict__.update(kwargs)
    
    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(self.embedding_dim1, input_dim=self.input_dim, activation='relu'))
        self.model.add(Dense(self.output_dim, activation='sigmoid'))

        self.model.compile(loss=self.loss,
              optimizer=self.optimizer,
              metrics=self.metrics)
        
    def run_experiment(self, sentence_train, ents_train, sentence_val, ents_val):
        self.build_model()
        self.history = self.model.fit(sentence_train, ents_train,
                    epochs=self.epochs,
                    verbose=True,
                    validation_data=(sentence_val, ents_val),
                    batch_size=self.batch_size)
    
    def save_model_history(self, output_dir):
        history_dict = self.history.history
        json.dump(history_dict, open(output_dir + 'history_mlp_wc_{}_epochs_{}_opt_{}_batchsize.json'.format(self.epochs, self.conv_dim1, self.embedding_dim), 'w'))

        self.model.save(os.path.join(output_dir, 'model_mlp__wc_{}_epochs_{}_opt_{}_batchsize.h5'.format(self.epochs, self.conv_dim1, self.embedding_dim)))
        
class MLPWordEmbeddings(MLPWordCounts):
    def build_model(self):
        self.model = Sequential()
        self.model.add(Embedding(self.input_dim, self.embedding_dim1, input_length=self.max_sentence_length))
        self.model.add(Flatten())
        self.model.add(Dense(embedding_dim2, activation='relu'))
        self.model.add(Dense(self.output_dim, activation='sigmoid'))
        
        self.model.compile(loss=self.loss,
              optimizer=self.optimizer,
              metrics=self.metrics)
        
    def save_model_history(self, output_dir):
        history_dict = self.history.history
        json.dump(history_dict, open(output_dir + 'history_mlp_we_{}_epochs_{}_opt_{}_batchsize.json'.format(self.epochs, self.conv_dim1, self.embedding_dim), 'w'))

        self.model.save(os.path.join(output_dir, 'model_mlp__we_{}_epochs_{}_opt_{}_batchsize.h5'.format(self.epochs, self.conv_dim1, self.embedding_dim)))

class MultiLabelTextCNN(object):
    def __init__(self, **kwargs):
        default_params = {
            "epochs" : 50,
            "optimizer" : 'adam',
            "metrics" : ['accuracy'],
            "loss" : 'binary_crossentropy',
            "loss_name" : 'binary_crossentropy',
            "batch_size" : 128,
            "input_dim" : 100,
            "output_dim" : 100,
            "max_sentence_length" : 10,
            "embedding_dim" : 1024,
            "conv_dim1" : 128,
            "conv_dim2" : 128,
            "bce_weight" : 0.5,
            "recall_weight" : 0.5,
            "verbose" : False
        }
        self.__dict__.update(default_params)
        self.__dict__.update(kwargs)
        
        if self.loss_name == 'custom_recall_spec':
            self.loss = 'custom_recall_spec'

        elif self.loss_name == 'combined_loss':
            self.loss = 'combined_loss'
      
    def build_model(self):
        self.model = Sequential()
        self.model.add(Embedding(self.input_dim, self.embedding_dim, input_length=self.max_sentence_length))
        self.model.add(Conv1D(self.conv_dim1, 3, activation='relu'))
#         self.model.add(MaxPooling1D(3))
#         self.model.add(Conv1D(self.conv_dim2, 3, activation='relu'))
        self.model.add(MaxPooling1D(3))
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.output_dim, activation='sigmoid'))

        if self.loss == 'custom_recall_spec':
            self.loss_name = 'custom_recall_spec'
            custom_loss = binary_recall_specificity_loss(self.recall_weight)
            self.loss = custom_loss

        elif self.loss == 'combined_loss':
            self.loss_name = 'combined_loss'
            custom_loss = combined_loss(self.bce_weight, self.recall_weight)
            self.loss = custom_loss

        self.model.compile(loss=self.loss, 
                      optimizer=self.optimizer, 
                      metrics=self.metrics)

    def run_experiment(self, sentence_train, ents_train, sentence_val, ents_val):
        self.build_model()
        self.history = self.model.fit(sentence_train, ents_train,
                    epochs=self.epochs,
                    verbose=self.verbose,
                    validation_data=(sentence_val, ents_val),
                    batch_size=self.batch_size)

    def get_params(self):
        self.params = {}
        keys = ['epochs', 'batch_size', 'input_dim', 'output_dim', 'max_sentence_length',\
                'embedding_dim', 'conv_dim1', 'conv_dim2', 'loss_name', 'bce_weight', 'recall_weight']
        for key in keys:
            value = self.__dict__[key]
            if isinstance(value, np.int64):
                value = int(value)
            self.params[key] = value
        return self.params
            
    def save_weights_history(self, output_dir):
        param_dict = self.get_params()
        param_fn = 'param_cnn_{}_epochs_{}_convdim_{}_embeddingdim_{}_loss_{:.1f}_bce_{:.1f}_recall.json'\
        .format(self.epochs, self.conv_dim1, self.embedding_dim, self.loss_name, self.bce_weight, self.recall_weight)
        json.dump(param_dict, open(output_dir + param_fn, 'w'))
        
        history_dict = self.history.history
        history_fn = 'history_cnn_{}_epochs_{}_convdim_{}_embeddingdim_{}_loss_{:.1f}_bce_{:.1f}_recall.json'\
        .format(self.epochs, self.conv_dim1, self.embedding_dim, self.loss_name, self.bce_weight, self.recall_weight)
        json.dump(history_dict, open(output_dir + history_fn, 'w'))

        weights_fn = 'weights_cnn_{}_epochs_{}_convdim_{}_embeddingdim_{}_loss_{:.1f}_bce_{:.1f}_recall.h5'\
    .format(self.epochs, self.conv_dim1, self.embedding_dim, self.loss_name, self.bce_weight, self.recall_weight)
        self.model.save_weights(output_dir + weights_fn)
        
    def load_weights_history(self, output_dir):
        history_fn = 'history_cnn_{}_epochs_{}_convdim_{}_embeddingdim_{}_loss_{:.1f}_bce_{:.1f}_recall.json'\
        .format(self.epochs, self.conv_dim1, self.embedding_dim, self.loss_name, self.bce_weight, self.recall_weight)
        self.history = json.load(open(output_dir + history_fn, 'r'))
        
        weights_fn = 'weights_cnn_{}_epochs_{}_convdim_{}_embeddingdim_{}_loss_{:.1f}_bce_{:.1f}_recall.h5'\
        .format(self.epochs, self.conv_dim1, self.embedding_dim, self.loss_name, self.bce_weight, self.recall_weight)
        self.model.load_weights(output_dir + weights_fn)
        
class MultiLabelTextCNN2(object):
    def __init__(self, **kwargs):
        default_params = {
            "epochs" : 100,
            "optimizer" : 'adam',
            "metrics" : ['accuracy'],
            "loss" : 'binary_crossentropy',
            "loss_name" : 'binary_crossentropy',
            "batch_size" : 128,
            "input_dim" : 100,
            "output_dim" : 100,
            "max_sentence_length" : 30,
            "embedding_dim" : 512,
            "hidden_dim" : 256,
            "kernel_sizes" : [3,4,5],
            "feature_maps" : [64, 128, 256],
            "bce_weight" : 0.5,
            "recall_weight" : 0.5,
            "verbose" : False
        }
        self.__dict__.update(default_params)
        self.__dict__.update(kwargs)
        
        if self.loss_name == 'custom_recall_spec':
            self.loss = 'custom_recall_spec'

        elif self.loss_name == 'combined_loss':
            self.loss = 'combined_loss'
            
    def build_model(self):
        input_layer = Input((self.max_sentence_length, ))
        embedding_layer = Embedding(self.input_dim, self.embedding_dim, input_length=self.max_sentence_length)(input_layer)
        embedding_layer = Dropout(0.5)(embedding_layer)
        # conv1
        conv_layer1 = SeparableConv1D(self.feature_maps[0], self.kernel_sizes[0], activation='relu', strides=1, padding='same',                     depth_multiplier=4)(embedding_layer)
        max_pool_layer1 = GlobalMaxPooling1D()(conv_layer1)
        dense_layer1 = Dense(self.hidden_dim)(max_pool_layer1)
        dense_layer1 = Dropout(0.5)(dense_layer1)
        # conv2
        conv_layer2 = SeparableConv1D(self.feature_maps[1], self.kernel_sizes[1], activation='relu', strides=1, padding='same',                     depth_multiplier=4)(embedding_layer)
        max_pool_layer2 = GlobalMaxPooling1D()(conv_layer2)
        dense_layer2 = Dense(self.hidden_dim)(max_pool_layer2)
        dense_layer2 = Dropout(0.5)(dense_layer2)
        # conv3
        conv_layer3 = SeparableConv1D(self.feature_maps[2], self.kernel_sizes[2], activation='relu', strides=1, padding='same',                     depth_multiplier=4)(embedding_layer)
        max_pool_layer3 = GlobalMaxPooling1D()(conv_layer3)
        dense_layer3 = Dense(self.hidden_dim)(max_pool_layer3)
        dense_layer3 = Dropout(0.5)(dense_layer3)
        # concatenate conv channels
        concat = concatenate([dense_layer1, dense_layer2, dense_layer3])
        #concat = Dropout(0.5)(concat)
        concat = Activation('relu')(concat)
        output_layer = Dense(self.output_dim, activation='sigmoid')(concat)

        self.model = Model(inputs=input_layer, outputs=output_layer)
        if self.loss == 'custom_recall_spec':
            self.loss_name = 'custom_recall_spec'
            custom_loss = binary_recall_specificity_loss(self.recall_weight)
            self.loss = custom_loss

        elif self.loss == 'combined_loss':
            self.loss_name = 'combined_loss'
            custom_loss = combined_loss(self.bce_weight, self.recall_weight)
            self.loss = custom_loss

        self.model.compile(loss=self.loss, 
                      optimizer=self.optimizer, 
                      metrics=self.metrics)

    def run_experiment(self, sentence_train, ents_train, sentence_val, ents_val):
        self.build_model()
        self.history = self.model.fit(sentence_train, ents_train,
                    epochs=self.epochs,
                    verbose=self.verbose,
                    validation_data=(sentence_val, ents_val),
                    batch_size=self.batch_size)

    def get_params(self):
        self.params = {}
        keys = ['epochs', 'batch_size', 'input_dim', 'output_dim', 'max_sentence_length',\
                'embedding_dim', 'kernel_sizes', 'feature_maps', 'loss_name', 'bce_weight', 'recall_weight']
        for key in keys:
            value = self.__dict__[key]
            if isinstance(value, np.int64):
                value = int(value)
            self.params[key] = value
        return self.params
            
    def save_weights_history(self, output_dir):
        param_dict = self.get_params()
        param_fn = 'param_cnn_{}_epochs_{}_kernel_sizes_{}_feature_maps_{}_embeddingdim_{}_loss_{:.1f}_bce_{:.1f}_recall.json'\
        .format(self.epochs, self.kernel_sizes, self.feature_maps, self.embedding_dim, self.loss_name, self.bce_weight, self.recall_weight)
        json.dump(param_dict, open(output_dir + param_fn, 'w'))
        
        history_dict = self.history.history
        history_fn = 'history_cnn_{}_epochs_{}_kernel_sizes_{}_feature_maps_{}_embeddingdim_{}_loss_{:.1f}_bce_{:.1f}_recall.json'\
        .format(self.epochs, self.kernel_sizes, self.feature_maps, self.embedding_dim, self.loss_name, self.bce_weight, self.recall_weight)
        json.dump(history_dict, open(output_dir + history_fn, 'w'))

        weights_fn = 'weights_cnn_{}_epochs_{}_kernel_sizes_{}_feature_maps_{}_embeddingdim_{}_loss_{:.1f}_bce_{:.1f}_recall.h5'\
    .format(self.epochs, self.kernel_sizes, self.feature_maps, self.embedding_dim, self.loss_name, self.bce_weight, self.recall_weight)
        self.model.save_weights(output_dir + weights_fn)
        
    def load_weights_history(self, output_dir):
        history_fn = 'history_cnn_{}_epochs_{}_kernel_sizes_{}_feature_maps_{}_embeddingdim_{}_loss_{:.1f}_bce_{:.1f}_recall.json'\
        .format(self.epochs, self.kernel_sizes, self.feature_maps, self.embedding_dim, self.loss_name, self.bce_weight, self.recall_weight)
        self.history = json.load(open(output_dir + history_fn, 'r'))
        
        weights_fn = 'weights_cnn_{}_epochs_{}_kernel_sizes_{}_feature_maps_{}_embeddingdim_{}_loss_{:.1f}_bce_{:.1f}_recall.h5'\
        .format(self.epochs, self.kernel_sizes, self.feature_maps, self.embedding_dim, self.loss_name, self.bce_weight, self.recall_weight)
        self.model.load_weights(output_dir + weights_fn)

class MultiLabelMultiOutputTextCNN(MultiLabelTextCNN):
    def __init__(self, **kwargs):
        default_params = {
            "dense_output_dims" : [10, 10, 10, 10], # dimensions of each dense output (num categorical classes per output)
                                                    # must be given in the order [pathology, anatomy, position, severity]
            "loss_weights" : {'pathology': 0.6, 'anatomy': 0.2, 'position': 0.1, 'severity': 0.1},
            "epochs" : 50,
            "optimizer" : 'adam',
            "metrics" : ['accuracy'],
            "loss" : 'categorical_crossentropy',
            "loss_name" : 'categorical_crossentropy',
            "batch_size" : 128,
            "input_dim" : 100,
            "output_dim" : [100, 100, 100, 100],
            "max_sentence_length" : 10,
            "embedding_dim" : 1024,
            "fc_dim" : 512,
            "conv_dim1" : 128,
            "conv_dim2" : 128,
            "bce_weight" : 0.5,
            "recall_weight" : 0.5,
            "verbose" : False
        }
        self.__dict__.update(default_params)
        self.__dict__.update(kwargs)
        
        if self.loss_name == 'custom_recall_spec':
            self.loss = 'custom_recall_spec'

        elif self.loss_name == 'combined_loss':
            self.loss = 'combined_loss'
          
    def conv_block(inp, filters=32, bn=True, pool=True):
        _ = Conv1D(filters=filters, kernel_size=3, activation='relu')(inp)
        if bn:
            _ = BatchNormalization()(_)
        if pool:
            _ = MaxPool2D()(_)
        return _

    def build_model(self):
        input_layer = Input((self.max_sentence_length, ))
        embedding_layer = Embedding(self.input_dim, self.embedding_dim, input_length=self.max_sentence_length)(input_layer)
        conv_layer = Conv1D(self.conv_dim1, 3, activation='relu')(embedding_layer)
        max_pool_layer = MaxPooling1D(3)(conv_layer)
#         conv_layer2 = Conv1D(self.conv_dim2, 3, activation='relu')(max_pool_layer)
#         max_pool_layer2 = MaxPooling1D(2)(conv_layer2)
        batch_norm_layer = BatchNormalization()(max_pool_layer)
        pool_layer = GlobalMaxPooling1D()(max_pool_layer)
        #flatten_layer = Flatten()(max_pool_layer)
        dropout_layer = Dropout(0.5)(pool_layer)
        #dense_layer = Dense(self.fc_dim, activation='relu')(dropout_layer)
        
        # pathology layer
        p_dense_layer = Dense(self.output_dim[0], activation='relu')(dropout_layer)
        p_output = Dense(units=self.dense_output_dims[0], activation='softmax', name='pathology')(p_dense_layer)
        
        # anatomy
        a_dense_layer = Dense(self.output_dim[1], activation='relu')(dropout_layer)
        a_output = Dense(units=self.dense_output_dims[1], activation='softmax', name='anatomy')(a_dense_layer)
        
        # position
        po_dense_layer = Dense(self.output_dim[2], activation='relu')(dropout_layer)
        po_output = Dense(units=self.dense_output_dims[2], activation='softmax', name='position')(po_dense_layer)
        
        # severity
        s_dense_layer = Dense(self.output_dim[3], activation='relu')(dropout_layer)
        s_output = Dense(units=self.dense_output_dims[3], activation='softmax', name='severity')(s_dense_layer)

        self.model = Model(inputs=input_layer, outputs=[p_output, a_output, po_output, s_output])
        
        self.model.compile(loss=self.loss, 
                      loss_weights=self.loss_weights,
                      optimizer=self.optimizer, 
                      metrics=self.metrics)
        
    def run_experiment(self, sentence_train, ents_train, sentence_val, ents_val):
        self.build_model()
        self.history = self.model.fit(sentence_train, ents_train,
                    epochs=self.epochs,
                    verbose=self.verbose,
                    validation_data=(sentence_val, ents_val),
                    batch_size=self.batch_size)

    def get_params(self):
        self.params = {}
        keys = ['dense_output_dims', 'loss_weights', 'epochs', 'batch_size', 'input_dim', 'output_dim', 'max_sentence_length',\
                'embedding_dim', 'conv_dim1', 'conv_dim2', 'fc_dim', 'loss_name', 'bce_weight', 'recall_weight']
        for key in keys:
            value = self.__dict__[key]
            if isinstance(value, np.int64):
                value = int(value)
            self.params[key] = value
        return self.params
    
    def save_weights_history(self, output_dir):
        param_dict = self.get_params()
        param_fn = 'param_cnn_{}_epochs_{}_convdim1_{}_convdim2_{}_fc_dim_{}_embeddingdim_{}_loss_weights.json'\
        .format(self.epochs, self.conv_dim1, self.conv_dim2, self.fc_dim, self.embedding_dim, [float(x) for x in self.loss_weights.values()])
        json.dump(param_dict, open(output_dir + param_fn, 'w'))
        
        history_dict = self.history.history
        history_fn = 'history_cnn_{}_epochs_{}_convdim1_{}_convdim2_{}_fc_dim_{}_embeddingdim_{}_loss_weights.json'\
        .format(self.epochs, self.conv_dim1, self.conv_dim2, self.fc_dim, self.embedding_dim, [float(x) for x in self.loss_weights.values()])
        json.dump(history_dict, open(output_dir + history_fn, 'w'))

        weights_fn = 'weights_cnn_{}_epochs_{}_convdim1_{}_convdim2_{}_fc_dim_{}_embeddingdim_{}_loss_weights.h5'\
    .format(self.epochs, self.conv_dim1, self.conv_dim2, self.fc_dim, self.embedding_dim, [float(x) for x in self.loss_weights.values()])
        self.model.save_weights(output_dir + weights_fn)
        
    def load_weights_history(self, output_dir):
        history_fn = 'history_cnn_{}_epochs_{}_convdim1_{}_convdim2_{}_fc_dim_{}_embeddingdim_{}_loss_weights.json'\
        .format(self.epochs, self.conv_dim1, self.conv_dim2, self.fc_dim, self.embedding_dim, [float(x) for x in self.loss_weights.values()])
        self.history = json.load(open(output_dir + history_fn, 'r'))
        
        weights_fn = 'weights_cnn_{}_epochs_{}_convdim1_{}_convdim2_{}_fc_dim_{}_embeddingdim_{}_loss_weights.h5'\
        .format(self.epochs, self.conv_dim1, self.conv_dim2, self.fc_dim, self.embedding_dim, [float(x) for x in self.loss_weights.values()])
        self.model.load_weights(output_dir + weights_fn)
    
class Seq2Seq(object):
    def __init__(self, **kwargs):
        default_params = {
            "epochs" : 50,
            "optimizer" : 'adam',
            "metrics" : ['accuracy'],
            "loss" : 'categorical_crossentropy',
            "loss_name" : 'categorical_crossentropy',
            "batch_size" : 128,
            "input_dim" : 100,
            "output_dim" : 100,
            "latent_dim" : 256,
            "input_seq_length" : 10,
            "output_seq_length" : 10,
            "verbose" : False
        }
        self.__dict__.update(default_params)
        self.__dict__.update(kwargs)
        
    def build_model(self):
        # training model
        encoder_inputs = Input(shape=(None, self.input_dim))
        encoder = LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, self.output_dim))
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = Dense(self.output_dim, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        
        self.model.compile(loss=self.loss, 
              optimizer=self.optimizer, 
              metrics=self.metrics)
        
        # sampling model
        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
        
        
    def run_experiment(self, sentence_train, ents_train, ents_shifted_train, sentence_val, ents_val, ents_shifted_val):
        self.build_model()
        self.history = self.model.fit([sentence_train, ents_train], ents_shifted_train,
                    epochs=self.epochs,
                    verbose=self.verbose,
                    validation_data=([sentence_val, ents_val], ents_shifted_val),
                    batch_size=self.batch_size)
        
    def run_experiment_batch(self, sentence_train, ents_train, ents_shifted_train, sentence_val, ents_val, ents_shifted_val):
        self.build_model()
        self.history = self.model.fit_generator([sentence_train, ents_train], ents_shifted_train,
                    epochs=self.epochs,
                    verbose=self.verbose,
                    validation_data=([sentence_val, ents_val], ents_shifted_val),
                    batch_size=self.batch_size)
        
    def decode_sequence(self, input_seq, id_to_ent, ent_to_id):
        states_value = self.encoder_model.predict(input_seq)
        print(states_value[0].shape)
        target_seq = np.zeros((1, 1, self.output_dim))
        target_seq[0, 0, ent_to_id['start']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)
            print(output_tokens.shape)
            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = id_to_ent[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '.' or
               len(decoded_sentence) > self.output_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.output_dim))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence
        
    def get_params(self):
        self.params = {}
        keys = ['epochs', 'batch_size', 'input_dim', 'output_dim', 'input_seq_length', 'output_seq_length',\
                'latent_dim', 'loss_name']
        for key in keys:
            value = self.__dict__[key]
            if isinstance(value, np.int64):
                value = int(value)
            self.params[key] = value
        return self.params
        
    def save_weights_history(self, output_dir):
        param_dict = self.get_params()
        param_fn = 'param_cnn_epochs_{}_latentdim_{}.json'\
        .format(self.epochs, self.latent_dim)
        json.dump(param_dict, open(output_dir + param_fn, 'w'))
        
        history_dict = self.history.history
        history_fn = 'history_cnn_epochs_{}_latentdim_{}.json'\
        .format(self.epochs, self.latent_dim)
        json.dump(history_dict, open(output_dir + history_fn, 'w'))

#         weights_fn = 'model_weights_cnn_epochs_{}_latentdim_{}.h5'\
#     .format(self.epochs, self.latent_dim)
#         self.model.save_weights(output_dir + weights_fn)
        
        weights_fn = 'encoder_weights_cnn_epochs_{}_latentdim_{}.h5'\
    .format(self.epochs, self.latent_dim)
        self.encoder_model.save_weights(output_dir + weights_fn)
        
        weights_fn = 'decoder_weights_cnn_epochs_{}_latentdim_{}.h5'\
    .format(self.epochs, self.latent_dim)
        self.decoder_model.save_weights(output_dir + weights_fn)
        
    def load_weights_history(self, output_dir):
        history_fn = 'history_cnn_epochs_{}_latentdim_{}.json'\
        .format(self.epochs, self.latent_dim)
        self.history = json.load(open(output_dir + history_fn, 'r'))
        
#         weights_fn = 'model_weights_cnn_epochs_{}_latentdim_{}.h5'\
#         .format(self.epochs, self.latent_dim)
#         self.model.load_weights(output_dir + weights_fn)
        
        weights_fn = 'encoder_weights_cnn_epochs_{}_latentdim_{}.h5'\
        .format(self.epochs, self.latent_dim)
        self.encoder_model.load_weights(output_dir + weights_fn)
        
        weights_fn = 'decoder_weights_cnn_epochs_{}_latentdim_{}.h5'\
        .format(self.epochs, self.latent_dim)
        self.decoder_model.load_weights(output_dir + weights_fn)