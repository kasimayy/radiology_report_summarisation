from tensorflow.keras import backend as K
from tensorflow.keras.backend import dot 
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, SeparableConv1D, Concatenate, Embedding, RepeatVector
from tensorflow.keras.layers import Flatten, Dropout, Input, LSTM, BatchNormalization, Activation, TimeDistributed, Lambda, Reshape
from utils.attention import AttentionLayer, AttentionLayer2, HierarchicalAttentionNetwork
from utils.attention_with_context import AttentionWithContext, AttLayer
from utils.custom_losses import binary_recall_specificity_loss, combined_loss
from utils.rnn_textsum_models import Seq2Seq
import pickle

import os
import numpy as np

def one_hot_decode(encoded_seq):
    return [np.argmax(vector) for vector in encoded_seq]
    
class Emb2SeqAtt(Seq2Seq):
    def __init__(self, **kwargs):
        self.model_name = 'emb2seq_att'
        self.emb = True
        self.epochs = 50
        self.optimizer = 'adam'
        self.metrics = ['accuracy']
        self.loss = 'categorical_crossentropy'
        self.loss_name = 'categorical_crossentropy'
        self.batch_size = 128
        self.input_dim = 100 # input vocab length
        self.output_dim = 100 # output vocab length
        self.embedding_dim = 256
        self.decoder_hidden_dim = 256        
        self.input_seq_length = 100 # max num words per input sentence
        self.output_seq_length = 20 # max num output words
        self.dropout_rate = 0.5
        self.verbose = False
        self.__dict__.update(kwargs)
        
    def build_model(self):
        # define word CNN encoder
        word_encoder_inputs = Input(batch_shape=(None, self.input_seq_length))
        word_encoder_emb = Embedding(self.input_dim, 
                                     self.embedding_dim, 
                                     input_length=self.input_seq_length)(word_encoder_inputs)
        
        self.encoder_model = Model(word_encoder_inputs, word_encoder_emb)
        # define sequential training decoder
        decoder_inputs = Input(batch_shape=(None, self.output_seq_length, self.output_dim))
        decoder = LSTM(self.decoder_hidden_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder(decoder_inputs)
        
        attn_layer = AttentionLayer(name='attention_layer')
        attn_outputs, attn_states = attn_layer([word_encoder_emb, decoder_outputs])
        decoder_concat_inputs = Concatenate(axis=-1, name='dec_concat_layer')([decoder_outputs, attn_outputs])
        
        dense = Dense(self.output_dim, activation='softmax')
        dense_time = TimeDistributed(dense)
        decoder_pred = dense_time(decoder_concat_inputs)
        
        self.model = Model(inputs=[word_encoder_inputs, decoder_inputs], outputs=decoder_pred)

        # define decoder inference model
        decoder_init_hstate = Input(batch_shape=(None, self.decoder_hidden_dim))
        decoder_init_cstate = Input(batch_shape=(None, self.decoder_hidden_dim))
        decoder_init_state = [decoder_init_hstate, decoder_init_cstate]
        
        encoder_inf_concat = Input(batch_shape=(None, self.input_seq_length, self.embedding_dim))
        decoder_inf_inputs = Input(batch_shape=(None, 1, self.output_dim))
        decoder_inf_outputs, state_h, state_c = decoder(decoder_inf_inputs)
        decoder_inf_states = [state_h, state_c]
        self.decoder_model = Model(inputs=decoder_inf_inputs, outputs=[decoder_inf_outputs, decoder_inf_states])
        
        attn_inf_outputs, attn_inf_states = attn_layer([encoder_inf_concat, decoder_inf_outputs])
        decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_inf_outputs, attn_inf_outputs])
        decoder_inf_pred = TimeDistributed(dense)(decoder_inf_concat)
        self.inference_model = Model(inputs=[encoder_inf_concat, decoder_init_state, decoder_inf_inputs], outputs=[decoder_inf_pred, attn_inf_states, decoder_inf_states])

        self.model.compile(loss=self.loss, 
              optimizer=self.optimizer, 
              metrics=self.metrics)
        

class CNN2SeqAtt(Seq2Seq):
    def __init__(self, **kwargs):
        self.model_name = 'cnn2seq_att'
        self.emb = True
        self.epochs = 50
        self.optimizer = 'adam'
        self.metrics = ['accuracy']
        self.loss = 'categorical_crossentropy'
        self.loss_name = 'categorical_crossentropy'
        self.batch_size = 128
        self.input_dim = 100 # input vocab length
        self.output_dim = 100 # output vocab length
        self.embedding_dim = 256
        self.decoder_hidden_dim = 256
        self.conv_dim = 128
        self.conv1_kernel = 3
        self.conv2_kernel = 4
        self.conv3_kernel = 5
        self.input_seq_length = 100 # max num words per input sentence
        self.output_seq_length = 20 # max num output words
        self.dropout_rate = 0.5
        self.verbose = False
        self.__dict__.update(kwargs)
        
    def build_model(self):
        # define word CNN encoder
#         word_encoder_inputs = Input(batch_shape=(None, self.input_seq_length))
#         word_encoder_emb = Embedding(self.input_dim, 
#                                      self.embedding_dim, 
#                                      input_length=self.input_seq_length)(word_encoder_inputs)
        word_encoder_emb = Input(batch_shape=(None, self.input_word_seq_length, self.input_dim))
        print('Word encoder emb: ', K.int_shape(word_encoder_inputs))
        
        word_conv1_outputs = Conv1D(self.conv_dim, 
                                    self.conv1_kernel, 
                                    activation='relu', 
                                    padding='same')(word_encoder_emb)
        print('Word conv1 output: ', K.int_shape(word_conv1_outputs))
        
        word_conv2_outputs = Conv1D(self.conv_dim, 
                                    self.conv2_kernel, 
                                    activation='relu', 
                                    padding='same')(word_encoder_emb)
        print('Word conv2 output: ', K.int_shape(word_conv2_outputs))
        
        word_conv3_outputs = Conv1D(self.conv_dim, 
                                    self.conv3_kernel, 
                                    activation='relu', 
                                    padding='same')(word_encoder_emb)
        print('Word conv3 output: ', K.int_shape(word_conv3_outputs))

        encoder_concat_outputs = Concatenate(axis=-1, name='enc_concat_layer')([word_conv1_outputs, word_conv2_outputs, word_conv3_outputs])
        print('Concat: ', K.int_shape(encoder_concat_outputs))
        
#         encoder_attn_layer = AttentionWithContext()
#         encoder_attn_outputs = encoder_attn_layer(encoder_concat_inputs)
#         print('Attention: ', K.int_shape(encoder_attn_outputs))
        
#         sent_emb_repeat = RepeatVector(self.output_seq_length)(encoder_attn_outputs)
#         print('Sent emb: ', K.int_shape(sent_emb))
        
        self.encoder_model = Model(word_encoder_inputs, encoder_concat_outputs)
        #doc_emb = Dense(self.output_dim, activation='sigmoid')()

        # define sequential training decoder
        decoder_inputs = Input(batch_shape=(None, self.output_seq_length, self.output_dim))
        decoder = LSTM(self.decoder_hidden_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder(decoder_inputs)
        
        attn_layer = AttentionLayer(name='attention_layer')
        attn_outputs, attn_states = attn_layer([encoder_concat_outputs, decoder_outputs])
        decoder_concat_inputs = Concatenate(axis=-1, name='dec_concat_layer')([decoder_outputs, attn_outputs])
        
        dense = Dense(self.output_dim, activation='softmax')
        dense_time = TimeDistributed(dense)
        decoder_pred = dense_time(decoder_concat_inputs)
        
        self.model = Model(inputs=[word_encoder_inputs, decoder_inputs], outputs=decoder_pred)

#         # define decoder inference model
        decoder_init_hstate = Input(batch_shape=(None, self.decoder_hidden_dim))
        decoder_init_cstate = Input(batch_shape=(None, self.decoder_hidden_dim))
        decoder_init_state = [decoder_init_hstate, decoder_init_cstate]
        
        encoder_inf_concat = Input(batch_shape=(None, self.input_seq_length, self.conv_dim*3))
        decoder_inf_inputs = Input(batch_shape=(None, 1, self.output_dim))
        decoder_inf_outputs, state_h, state_c = decoder(decoder_inf_inputs)
        decoder_inf_states = [state_h, state_c]
        self.decoder_model = Model(inputs=decoder_inf_inputs, outputs=[decoder_inf_outputs, decoder_inf_states])
        
        attn_inf_outputs, attn_inf_states = attn_layer([encoder_inf_concat, decoder_inf_outputs])
        decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_inf_outputs, attn_inf_outputs])
        decoder_inf_pred = TimeDistributed(dense)(decoder_inf_concat)
        self.inference_model = Model(inputs=[encoder_inf_concat, decoder_init_state, decoder_inf_inputs], outputs=[decoder_inf_pred, attn_inf_states, decoder_inf_states])

        self.model.compile(loss=self.loss, 
              optimizer=self.optimizer, 
              metrics=self.metrics)

    def get_params(self):
        self.params = {}
        keys = ['model_name', 'epochs', 'batch_size', 'input_dim', 'output_dim', 'conv_dim', 'conv1_kernel', 'conv2_kernel', 'conv3_kernel',  'embedding_dim', 'input_seq_length', 'output_seq_length',  'decoder_hidden_dim', 'loss_name']
        for key in keys:
            value = self.__dict__[key]
            if isinstance(value, np.int64):
                value = int(value)
            self.params[key] = value
        return self.params
    
    def get_filename(self):
        fn = '_{}_convdim_{}_kernels_{}{}{}_epochs_{}.pkl'.format(
             self.model_name,
             self.conv_dim,
             self.conv1_kernel, 
             self.conv2_kernel,
             self.conv3_kernel,
             self.epochs)
        return fn
    
    def save_weights_history(self, output_dir):
        param_dict = self.get_params()
        param_fn = 'param' + self.get_filename()
        with open(output_dir + param_fn, 'wb') as f:
            pickle.dump(param_dict, f)
        
        history_dict = self.history.history
        history_fn = 'history' + self.get_filename()
        with open(output_dir + history_fn, 'wb') as f:
            pickle.dump(history_dict, f)
        
        weights_fn = 'encoder_weights' + self.get_filename()
        self.encoder_model.save_weights(output_dir + weights_fn)
        
        weights_fn = 'decoder_weights' + self.get_filename()
        self.decoder_model.save_weights(output_dir + weights_fn)
        
        weights_fn = 'inference_weights' + self.get_filename()
        self.inference_model.save_weights(output_dir + weights_fn)

    def load_weights_history(self, output_dir):
        history_fn = 'history' + self.get_filename()
        with open(output_dir + history_fn, 'rb') as f:
            self.history = pickle.load(f)

        weights_fn = 'encoder_weights' + self.get_filename()
        self.encoder_model.load_weights(output_dir + weights_fn)
        
        weights_fn = 'decoder_weights' + self.get_filename()
        self.decoder_model.load_weights(output_dir + weights_fn)
        
        weights_fn = 'inference_weights' + self.get_filename()
        self.inference_model.load_weights(output_dir + weights_fn)
        
class HierCNN2SeqAtt(CNN2SeqAtt):
    def __init__(self, **kwargs):
        self.model_name = 'hiercnn2seq_att'
        self.emb = True
        self.epochs = 50
        self.optimizer = 'adam'
        self.metrics = ['accuracy']
        self.loss = 'categorical_crossentropy'
        self.loss_name = 'categorical_crossentropy'
        self.batch_size = 128
        self.input_dim = 100 # input vocab length
        self.output_dim = 100 # output vocab length
        self.embedding_dim = 256
        self.word_conv_dim = 64
        self.phrase_conv_dim = 128
        self.conv1_kernel = 1
        self.conv2_kernel = 3
        self.conv3_kernel = 5
        self.input_seq_length = 100 # max num words per input sentence
        self.output_seq_length = 20 # max num output words
        self.dropout_rate = 0.5
        self.verbose = False
        self.__dict__.update(kwargs)
        
    def build_model(self):
        self.decoder_hidden_dim = self.phrase_conv_dim*3+self.word_conv_dim*4
        self.encoder_concat_dim = self.phrase_conv_dim*3+self.word_conv_dim*4
        # define word CNN encoder
#         word_encoder_inputs = Input(batch_shape=(None, self.input_seq_length))
#         word_encoder_emb = Embedding(self.input_dim, 
#                                      self.embedding_dim, 
#                                      input_length=self.input_seq_length)(word_encoder_inputs)

        word_encoder_inputs = Input(batch_shape=(None, self.input_seq_length, self.input_dim))
        print('Word encoder emb: ', K.int_shape(word_encoder_inputs))
        
        word_conv1_outputs = Conv1D(self.word_conv_dim, 
                                    self.conv1_kernel, 
                                    activation='relu', 
                                    padding='same', 
                                    strides=1)(word_encoder_inputs)
        print('Word conv1 output: ', K.int_shape(word_conv1_outputs))
        
        word_conv2_outputs = Conv1D(self.word_conv_dim, 
                                    self.conv2_kernel, 
                                    activation='relu', 
                                    padding='same',
                                    strides=1)(word_encoder_inputs)
        print('Word conv2 output: ', K.int_shape(word_conv2_outputs))
        
        word_conv3_outputs = Conv1D(self.word_conv_dim, 
                                    self.conv3_kernel, 
                                    activation='relu', 
                                    padding='same',
                                    strides=1)(word_encoder_inputs)
        print('Word conv3 output: ', K.int_shape(word_conv3_outputs))

        word_encoder_outputs = Concatenate(axis=-1, name='word_concat_layer')([word_conv1_outputs, word_conv2_outputs, word_conv3_outputs])
        print('Word Concat: ', K.int_shape(word_encoder_outputs))
        
        max_pool_word_outputs = MaxPooling1D(pool_size=3,
                                             padding='same',
                                             strides=1)(word_encoder_outputs)
        print('Max pool outputs: ', K.int_shape(max_pool_word_outputs))
        
        phrase_conv1_outputs = Conv1D(self.phrase_conv_dim, 
                                   self.conv1_kernel, 
                                   activation='relu',
                                   padding='same')(word_encoder_outputs)
        
        phrase_conv2_outputs = Conv1D(self.phrase_conv_dim, 
                                   self.conv2_kernel, 
                                   activation='relu',
                                   padding='same')(word_encoder_outputs)
        
        phrase_conv3_outputs = Conv1D(self.phrase_conv_dim, 
                                   self.conv3_kernel, 
                                   activation='relu',
                                   padding='same')(word_encoder_outputs)
        
        phrase_encoder_outputs = Concatenate(axis=-1, name='phrase_concat_layer')([word_conv1_outputs, phrase_conv1_outputs, phrase_conv2_outputs, phrase_conv3_outputs, max_pool_word_outputs])
        print('Phrase Concat: ', K.int_shape(phrase_encoder_outputs))
        
        encoder_state = Lambda(lambda x: K.mean(x, axis=1))(phrase_encoder_outputs)
        encoder_states = [encoder_state, encoder_state]
        print('Encoder states: ', K.int_shape(encoder_state))
        
        encoder = Model(word_encoder_inputs, [phrase_encoder_outputs, encoder_states])
        #doc_emb = Dense(self.output_dim, activation='sigmoid')()

        # define sequential training decoder
        decoder_inputs = Input(batch_shape=(None, self.output_seq_length, self.output_dim))
        decoder = LSTM(self.decoder_hidden_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)
        print('Decoder outputs: ', K.int_shape(decoder_outputs))
        
        attn_layer = AttentionLayer(name='attention_layer')
        attn_outputs, attn_states = attn_layer([phrase_encoder_outputs, decoder_outputs])
        print('Attention outputs: ', K.int_shape(attn_outputs))
        decoder_concat_inputs = Concatenate(axis=-1, name='dec_concat_layer')([decoder_outputs, attn_outputs])
        
        dense = Dense(self.output_dim, activation='softmax')
        dense_time = TimeDistributed(dense)
        decoder_pred = dense_time(decoder_concat_inputs)
        
        self.model = Model(inputs=[word_encoder_inputs, decoder_inputs], outputs=decoder_pred)

        # define encoder inference model
        encoder_inf_inputs = Input(batch_shape=(None, self.input_seq_length, self.input_dim))
        encoder_inf_outputs, encoder_inf_states = encoder(encoder_inf_inputs)
#         encoder_inf_states = [state_h, state_c]
        self.encoder_model = Model(inputs=encoder_inf_inputs, outputs=[encoder_inf_outputs, encoder_inf_states])
        
        # define decoder inference model
        encoder_inf_concat = Input(batch_shape=(None, self.input_seq_length, self.encoder_concat_dim))
        #encoder_inf_states = Input(batch_shape=(None, self.decoder_hidden_dim))
        decoder_inf_inputs = Input(batch_shape=(None, 1, self.output_dim))
        decoder_init_hstate = Input(batch_shape=(None, self.decoder_hidden_dim))
        decoder_init_cstate = Input(batch_shape=(None, self.decoder_hidden_dim))
        decoder_init_state = [decoder_init_hstate, decoder_init_cstate]

        decoder_inf_outputs, state_h, state_c = decoder(decoder_inf_inputs, initial_state=decoder_init_state)
        decoder_inf_states = [state_h, state_c]
        attn_inf_outputs, attn_inf_states = attn_layer([encoder_inf_concat, decoder_inf_outputs])
        decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_inf_outputs, attn_inf_outputs])
        decoder_inf_pred = TimeDistributed(dense)(decoder_inf_concat)
        self.decoder_model = Model(inputs=[encoder_inf_concat, decoder_init_state, decoder_inf_inputs],
                              outputs=[decoder_inf_pred, attn_inf_states, decoder_inf_states])
        
        # define decoder inference model
#         decoder_init_hstate = Input(batch_shape=(None, self.decoder_hidden_dim))
#         decoder_init_cstate = Input(batch_shape=(None, self.decoder_hidden_dim))
#         decoder_init_state = [decoder_init_hstate, decoder_init_cstate]
        
#         encoder_inf_concat = Input(batch_shape=(None, self.input_seq_length, self.encoder_concat_dim))
        
#         decoder_inf_inputs = Input(batch_shape=(None, 1, self.output_dim))
#         decoder_inf_outputs, state_h, state_c = decoder(decoder_inf_inputs, initial_state=decoder_init_state)
#         decoder_inf_states = [state_h, state_c]
# #         self.decoder_model = Model(inputs=decoder_inf_inputs, outputs=[decoder_inf_outputs, decoder_inf_states])
        
#         attn_inf_outputs, attn_inf_states = attn_layer([encoder_inf_concat, decoder_inf_outputs])
#         decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_inf_outputs, attn_inf_outputs])
#         decoder_inf_pred = TimeDistributed(dense)(decoder_inf_concat)
#         self.decoder_model = Model(inputs=[encoder_inf_states, decoder_init_state, decoder_inf_inputs], outputs=[decoder_inf_pred, attn_inf_states, decoder_inf_states])

        self.model.compile(loss=self.loss, 
              optimizer=self.optimizer, 
              metrics=self.metrics)

    def get_params(self):
        self.params = {}
        keys = ['model_name', 'epochs', 'batch_size', 'input_dim', 'output_dim', 'word_conv_dim', 'phrase_conv_dim', 'conv1_kernel', 'conv2_kernel', 'conv3_kernel',  'embedding_dim', 'input_seq_length', 'output_seq_length',  'decoder_hidden_dim', 'loss_name']
        for key in keys:
            value = self.__dict__[key]
            if isinstance(value, np.int64):
                value = int(value)
            self.params[key] = value
        return self.params
    
    def get_filename(self):
        fn = '_{}_wordconvdim_{}_phraseconvdim_{}_kernels_{}{}{}_epochs_{}.pkl'.format(
             self.model_name,
             self.word_conv_dim, 
             self.phrase_conv_dim, 
             self.conv1_kernel, 
             self.conv2_kernel,
             self.conv3_kernel,
             self.epochs)
        return fn
    
    def save_weights_history(self, output_dir):
        param_dict = self.get_params()
        param_fn = 'param' + self.get_filename()
        with open(output_dir + param_fn, 'wb') as f:
            pickle.dump(param_dict, f)
        
        history_dict = self.history.history
        history_fn = 'history' + self.get_filename()
        with open(output_dir + history_fn, 'wb') as f:
            pickle.dump(history_dict, f)
        
        weights_fn = 'encoder_weights' + self.get_filename()
        self.encoder_model.save_weights(output_dir + weights_fn)
        
        weights_fn = 'decoder_weights' + self.get_filename()
        self.decoder_model.save_weights(output_dir + weights_fn)
        
    def load_weights_history(self, output_dir):
        history_fn = 'history' + self.get_filename()
        with open(output_dir + history_fn, 'rb') as f:
            self.history = pickle.load(f)

        weights_fn = 'encoder_weights' + self.get_filename()
        self.encoder_model.load_weights(output_dir + weights_fn)
        
        weights_fn = 'decoder_weights' + self.get_filename()
        self.decoder_model.load_weights(output_dir + weights_fn)
    
class CNN2HierSeqAtt3(Seq2Seq):
    def __init__(self, **kwargs):
        self.model_name = 'cnn2hierseq_att'
        self.emb = True
        self.epochs = 50
        self.optimizer = 'adam'
        self.metrics = ['accuracy']
        self.loss = 'categorical_crossentropy'
        self.loss_name = 'categorical_crossentropy'
        self.batch_size = 128
        self.input_dim = 100 # input vocab length
        self.output_dim = 100 # output vocab length
        self.embedding_dim = 256
        self.decoder_hidden_dim = 256
        self.conv1_dim = 128
        self.conv1_kernel = 3
        self.conv2_dim = 128
        self.conv2_kernel = 3
        self.input_seq_length = 100 # max num words per input sentence
        self.output_seq_length = 20 # max num output words
        self.dropout_rate = 0.5
        self.verbose = False
        self.__dict__.update(kwargs)
        
    def build_model(self):
        # define word CNN encoder
        word_encoder_inputs = Input(batch_shape=(None, self.input_seq_length))
        word_encoder_emb = Embedding(self.input_dim, 
                                     self.embedding_dim, 
                                     input_length=self.input_seq_length)(word_encoder_inputs)
        word_conv_outputs = Conv1D(self.conv1_dim, self.conv1_kernel, activation='relu')(word_encoder_emb)
        word_conv_outputs = MaxPooling1D(3)(word_conv_outputs)
        print('Word conv output: ', K.int_shape(word_conv_outputs))
        
        # define sentence CNN encoder
        sent_conv_outputs = Conv1D(self.conv2_dim, self.conv2_kernel, activation='relu')(word_conv_outputs)
        sent_conv_outputs = MaxPooling1D(3)(sent_conv_outputs)
        print('Sent conv output: ', K.int_shape(sent_conv_outputs))
        
        sent_emb = GlobalMaxPooling1D()(sent_conv_outputs)
        sent_emb_repeat = RepeatVector(self.output_seq_length)(sent_emb)
        print('Sent emb: ', K.int_shape(sent_emb))
        
        encoder = Model(word_encoder_inputs, sent_emb)
        #doc_emb = Dense(self.output_dim, activation='sigmoid')()

        # define sequential training decoder
        decoder_inputs = Input(batch_shape=(None, self.output_seq_length, self.output_dim))
        decoder = LSTM(self.decoder_hidden_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder(decoder_inputs)
        
        attn_layer = AttentionLayer(name='attention_layer')
        attn_outputs, attn_states = attn_layer([sent_emb_repeat, decoder_outputs])
        decoder_concat_inputs = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_outputs])
        
        dense = Dense(self.output_dim, activation='softmax')
        dense_time = TimeDistributed(dense)
        decoder_pred = dense_time(decoder_concat_inputs)
        
        self.model = Model(inputs=[word_encoder_inputs, decoder_inputs], outputs=decoder_pred)
        
        # define inference encoder
        word_encoder_inf_inputs = Input(batch_shape=(None, self.input_seq_length))
        sent_inf_emb = encoder(word_encoder_inf_inputs)
        sent_inf_emb_repeat = RepeatVector(self.output_seq_length)(sent_inf_emb)
        self.encoder_model = Model(word_encoder_inf_inputs, sent_inf_emb_repeat)

        # define decoder inference model
        decoder_init_hstate = Input(batch_shape=(None, self.decoder_hidden_dim))
        decoder_init_cstate = Input(batch_shape=(None, self.decoder_hidden_dim))
        decoder_init_state = [decoder_init_hstate, decoder_init_cstate]
        
        encoder_inf_emb = Input(batch_shape=(None, self.output_seq_length, self.conv2_dim))
        decoder_inf_inputs = Input(batch_shape=(None, 1, self.output_dim))
        decoder_inf_outputs, state_h, state_c = decoder(decoder_inf_inputs)
        decoder_inf_states = [state_h, state_c]
        self.decoder_model = Model(inputs=decoder_inf_inputs, outputs=[decoder_inf_outputs, decoder_inf_states])
        
        attn_inf_outputs, attn_inf_states = attn_layer([encoder_inf_emb, decoder_inf_outputs])
        decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_inf_outputs, attn_inf_outputs])
        decoder_inf_pred = TimeDistributed(dense)(decoder_inf_concat)
        self.inference_model = Model(inputs=[encoder_inf_emb, decoder_init_state, decoder_inf_inputs], outputs=[decoder_inf_pred, attn_inf_states, decoder_inf_states])

        self.model.compile(loss=self.loss, 
              optimizer=self.optimizer, 
              metrics=self.metrics)

    def predict_sequence(self, source, max_seq_len, id_to_mesh, start_token='start', end_token='end'):
        # encode source
        sent_emb = self.encoder_model.predict(source)
        print(sent_emb.shape)

        # start of sequence input
        in_text = []
        in_seq = dpt.pad_sequence(in_text, max_seq_len, 
                                  start_token=start_token, 
                                  end_token=end_token)
        # integer encoder
        in_seq_ids = dpt.mesh_to_vectors([in_seq], dicts_dir=data_dir+'dicts/', 
                       load_dicts=True, save=False)[0]
        in_seq_ids = in_seq_ids.tolist()
        print(in_seq_ids)
        # one-hot encode
        in_seq_onehot = one_hot_encode(in_seq_ids, mesh_vocab_length)
        in_seq_onehot = np.array(in_seq_onehot).reshape(1, 1, in_seq_onehot.shape[-1])
        
        # initialise decoder state
        dec_out, attention, h, c = self.decoder_model.predict(in_seq_onehot)
        dec_state = [h,c]
        
        # collect predictions
        output = []
        attention_weights = []
        
        for t in range(self.output_seq_length):
            dec_out, attention, h, c = self.inference_model.predict([enc_outs] + dec_state + [target_seq])
            dec_state = [h, c]
            
            # store prediction
            output.append(dec_out[0,0,:])
            dec_ind = np.argmax(dec_out, axis=-1)[0, 0]
            print(dec_ind)
            attention_weights.append((dec_ind, attention))
            target_seq = dec_out
        return np.array(output)  
    
    def get_params(self):
        self.params = {}
        keys = ['model_name', 'epochs', 'batch_size', 'input_dim', 'output_dim', 'conv1_dim', 'conv1_kernel', 'conv2_dim', 'conv2_kernel', 'embedding_dim', 'input_seq_length', 'output_seq_length',  'decoder_hidden_dim', 'loss_name']
        for key in keys:
            value = self.__dict__[key]
            if isinstance(value, np.int64):
                value = int(value)
            self.params[key] = value
        return self.params
    
    def save_weights_history(self, output_dir):
        param_dict = self.get_params()
        param_fn = 'param_{}_embdim_{}_conv1_{}_conv2_{}_epochs_{}.pkl'.format(self.model_name,
                                                                     self.embedding_dim,
                                                                     self.conv1_dim, 
                                                                     self.conv2_dim, 
                                                                     self.epochs)
        with open(output_dir + param_fn, 'wb') as f:
            pickle.dump(param_dict, f)
        
        history_dict = self.history.history
        history_fn = 'history_{}_embdim_{}_conv1_{}_conv2_{}_epochs_{}.pkl'.format(self.model_name,
                                                                     self.embedding_dim,
                                                                     self.conv1_dim, 
                                                                     self.conv2_dim, 
                                                                     self.epochs)
        with open(output_dir + history_fn, 'wb') as f:
            pickle.dump(history_dict, f)
        
        weights_fn = 'encoder_weights_{}_embdim_{}_conv1_{}_conv2_{}_epochs_{}.h5'.format(self.model_name,
                                                                     self.embedding_dim,
                                                                     self.conv1_dim, 
                                                                     self.conv2_dim, 
                                                                     self.epochs)
        self.encoder_model.save_weights(output_dir + weights_fn)
        
        weights_fn = 'decoder_weights_{}_embdim_{}_conv1_{}_conv2_{}_epochs_{}.h5'.format(self.model_name,
                                                                     self.embedding_dim,
                                                                     self.conv1_dim, 
                                                                     self.conv2_dim, 
                                                                     self.epochs)
        self.decoder_model.save_weights(output_dir + weights_fn)
        
        weights_fn = 'inference_weights_{}_embdim_{}_conv1_{}_conv2_{}_epochs_{}.h5'.format(self.model_name,
                                                                     self.embedding_dim,
                                                                     self.conv1_dim, 
                                                                     self.conv2_dim, 
                                                                     self.epochs)
        self.inference_model.save_weights(output_dir + weights_fn)
        
    def load_weights_history(self, output_dir):
        history_fn = 'history_{}_embdim_{}_conv1_{}_conv2_{}_epochs_{}.pkl'.format(self.model_name,
                                                                     self.embedding_dim,
                                                                     self.conv1_dim, 
                                                                     self.conv2_dim, 
                                                                     self.epochs)
        with open(output_dir + history_fn, 'rb') as f:
            self.history = pickle.load(f)

        weights_fn = 'encoder_weights_{}_embdim_{}_conv1_{}_conv2_{}_epochs_{}.h5'.format(self.model_name,
                                                                     self.embedding_dim,
                                                                     self.conv1_dim, 
                                                                     self.conv2_dim, 
                                                                     self.epochs)
        self.encoder_model.load_weights(output_dir + weights_fn)
        
        weights_fn = 'decoder_weights_{}_embdim_{}_conv1_{}_conv2_{}_epochs_{}.h5'.format(self.model_name,
                                                                     self.embedding_dim,
                                                                     self.conv1_dim, 
                                                                     self.conv2_dim, 
                                                                     self.epochs)
        self.decoder_model.load_weights(output_dir + weights_fn)
        
        weights_fn = 'inference_weights_{}_embdim_{}_conv1_{}_conv2_{}_epochs_{}.h5'.format(self.model_name,
                                                                     self.embedding_dim,
                                                                     self.conv1_dim, 
                                                                     self.conv2_dim, 
                                                                     self.epochs)
        self.inference_model.load_weights(output_dir + weights_fn)
    
# class CNN2HierSeqAtt_2(CNN2HierSeq):
#     def __init__(self, **kwargs):
#         self.epochs = 50
#         self.optimizer = 'adam'
#         self.metrics = ['accuracy']
#         self.loss = 'categorical_crossentropy'
#         self.loss_name = 'categorical_crossentropy'
#         self.batch_size = 128
#         self.input_dim = 100 # input vocab length
#         self.output_dim = 100 # output vocab length
#         self.embedding_dim = 256
#         self.decoder_hidden_dim = 256
#         self.conv1_dim = 128
#         self.conv1_kernel = 3
#         self.conv2_dim = 128
#         self.conv2_kernel = 3
#         self.input_seq_length = 100 # max num words per input sentence
#         self.output_seq_length = 20 # max num output words
#         self.dropout_rate = 0.5
#         self.verbose = False
#         self.__dict__.update(kwargs)
        
#     def build_model(self):
#         # define word CNN encoder
#         word_encoder_inputs = Input(batch_shape=(self.batch_size, self.input_seq_length))
#         word_encoder_emb = Embedding(self.input_dim, 
#                                      self.embedding_dim, 
#                                      input_length=self.input_seq_length)(word_encoder_inputs)
#         word_conv_outputs = Conv1D(self.conv1_dim, self.conv1_kernel, activation='relu')(word_encoder_emb)
#         word_conv_outputs = MaxPooling1D(3)(word_conv_outputs)
#         print('Word conv output: ', K.int_shape(word_conv_outputs))
#         word_attention_layer = AttentionWithContext(name='word_attention_layer')
#         word_attention = word_attention_layer(word_conv_outputs)
#         print('Word attention: ', K.int_shape(word_attention))
#         word_attention = Reshape((1, self.conv1_dim))(word_attention)
#         word_attention_outputs = Lambda(lambda x:x[1]*x[0], output_shape=lambda x:x[0])([word_conv_outputs, word_attention])
#         print('Word attention outputs: ', K.int_shape(word_attention_outputs))
#         sent_emb = Lambda(lambda x:K.sum(x, axis=1), output_shape=lambda x:(x[0],x[2]))(word_attention_outputs)
#         print('Sentence emb: ', K.int_shape(sent_emb))
        
#         # define sentence CNN encoder
#         sent_conv_outputs = Conv1D(self.conv2_dim, self.conv2_kernel, activation='relu')(word_attention_outputs)
#         sent_conv_outputs = MaxPooling1D(3)(sent_conv_outputs)
#         print('Sent conv output: ', K.int_shape(sent_conv_outputs))
#         sent_attention_layer = AttentionWithContext(name='sent_attention_layer')
#         sent_attention = sent_attention_layer(sent_conv_outputs)
#         print('Sent attention: ', K.int_shape(sent_attention))
#         sent_attention = Reshape((1, self.conv2_dim))(sent_attention)
#         sent_attention_outputs = Lambda(lambda x:x[1]*x[0], output_shape=lambda x:x[0])([sent_conv_outputs, sent_attention])
#         print('Sent attention output: ', K.int_shape(sent_attention_outputs))
#         doc_emb = Lambda(lambda x:K.sum(x, axis=1), output_shape=lambda x:(x[0],x[2]))(sent_attention_outputs)
#         print('Document emb: ', K.int_shape(doc_emb))

#         # define sequential training decoder
#         decoder_inputs = Input(batch_shape=(self.batch_size, self.output_seq_length, self.output_dim))
#         decoder = LSTM(self.decoder_hidden_dim, return_sequences=True, return_state=True)
#         decoder_outputs, _, _ = decoder(decoder_inputs)
        
#         # define attention
#         dec_attention_layer = AttentionWithContext(name='dec_attention_layer')
#         dec_attention = dec_attention_layer(decoder_inputs)
#         print('Decoder attention output: ', K.int_shape(dec_attention))
#         dec_attention = Reshape((1, self.output_dim))(dec_attention)
#         decoder_attention_inputs = Lambda(lambda x:x[1]*x[0], output_shape=lambda x:x[0])([decoder_inputs, dec_attention])
        
#         word_attention_time = RepeatVector(self.output_seq_length)(sent_emb)
#         sent_attention_time = RepeatVector(self.output_seq_length)(doc_emb)
        
#         decoder_concat_inputs = Concatenate(axis=-1, name='concat_layer')([decoder_attention_inputs, word_attention_time, sent_attention_time])
        
#         dense = Dense(self.output_dim, activation='softmax')
#         dense_time = TimeDistributed(dense)
#         decoder_pred = dense_time(decoder_concat_inputs)
        
#         self.model = Model(inputs=[word_encoder_inputs, decoder_inputs], outputs=decoder_pred)
        
#         self.model.compile(loss=self.loss, 
#               optimizer=self.optimizer, 
#               metrics=self.metrics)
        
#         # define inference encoder
#         word_encoder_inf_inputs = Input(batch_shape=(None, self.input_seq_length))
#         sent_inf_emb = encoder(word_encoder_inf_inputs)
#         sent_inf_emb_repeat = RepeatVector(self.output_seq_length)(sent_inf_emb)
#         self.encoder_model = Model(word_encoder_inf_inputs, sent_inf_emb)

#         # define decoder inference model
#         decoder_inf_inputs = Input(batch_shape=(None, 1, self.output_dim))
#         encoder_inf_states = Input(batch_shape=(None, self.input_seq_length, self.conv2_dim))
# #         decoder_init_hstate = Input(batch_shape=(None, self.decoder_hidden_dim))
# #         decoder_init_cstate = Input(batch_shape=(None, self.decoder_hidden_dim))
# #         decoder_init_state = [decoder_init_hstate, decoder_init_cstate]

#         decoder_inf_outputs, state_h, state_c = decoder(decoder_inf_inputs)
#         decoder_inf_states = [state_h, state_c]
#         attn_inf_outputs, attn_inf_states = attn_layer([encoder_inf_states, decoder_inf_outputs])
#         decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_inf_outputs, attn_inf_outputs])
#         decoder_inf_pred = TimeDistributed(dense)(decoder_inf_concat)
#         self.decoder_model = Model(inputs=[encoder_inf_states, decoder_inf_inputs],
#                               outputs=[decoder_inf_pred, attn_inf_states, decoder_inf_states])

# def class HierSeq2HierSeqAtt(Seq2Seq):
#     def __init__(self, **kwargs):
#         self.epochs = 50
#         self.optimizer = 'adam'
#         self.metrics = ['accuracy']
#         self.loss = 'categorical_crossentropy'
#         self.loss_name = 'categorical_crossentropy'
#         self.batch_size = 128
#         self.input_dim = 100 # input vocab length
#         self.output_dim = 100 # output vocab length
#         self.sent_dim = 10 # max sentence length
#         self.embedding_dim = 256
#         self.hidden_dim = 256
#         self.input_word_seq_length = 10 # max num words per input sentence
#         self.input_sent_seq_length = 5 # max num input sentence
#         self.output_seq_length = 10 # max num output words
#         self.dropout_rate = 0.5
#         self.verbose = False
#         self.__dict__.update(kwargs)
        
#     def build_model(self): 

# def class HierCNN2HierSeqAtt(Seq2Seq):