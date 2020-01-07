from tensorflow.keras import backend as K
from tensorflow.keras.backend import dot 
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, SeparableConv1D, Concatenate, Embedding, RepeatVector
from tensorflow.keras.layers import Flatten, Dropout, Input, LSTM, BatchNormalization, Activation, TimeDistributed, Lambda, Reshape
from utils.attention import AttentionLayer, AttentionLayer2, HierarchicalAttentionNetwork
from utils.attention_with_context import AttentionWithContext, AttLayer
from utils.custom_losses import binary_recall_specificity_loss, combined_loss
import pickle

import os
import numpy as np

def one_hot_decode(encoded_seq):
    return [np.argmax(vector) for vector in encoded_seq]

class Seq2Seq(object):
    def __init__(self, **kwargs):
        self.model_name = 'seq2seq'
        self.emb = True
        self.epochs = 50
        self.optimizer = 'adam'
        self.metrics = ['accuracy']
        self.loss = 'categorical_crossentropy'
        self.loss_name = 'categorical_crossentropy'
        self.batch_size = 128
        self.input_dim = 100 # input vocab length
        self.output_dim = 100 # output vocab length
        self.encoder_emb_dim = 512 # embedding dim of words going into encoder
        self.decoder_emb_dim = 256 # embedding dim of words going into decoder
        self.hidden_dim = 256 # hidden dim of LSTM
        self.input_seq_length = 10 # max num input words
        self.output_seq_length = 10 # max num output words
        self.dropout_rate = 0.5
        self.verbose = False
        self.__dict__.update(kwargs)
        
    def build_model(self):
        # define training encoder
#         if self.emb:
#             encoder_inputs = Input(shape=(self.input_seq_length,))
#             encoder_inputs = Embedding(self.input_dim, self.encoder_emb_dim)(encoder_inputs)
#         else:
        encoder_inputs = Input(batch_shape=(self.batch_size, self.input_seq_length, self.input_dim))
        encoder = LSTM(self.hidden_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]
        
        # define training decoder
        decoder_inputs = Input(shape=(None, self.output_dim))
        decoder_lstm = LSTM(self.hidden_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.output_dim, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        
        # define inference encoder
        self.encoder_model = Model(encoder_inputs, encoder_states)
        
        # define inference decoder
        decoder_state_input_h = Input(shape=(self.hidden_dim,))
        decoder_state_input_c = Input(shape=(self.hidden_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        
        self.model.compile(loss=self.loss, 
                      optimizer=self.optimizer, 
                      metrics=self.metrics)
        
        
    def run_experiment(self, sentence_train, ents_shifted_train, ents_train, sentence_val, ents_shifted_val, ents_val):
        self.build_model()
        self.history = self.model.fit([sentence_train, ents_shifted_train], ents_train,
                    epochs=self.epochs,
                    verbose=self.verbose,
                    validation_data=([sentence_val, ents_shifted_val], ents_val),
                    batch_size=self.batch_size)
        
    def run_experiment_batch(self, sentence_train, ents_train, ents_shifted_train, sentence_val, ents_val, ents_shifted_val):
        self.build_model()
        self.history = self.model.fit_generator([sentence_train, ents_train], ents_shifted_train,
                    epochs=self.epochs,
                    verbose=self.verbose,
                    validation_data=([sentence_val, ents_val], ents_shifted_val),
                    batch_size=self.batch_size)

    def predict_sequence(self, source):
        # encode
        state = self.encoder_model.predict(source)
        # start of sequence input
        target_seq = np.array([0.0 for _ in range(self.output_dim)]).reshape(1, 1, self.output_dim)
        # collect predictions
        output = list()
        for t in range(self.output_seq_length):
            # predict next word
            yhat, h, c = self.decoder_model.predict([target_seq] + state)
            # store prediction
            output.append(yhat[0,0,:])
            # update state
            state = [h, c]
            # update target sequence
            target_seq = yhat
        return np.array(output)

#     def decode_sequence(self, input_seq, id_to_ent, ent_to_id, start_token='start', end_token='end'):
#         states_value = self.encoder_model.predict(input_seq)
#         #print(states_value[0].shape)
#         target_seq = np.zeros((1, 1, self.output_dim))
#         target_seq[0, 0, ent_to_id[start_token]] = 1.

#         # Sampling loop for a batch of sequences
#         # (to simplify, here we assume a batch of size 1).
#         stop_condition = False
#         decoded_sentence = []
#         while not stop_condition:
#             output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)
#             #print(output_tokens.shape)
#             # Sample a token
#             sampled_token_index = np.argmax(output_tokens[0, -1, :])
#             sampled_char = id_to_ent[sampled_token_index]
#             decoded_sentence.append(sampled_char)

#             # Exit condition: either hit max length
#             # or find stop character.
#             if (sampled_char == end_token or
#                len(decoded_sentence) > self.output_seq_length):
#                 stop_condition = True

#             # Update the target sequence (of length 1).
#             target_seq = np.zeros((1, 1, self.output_dim))
#             target_seq[0, 0, sampled_token_index] = 1.

#             # Update states
#             states_value = [h, c]

#         return decoded_sentence
        
    def get_params(self):
        self.params = {}
        keys = ['model_name', 'emb', 'epochs', 'batch_size', 'input_dim', 'output_dim', 'input_seq_length', 'output_seq_length', 'encoder_emb_dim', 'hidden_dim', 'loss_name']
        for key in keys:
            value = self.__dict__[key]
            if isinstance(value, np.int64):
                value = int(value)
            self.params[key] = value
        return self.params
    
    def get_filename(self):
        fn = '_encoderembdim_{}_hiddendim_{}_epochs_{}'.format(self.encoder_emb_dim,
                                                              self.hidden_dim,
                                                              self.epochs)
        return fn
        
    def save_weights_history(self, output_dir):
        param_dict = self.get_params()
        param_fn = 'param' + self.get_filename() + '.pkl'
        with open(output_dir + param_fn, 'wb') as f:
            pickle.dump(param_dict, f)
        
        history_dict = self.history.history
        history_fn = 'history' + self.get_filename() + '.pkl'
        with open(output_dir + history_fn, 'wb') as f:
            pickle.dump(history_dict, f)
        
        weights_fn = 'encoder_weights' + self.get_filename() + '.h5'
        self.encoder_model.save_weights(output_dir + weights_fn)
        
        weights_fn = 'decoder_weights' + self.get_filename() + '.h5'
        self.decoder_model.save_weights(output_dir + weights_fn)

    def load_weights_history(self, output_dir):
        history_fn = 'history' + self.get_filename() + '.pkl'
        with open(output_dir + history_fn, 'rb') as f:
            self.history = pickle.load(f)

        weights_fn = 'encoder_weights' + self.get_filename() + '.h5'
        self.encoder_model.load_weights(output_dir + weights_fn)
        
        weights_fn = 'decoder_weights' + self.get_filename() + '.h5'
        self.decoder_model.load_weights(output_dir + weights_fn)
        
class Seq2SeqAtt(Seq2Seq):
    def __init__(self, **kwargs):
        self.model_name = 'seq2seqatt'
        self.emb = True
        self.epochs = 50
        self.optimizer = 'adam'
        self.metrics = ['accuracy']
        self.loss = 'categorical_crossentropy'
        self.loss_name = 'categorical_crossentropy'
        self.batch_size = 128
        self.input_dim = 100 # input vocab length
        self.output_dim = 100 # output vocab length
        self.encoder_emb_dim = 512 # embedding dim of words going into encoder
        self.decoder_emb_dim = 256 # embedding dim of words going into decoder
        self.hidden_dim = 256 # hidden dim of LSTM
        self.input_seq_length = 10 # max num input words
        self.output_seq_length = 10 # max num output words
        self.dropout_rate = 0.5
        self.verbose = False
        self.__dict__.update(kwargs)
        
    def build_model(self):
        # define training encoder
#         if self.emb:
#             encoder_inputs = Input(shape=(self.input_seq_length,))
#             encoder_inputs = Embedding(self.input_dim, self.encoder_emb_dim)(encoder_inputs)
#         else:
        encoder_inputs = Input(batch_shape=(self.batch_size, self.input_seq_length, self.input_dim))
        encoder = LSTM(self.hidden_dim, return_sequences=True, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]
        
        # define training decoder
        decoder_inputs = Input(batch_shape=(self.batch_size, self.output_seq_length, self.output_dim))
        decoder = LSTM(self.hidden_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)
        
        # define attention
        attn_layer = AttentionLayer(name='attention_layer')
        attn_outputs, attn_states = attn_layer([encoder_outputs, decoder_outputs])
        decoder_concat_inputs = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_outputs])
        
        dense = Dense(self.output_dim, activation='softmax')
        dense_time = TimeDistributed(dense)
        decoder_pred = dense_time(decoder_concat_inputs)
        
        self.model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
        
        # define inference model
#         if self.emb:
#             encoder_inf_inputs = Input(shape=(self.input_seq_length,))
#             encoder_inf_inputs = Embedding(self.input_dim, self.encoder_emb_dim)(encoder_inf_inputs)
#         else:
        encoder_inf_inputs = Input(batch_shape=(None, self.input_seq_length, self.input_dim))
        encoder_inf_outputs, state_h, state_c = encoder(encoder_inf_inputs)
        encoder_inf_states = [state_h, state_c]
        self.encoder_model = Model(inputs=encoder_inf_inputs, outputs=[encoder_inf_outputs, encoder_inf_states])

        # define decoder inference model
        decoder_inf_inputs = Input(batch_shape=(None, 1, self.output_dim))
        encoder_inf_states = Input(batch_shape=(None, self.input_seq_length, self.hidden_dim))
        decoder_init_hstate = Input(batch_shape=(None, self.hidden_dim))
        decoder_init_cstate = Input(batch_shape=(None, self.hidden_dim))
        decoder_init_state = [decoder_init_hstate, decoder_init_cstate]

        decoder_inf_outputs, state_h, state_c = decoder(decoder_inf_inputs, initial_state=decoder_init_state)
        decoder_inf_states = [state_h, state_c]
        attn_inf_outputs, attn_inf_states = attn_layer([encoder_inf_states, decoder_inf_outputs])
        decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_inf_outputs, attn_inf_outputs])
        decoder_inf_pred = TimeDistributed(dense)(decoder_inf_concat)
        self.decoder_model = Model(inputs=[encoder_inf_states, decoder_init_state, decoder_inf_inputs],
                              outputs=[decoder_inf_pred, attn_inf_states, decoder_inf_states])
        
        self.model.compile(loss=self.loss, 
                      optimizer=self.optimizer, 
                      metrics=self.metrics)
        
    def predict_sequence(self, source):
        # encode source
        enc_outs, state_h, state_c = self.encoder_model.predict(source)
        enc_last_state = [state_h, state_c]
        
        dec_state = enc_last_state
        attention_weights = []
        
        # start of sequence input
        target_seq = np.array([0.0 for _ in range(self.output_dim)]).reshape(1, 1, self.output_dim)
        
        # collect predictions
        output = []
        
        for t in range(self.output_seq_length):
            dec_out, attention, h, c = self.decoder_model.predict([enc_outs] + dec_state + [target_seq])
            dec_state = [h, c]
            
            # store prediction
            output.append(dec_out[0,0,:])
            dec_ind = np.argmax(dec_out, axis=-1)[0, 0]
            
            attention_weights.append((dec_ind, attention))
            target_seq = dec_out
        return np.array(output)  

class HierSeq2Seq(Seq2Seq):
    def __init__(self, **kwargs):
        self.model_name = 'hierseq2seq'
        self.emb = True
        self.epochs = 50
        self.optimizer = 'adam'
        self.metrics = ['accuracy']
        self.loss = 'categorical_crossentropy'
        self.loss_name = 'categorical_crossentropy'
        self.batch_size = 128
        self.input_dim = 100 # input vocab length
        self.output_dim = 100 # output vocab length
        self.encoder_emb_dim = 256 # embedding dim of words going into encoder
        self.decoder_emb_dim = 256 # embedding dim of words going into decoder
        self.hidden_dim = 256
        self.input_word_seq_length = 10 # max num words per input sentence
        self.input_sent_seq_length = 5 # max num input sentences
        self.output_seq_length = 10 # max num output words
        self.dropout_rate = 0.5
        self.verbose = False
        self.__dict__.update(kwargs)
        
    def build_model(self): 
        # define word-level encoder
#         if self.emb:
        word_encoder_inputs = Input(shape=(self.input_word_seq_length,))
        word_encoder_emb = Embedding(self.input_dim, 
                                        self.encoder_emb_dim, 
                                        input_length=self.input_word_seq_length)(word_encoder_inputs)
#         else:
#         word_encoder_inputs = Input(batch_shape=(self.batch_size, 
#                                                  self.input_word_seq_length, 
#                                                  self.input_dim))
          
        word_encoder = LSTM(self.hidden_dim, return_sequences=True, return_state=True)
        word_encoder_outputs, state_h, state_c = word_encoder(word_encoder_emb)
        word_encoder_states = [state_h, state_c]
        sentence_emb = Lambda(lambda x:K.sum(x, axis=1), output_shape=lambda x:(x[0],x[2]))(word_encoder_outputs)
        self.sentence_encoder_model = Model(word_encoder_inputs, sentence_emb)
        
        # define sentence-level encoder
        sent_encoder_inputs = Input(batch_shape=(None, 
                                                 self.input_sent_seq_length,
                                                 self.input_word_seq_length))
        sent_encoder_t = TimeDistributed(self.sentence_encoder_model)(sent_encoder_inputs)
        sent_encoder = LSTM(self.hidden_dim, return_state=True)
        sent_encoder_outputs, state_h, state_c = sent_encoder(sent_encoder_t)
        sent_encoder_states = [state_h, state_c]
        
        # define training decoder
        #decoder_inputs = Input(batch_shape=(None, self.output_seq_length, self.output_dim))
        decoder_inputs = Input(shape=(None, self.output_dim))
        decoder_lstm = LSTM(self.hidden_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=sent_encoder_states)
        decoder_dense = Dense(self.output_dim, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        self.model = Model([sent_encoder_inputs, decoder_inputs], decoder_outputs)
        
        # define inference encoder
#         sent_encoder_inf_inputs = Input(batch_shape=(None, 
#                                                  self.input_sent_seq_length,
#                                                  self.input_word_seq_length))
#         sent_encoder_inf_t = TimeDistributed(self.sentence_encoder_model)(sent_encoder_inf_inputs)
#         sent_encoder_inf = LSTM(self.hidden_dim, return_state=True)
#         sent_encoder_inf_outputs, state_h, state_c = sent_encoder_inf(sent_encoder_inf_t)
#         sent_encoder_inf_states = [state_h, state_c]
        self.encoder_model = Model(sent_encoder_inputs, sent_encoder_states)
        
        # define inference decoder
        decoder_state_input_h = Input(shape=(self.hidden_dim,))
        decoder_state_input_c = Input(shape=(self.hidden_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        
        self.model.compile(loss=self.loss, 
                      optimizer=self.optimizer, 
                      metrics=self.metrics)
        
    def predict_sequence(self, source):
        # encode
        state = self.encoder_model.predict(source)
        #sent_emb = self.sentence_encoder_model.predict(source)
        # start of sequence input
        target_seq = np.array([0.0 for _ in range(self.output_dim)]).reshape(1, 1, self.output_dim)
        # collect predictions
        output = list()
        for t in range(self.output_seq_length):
            # predict next word
            yhat, h, c = self.decoder_model.predict([target_seq] + state)
            # store prediction
            output.append(yhat[0,0,:])
            # update state
            state = [h, c]
            # update target sequence
            target_seq = yhat
        return np.array(output)

    def get_params(self):
        self.params = {}
        keys = ['model_name', 'emb', 'epochs', 'batch_size', 'input_dim', 'output_dim', 'encoder_emb_dim', 'input_word_seq_length', 'input_sent_seq_length', 'output_seq_length',  'hidden_dim', 'loss_name']
        for key in keys:
            value = self.__dict__[key]
            if isinstance(value, np.int64):
                value = int(value)
            self.params[key] = value
        return self.params
    
    def get_filename(self):
        fn = '_encoderembdim_{}_hiddendim_{}_epochs_{}'.format(self.encoder_emb_dim,
                                                              self.hidden_dim,
                                                              self.epochs)
        return fn
        
class HierSeq2SeqAtt(Seq2Seq):
    def __init__(self, **kwargs):
        self.model_name = 'hierseq2seqatt'
        self.emb = True
        self.epochs = 50
        self.optimizer = 'adam'
        self.metrics = ['accuracy']
        self.loss = 'categorical_crossentropy'
        self.loss_name = 'categorical_crossentropy'
        self.batch_size = 128
        self.input_dim = 100 # input vocab length
        self.output_dim = 100 # output vocab length
        self.encoder_emb_dim = 256 # embedding dim of words going into encoder
        self.decoder_emb_dim = 256 # embedding dim of words going into decoder
        self.hidden_dim = 256
        self.input_word_seq_length = 10 # max num words per input sentence
        self.input_sent_seq_length = 5 # max num input sentences
        self.output_seq_length = 10 # max num output words
        self.dropout_rate = 0.5
        self.verbose = False
        self.__dict__.update(kwargs)
        
    def build_model(self):
        # define word encoder
        word_encoder_inputs = Input(shape=(self.input_word_seq_length,))
        word_encoder_emb = Embedding(self.input_dim, 
                                        self.encoder_emb_dim, 
                                        input_length=self.input_word_seq_length)(word_encoder_inputs)
#         else:
#         word_encoder_inputs = Input(batch_shape=(self.batch_size, 
#                                                  self.input_word_seq_length, 
#                                                  self.input_dim))
#         word_encoder_inputs = Input(batch_shape=(None, self.input_word_seq_length, self.input_dim))
        print('Word encoder emb: ', K.int_shape(word_encoder_emb))
        word_encoder = LSTM(self.hidden_dim, return_sequences=True, return_state=True)
        word_encoder_outputs, state_h, state_c = word_encoder(word_encoder_emb)
        word_encoder_states = [state_h, state_c]
        sentence_emb = Lambda(lambda x:K.sum(x, axis=1), output_shape=lambda x:(x[0],x[2]))(word_encoder_outputs)
        self.sentence_encoder_model = Model(word_encoder_inputs, sentence_emb)
        print('Sent emb: ', K.int_shape(sentence_emb))
        # define sentence-level encoder
        sent_encoder_inputs = Input(batch_shape=(None, 
                                                 self.input_sent_seq_length,
                                                 self.input_word_seq_length))
        sent_encoder_t = TimeDistributed(self.sentence_encoder_model)(sent_encoder_inputs)
        sent_encoder = LSTM(self.hidden_dim, return_sequences=True, return_state=True)
        sent_encoder_outputs, state_h, state_c = sent_encoder(sent_encoder_t)
        sent_encoder_states = [state_h, state_c]
        print('Encoder outputs: ', K.int_shape(sent_encoder_outputs))
        print('Encoder state: ', K.int_shape(state_h))
        encoder = Model(inputs=[sent_encoder_inputs], outputs=[sent_encoder_outputs, sent_encoder_states])
        
        # define training decoder
        #decoder_inputs = Input(batch_shape=(None, self.output_seq_length, self.output_dim))
        decoder_inputs = Input(shape=(None, self.output_dim))
        decoder_lstm = LSTM(self.hidden_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=sent_encoder_states)
        
        # define attention
        attn_layer = AttentionLayer(name='attention_layer')
        attn_outputs, attn_states = attn_layer([sent_encoder_outputs, decoder_outputs])
        decoder_concat_inputs = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_outputs])
        
        dense = Dense(self.output_dim, activation='softmax')
        dense_time = TimeDistributed(dense)
        decoder_pred = dense_time(decoder_concat_inputs)
        
        self.model = Model(inputs=[sent_encoder_inputs, decoder_inputs], outputs=decoder_pred)
        
        # define inference model
#         if self.emb:
#             encoder_inf_inputs = Input(shape=(self.input_seq_length,))
#             encoder_inf_inputs = Embedding(self.input_dim, self.encoder_emb_dim)(encoder_inf_inputs)
#         else:
        sent_encoder_inf_inputs = Input(batch_shape=(None, 
                                                 self.input_sent_seq_length,
                                                 self.input_word_seq_length))
        sent_encoder_inf_outputs, sent_encoder_inf_states = encoder(sent_encoder_inf_inputs)
        self.encoder_model = Model(inputs=sent_encoder_inf_inputs, outputs=[sent_encoder_inf_outputs, sent_encoder_inf_states])

        # define decoder inference model
        decoder_inf_inputs = Input(batch_shape=(None, 1, self.output_dim))
        sent_encoder_inf_states = Input(batch_shape=(None, self.input_sent_seq_length, self.hidden_dim))
        decoder_init_hstate = Input(batch_shape=(None, self.hidden_dim))
        decoder_init_cstate = Input(batch_shape=(None, self.hidden_dim))
        decoder_init_state = [decoder_init_hstate, decoder_init_cstate]

        decoder_inf_outputs, state_h, state_c = decoder_lstm(decoder_inf_inputs, initial_state=decoder_init_state)
        decoder_inf_states = [state_h, state_c]
        attn_inf_outputs, attn_inf_states = attn_layer([sent_encoder_inf_states, decoder_inf_outputs])
        decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_inf_outputs, attn_inf_outputs])
        decoder_inf_pred = TimeDistributed(dense)(decoder_inf_concat)
        self.decoder_model = Model(inputs=[sent_encoder_inf_states, decoder_init_state, decoder_inf_inputs],
                              outputs=[decoder_inf_pred, attn_inf_states, decoder_inf_states])
        
        self.model.compile(loss=self.loss, 
                      optimizer=self.optimizer, 
                      metrics=self.metrics)

    def predict_sequence(self, source):
        # encode source
        enc_outs, state_h, state_c = self.encoder_model.predict(source)
        enc_last_state = [state_h, state_c]
        
        dec_state = enc_last_state
        attention_weights = []
        
        # start of sequence input
        target_seq = np.array([0.0 for _ in range(self.output_dim)]).reshape(1, 1, self.output_dim)
        
        # collect predictions
        output = []
        
        for t in range(self.output_seq_length):
            dec_out, attention, h, c = self.decoder_model.predict([enc_outs] + dec_state + [target_seq])
            dec_state = [h, c]
            
            # store prediction
            output.append(dec_out[0,0,:])
            dec_ind = np.argmax(dec_out, axis=-1)[0, 0]
            
            attention_weights.append((dec_ind, attention))
            target_seq = dec_out
        return np.array(output) 
    
    def get_params(self):
        self.params = {}
        keys = ['model_name', 'emb', 'epochs', 'batch_size', 'input_dim', 'output_dim', 'encoder_emb_dim', 'decoder_emb_dim', 'input_word_seq_length', 'input_sent_seq_length', 'output_seq_length',  'hidden_dim', 'loss_name']
        for key in keys:
            value = self.__dict__[key]
            if isinstance(value, np.int64):
                value = int(value)
            self.params[key] = value
        return self.params
    
    def get_filename(self):
        fn = '_encoderembdim_{}_decoderembdim_{}_hiddendim_{}_epochs_{}'.format(self.encoder_emb_dim,
                                                                               self.decoder_emb_dim,
                                                                               self.hidden_dim,
                                                                               self.epochs)
        return fn