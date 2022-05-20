import io
import re
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import util 
import numpy as np
import util
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_addons as tfa
import os
  
  
class Seq2SeqRNN():
  def __init__(self, version, num_examples, BUFFER_SIZE, BATCH_SIZE, attention_type, train_path, 
               dev_path, test_path, embedding_dim, encoder_units, decoder_units):
    self.version = version    
    self.attention_type = attention_type
    self.num_examples = num_examples
    self.BUFFER_SIZE = BUFFER_SIZE 
    self.BATCH_SIZE = BATCH_SIZE 
    self.embedding_dim = embedding_dim
    self.encoder_units = encoder_units
    self.decoder_units = decoder_units
    print("Number of examples:", num_examples)
    print("Buffer size:", BUFFER_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Embedding dimension:", embedding_dim)
    print("Encoder units, Decoder units:", encoder_units, decoder_units)

    self.dataset_creator = self.RNNDataset(num_examples, train_path, dev_path, test_path)
    if os.path.exists("tokenizer.json"):
      tokenizer = util.object_load("tokenizer.json", int_keys=True)
    else:
      print("Building Tokenizer")
      tokenizer = keras.preprocessing.text.Tokenizer(filters='', oov_token='<OOV>', split=' ')
      questions = []
      answers = []
      QA_pairs = self.dataset_creator.create_dataset("RNN_data/RNN_dataset.txt")
      for pair in QA_pairs:
        answers.append(pair[1])
        questions.append(pair[0])
      tokenizer.fit_on_texts(questions + answers)
      util.object_save(tokenizer, "tokenizer.json", int_keys=True)    
    self.tokenizer = tokenizer
    self.train_dataset, self.val_dataset, self.test_dataset = self.dataset_creator.call(self.tokenizer,
                                                                                        self.BUFFER_SIZE, 
                                                                                        self.BATCH_SIZE)
    self.vocab_size = len(tokenizer.word_index)+1
    print("Vocab size: ", self.vocab_size)
    example_input_batch, example_target_batch = next(iter(self.train_dataset))
    self.max_length_input = example_input_batch.shape[1]
    self.max_length_output = example_target_batch.shape[1]
    print("Maximum question length:", self.max_length_input)
    print("Maximum answer length:", self.max_length_output)


    self.encoder = self.Encoder(self.tokenizer, self.vocab_size, self.embedding_dim, 
                                self.encoder_units, self.BATCH_SIZE)
    # sample input
    sample_hidden = self.encoder.initialize_hidden_state()
    sample_output, sample_h, sample_c = self.encoder(example_input_batch, sample_hidden)
    print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    print ('Encoder h vector shape: (batch size, units) {}'.format(sample_h.shape))
    print ('Encoder c vector shape: (batch size, units) {}'.format(sample_c.shape))
    
    # Test decoder stack  
    self.decoder = self.Decoder(self.tokenizer, self.vocab_size, self.embedding_dim, self.decoder_units, 
                                self.BATCH_SIZE, self.attention_type, self.max_length_input,
                                self.max_length_output)
    sample_x = tf.random.uniform((self.BATCH_SIZE, self.max_length_output))
    self.decoder.attention_mechanism.setup_memory(sample_output)
    initial_state = self.decoder.build_initial_state([sample_h, sample_c], tf.float32)

    sample_decoder_outputs = self.decoder(sample_x, initial_state)
    print("Decoder output shape: (batch size, ?, vocab size)", sample_decoder_outputs.rnn_output.shape)
    
    self.optimizer = tf.keras.optimizers.Adam()
    self.checkpoint_dir = f'./training_checkpoints_{version}'
    self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                          encoder=self.encoder,
                                          decoder=self.decoder)
        
    
                 
  class RNNDataset:
    def __init__(self, num_examples, train_path, dev_path, test_path):
      self.num_examples = num_examples
      self.train_path, self.dev_path, self.test_path = train_path, dev_path, test_path
        
        
    ## Step 1 and Step 2 
    def preprocess_sentence(self, w):
      w = w.lower().strip()
      w = re.sub("(<br \/>)", " ", w)

      # creating a space between a word and the punctuation following it
      # eg: "he is a boy." => "he is a boy ."
      # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
      w = re.sub(r"([?,.!])", r" \1 ", w)
      w = re.sub(r'[" "]+', " ", w)

      # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
      w = re.sub(r"[^a-zA-Z?.!,0-9'`$%]+", " ", w)

      w = w.strip()

      # adding a start and an end token to the sentence
      # so that the model know when to start and stop predicting.
      w = '<start> ' + w + ' <end>'
      
      return w


    def create_dataset(self, path):
      # path : path to data file
      # n : Limit the total number of training example for faster training (set num_examples = len(lines) to use full data)
      lines = io.open(path).read().strip().split('\n')
      if self.num_examples == -1:
        self.num_examples = len(lines)
      QA_pairs = []
      for l in lines[:self.num_examples]:
          pair = []
          for w in l.split('\t'):
              pair.append(self.preprocess_sentence(w))
          QA_pairs.append(pair)                
      
      return QA_pairs


    def example_to_tensors(self, QA_pairs):
      # lang = list of sentences in a language
      # print(len(lang), "example sentence: {}".format(lang[0]))
      questions = []
      answers = []
      for pair in QA_pairs:
        try:
          answers.append(pair[1])
          questions.append(pair[0])
        except:
          pass
      
      ## tf.keras.preprocessing.text.Tokenizer.texts_to_sequences converts string (w1, w2, w3, ......, wn) 
      ## to a list of correspoding integer ids of words (id_w1, id_w2, id_w3, ...., id_wn)
      input_tensor = self.tokenizer.texts_to_sequences(questions) 
      target_tensor = self.tokenizer.texts_to_sequences(answers) 
      
      ## tf.keras.preprocessing.sequence.pad_sequences takes argument a list of integer id sequences 
      ## and pads the sequences to match the longest sequences in the given input
      input_tensor = pad_sequences(input_tensor, padding='post')
      target_tensor = pad_sequences(target_tensor, padding='post')
      
      return input_tensor, target_tensor


    def load_dataset(self, path):
      # creating cleaned input, output pairs
      QA_pairs = self.create_dataset(path)
      input_tensor, target_tensor = self.example_to_tensors(QA_pairs)
      
      return input_tensor, target_tensor


    def call(self, tokenizer, BUFFER_SIZE, BATCH_SIZE):
      self.tokenizer = tokenizer
      input_tensor_train, target_tensor_train = self.load_dataset(self.train_path)
      input_tensor_val, target_tensor_val = self.load_dataset(self.dev_path)
      input_tensor_test, target_tensor_test = self.load_dataset(self.test_path)

      train_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train))
      train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
      val_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val))
      val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)
      test_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_test, target_tensor_test))
      test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)
      
      return train_dataset, val_dataset, test_dataset
        
        
        
  class Encoder(tf.keras.Model):
    def __init__(self, tokenizer, vocab_size, embedding_dim, enc_units, batch_sz):
      super().__init__()
      self.batch_sz = batch_sz
      self.enc_units = enc_units
      # self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
      embedding_matrix = util.embedding_matrix(tokenizer, embedding_dim, extension="")
      self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, 
                  embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                  trainable=True)
      ##-------- LSTM layer in Encoder ------- ##
      self.lstm_layer = tf.keras.layers.LSTM(self.enc_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')


    def call(self, x, hidden):
      # print(f"\nencoder input shape: {x.shape}\n")
      x = self.embedding(x)
      output, h, c = self.lstm_layer(x, initial_state = hidden)
      return output, h, c


    def initialize_hidden_state(self):
      return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))]



  class Decoder(tf.keras.Model):
    def __init__(self, tokenizer, vocab_size, embedding_dim, dec_units, batch_sz, attention_type, 
                max_length_input, max_length_output):
      super().__init__()
      self.batch_sz = batch_sz
      self.dec_units = dec_units
      self.attention_type = attention_type
      self.max_length_output = max_length_output
      # self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
      embedding_matrix = util.embedding_matrix(tokenizer, embedding_dim, extension="")
      self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, 
                  embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                  trainable=True)
      #Final Dense layer on which softmax will be applied
      self.fc = tf.keras.layers.Dense(vocab_size)
      # Define the fundamental cell for decoder recurrent structure
      self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units)
      # Sampler
      self.sampler = tfa.seq2seq.sampler.TrainingSampler()
      # Create attention mechanism with memory = None
      self.attention_mechanism = self.build_attention_mechanism(self.dec_units, None, 
                                                                self.batch_sz*[max_length_input], 
                                                                self.attention_type)
      # Wrap attention mechanism with the fundamental rnn cell of decoder
      self.rnn_cell = self.build_rnn_cell()
      # Define the decoder with respect to fundamental rnn cell
      self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.fc)
      # self.decoder = tfa.seq2seq.BeamSearchDecoder(self.rnn_cell, beam_width=5, output_layer=self.fc)


    def build_rnn_cell(self):
      rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnn_cell, 
                                    self.attention_mechanism, attention_layer_size=self.dec_units)
      return rnn_cell


    def build_attention_mechanism(self, dec_units, memory, memory_sequence_length, attention_type='luong'):
      # ------------- #
      # typ: Which sort of attention (Bahdanau, Luong)
      # dec_units: final dimension of attention outputs 
      # memory: encoder hidden states of shape (batch_size, max_length_input, enc_units)
      # memory_sequence_length: 1d array of shape (batch_size) with every element set to max_length_input (for masking purpose)
      if(attention_type=='bahdanau'):
        return tfa.seq2seq.BahdanauAttention(units=dec_units, memory=memory, 
                                             memory_sequence_length=memory_sequence_length)
      else:
        return tfa.seq2seq.LuongAttention(units=dec_units, memory=memory, 
                                          memory_sequence_length=memory_sequence_length)


    def build_initial_state(self, encoder_state, Dtype):
      decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=self.batch_sz, dtype=Dtype)
      decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
      return decoder_initial_state


    def call(self, inputs, initial_state):
      # print(f"\ndecoder input shape: {inputs.shape}\n")
      x = self.embedding(inputs)
      outputs, _, _ = self.decoder(x, initial_state=initial_state, 
                                   sequence_length=self.batch_sz*[self.max_length_output-1])

      return outputs



  def train(self, EPOCHS, steps_per_epoch):
    total_loss = 0
    checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
    for epoch in range(EPOCHS):
      start = time.time()
      enc_hidden = self.encoder.initialize_hidden_state()
      total_loss = 0
      for (batch, (inp, targ)) in enumerate(self.train_dataset.take(steps_per_epoch)):
        batch_loss = self.train_step(inp, targ, enc_hidden)
        total_loss += batch_loss
        if batch % 100 == 0:
          print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                      batch,
                                                      batch_loss.numpy()))
      # saving (checkpoint) the model every 2 epochs
      if (epoch + 1) % 2 == 0:
          self.checkpoint.save(file_prefix = checkpoint_prefix)
      print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                          total_loss / steps_per_epoch))
      print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
      
      
  @tf.function
  def train_step(self, inp, targ, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
      enc_output, enc_h, enc_c = self.encoder(inp, enc_hidden)
      dec_input = targ[ : , :-1 ] # Ignore <end> token   
      real = targ[ : , 1: ]         # ignore <start> token
      # Set the AttentionMechanism object with encoder_outputs
      self.decoder.attention_mechanism.setup_memory(enc_output)
      # Create AttentionWrapperState as initial_state for decoder
      decoder_initial_state = self.decoder.build_initial_state([enc_h, enc_c], tf.float32)
      pred = self.decoder(dec_input, decoder_initial_state)
      logits = pred.rnn_output
      loss = self.loss_function(real, logits)
    variables = self.encoder.trainable_variables + self.decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))
    return loss


  def loss_function(self, real, pred):
    # real shape = (BATCH_SIZE, max_length_output)
    # pred shape = (BATCH_SIZE, max_length_output, tar_vocab_size )
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = cross_entropy(y_true=real, y_pred=pred)
    mask = tf.logical_not(tf.math.equal(real,0))   #output 0 for y=0 else output 1
    mask = tf.cast(mask, dtype=loss.dtype)  
    loss = mask*loss
    loss = tf.reduce_mean(loss)
    return loss


  def evaluate_question(self, question):
    question = self.dataset_creator.preprocess_sentence(question)
    inputs = []
    for i in question.split(' '):
      try:
        inputs.append(self.tokenizer.word_index[i])
      except:
        inputs.append(0)
    inputs = pad_sequences([inputs], maxlen=self.max_length_input, padding='post')
    
    inputs = tf.convert_to_tensor(inputs)
    inference_batch_size = inputs.shape[0]
    
    enc_start_state = [tf.zeros((inference_batch_size, self.encoder_units)), 
                      tf.zeros((inference_batch_size, self.encoder_units))]
    enc_out, enc_h, enc_c = self.encoder(inputs, enc_start_state)
    
    dec_h = enc_h
    dec_c = enc_c
    
    start_tokens = tf.fill([inference_batch_size], self.tokenizer.word_index['<start>'])
    end_token = self.tokenizer.word_index['<end>']
    
    greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()
    
    # Instantiate BasicDecoder object
    decoder_instance = tfa.seq2seq.BasicDecoder(cell=self.decoder.rnn_cell, sampler=greedy_sampler, 
                                                output_layer=self.decoder.fc)
    
    # Setup Memory in decoder stack
    self.decoder.attention_mechanism.setup_memory(enc_out)
    # set decoder_initial_state
    decoder_initial_state = self.decoder.build_initial_state(inference_batch_size, [dec_h, dec_c], 
                                                             tf.float32)
    ### Since the BasicDecoder wraps around Decoder's rnn cell only, you have to ensure that the inputs to BasicDecoder 
    ### decoding step is output of embedding layer. tfa.seq2seq.GreedyEmbeddingSampler() takes care of this. 
    ### You only need to get the weights of embedding layer, which can be done by decoder.embedding.variables[0] and pass this callabble to BasicDecoder's call() function
    decoder_embedding_matrix = self.decoder.embedding.variables[0]
    outputs, _, _ = decoder_instance(decoder_embedding_matrix, start_tokens = start_tokens, 
                                     end_token= end_token, initial_state=decoder_initial_state)
    
    return outputs.sample_id.numpy()


  def beam_evaluate_question(self, question, beam_width):
    question = self.dataset_creator.preprocess_sentence(question)

    inputs= []
    for i in question.split(' '):
        try:
          inputs.append(self.tokenizer.word_index[i])
        except:
          inputs.append(0)
    inputs = pad_sequences([inputs], maxlen=self.max_length_input, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    inference_batch_size = inputs.shape[0]

    enc_start_state = [tf.zeros((inference_batch_size, self.encoder_units)),
                        tf.zeros((inference_batch_size, self.encoder_units))]
    enc_out, enc_h, enc_c = self.encoder(inputs, enc_start_state)

    dec_h = enc_h
    dec_c = enc_c

    start_tokens = tf.fill([inference_batch_size], self.tokenizer.word_index['<start>'])
    end_token = self.tokenizer.word_index['<end>']

    # From official documentation
    # NOTE If you are using the BeamSearchDecoder with a cell wrapped in AttentionWrapper, then you must ensure that:
    # The encoder output has been tiled to beam_width via tfa.seq2seq.tile_batch (NOT tf.tile).
    # The batch_size argument passed to the get_initial_state method of this wrapper is equal to true_batch_size * beam_width.
    # The initial state created with get_initial_state above contains a cell_state value containing properly tiled final state from the encoder.
    enc_out = tfa.seq2seq.tile_batch(enc_out, multiplier=beam_width)
    self.decoder.attention_mechanism.setup_memory(enc_out)
    # print("beam_with * [batch_size, max_length_input, rnn_units] :  3 * [1, 16, 1024]] :", enc_out.shape)

    # set decoder_inital_state which is an AttentionWrapperState considering beam_width
    hidden_state = tfa.seq2seq.tile_batch([dec_h, dec_c], multiplier=beam_width)
    decoder_initial_state = self.decoder.rnn_cell.get_initial_state(batch_size=beam_width*inference_batch_size,
                                                                    dtype=tf.float32)
    decoder_initial_state = decoder_initial_state.clone(cell_state=hidden_state)

    # Instantiate BeamSearchDecoder
    decoder_instance = tfa.seq2seq.BeamSearchDecoder(self.decoder.rnn_cell,beam_width=beam_width, 
                                                     output_layer=self.decoder.fc)
    decoder_embedding_matrix = self.decoder.embedding.variables[0]

    # The BeamSearchDecoder object's call() function takes care of everything.
    outputs, final_state, sequence_lengths = decoder_instance(decoder_embedding_matrix, 
                                                              start_tokens=start_tokens, 
                                                              end_token=end_token, 
                                                              initial_state=decoder_initial_state)
    # outputs is tfa.seq2seq.FinalBeamSearchDecoderOutput object. 
    # The final beam predictions are stored in outputs.predicted_id
    # outputs.beam_search_decoder_output is a tfa.seq2seq.BeamSearchDecoderOutput object which keep tracks of beam_scores and parent_ids while performing a beam decoding step
    # final_state = tfa.seq2seq.BeamSearchDecoderState object.
    # Sequence Length = [inference_batch_size, beam_width] details the maximum length of the beams that are generated
    # outputs.predicted_id.shape = (inference_batch_size, time_step_outputs, beam_width)
    # outputs.beam_search_decoder_output.scores.shape = (inference_batch_size, time_step_outputs, beam_width)
    
    # Convert the shape of outputs and beam_scores to (inference_batch_size, beam_width, time_step_outputs)
    final_outputs = tf.transpose(outputs.predicted_ids, perm=(0,2,1))
    beam_scores = tf.transpose(outputs.beam_search_decoder_output.scores, perm=(0,2,1))
    
    return final_outputs.numpy(), beam_scores.numpy()


  def respond(self, question):
    print('Question: %s' % (question))
    result = self.evaluate_question(question)
    response = self.tokenizer.sequences_to_texts(result)
    print('Response: {}\n'.format(response))
    
    
  def beam_respond(self, question, beam_width):
    print('Question: %s' % (question))
    predictions, beam_scores = self.beam_evaluate_question(question, beam_width)
    predictions = predictions[0]
    beam_scores = beam_scores[0]
    predictions = self.tokenizer.sequences_to_texts(predictions)
    beam_scores = [scores.sum() for scores in beam_scores]
    for i in range(len(predictions)):
      print('Prediction {}: {}  {}'.format(i+1, predictions[i][:predictions[i].index('<end>')]
                                           , beam_scores[i]))
    print()


  def test(self, beam=False, beam_width=1):
    TP, FP, FN, f1, top_n_TP, top_n_FP, top_n_FN, top_n_f1 = 0, 0, 0, 0, 0, 0, 0, 0
    correct, top_n_correct = 0, 0
    for (batchno, (inp, targ)) in enumerate(self.test_dataset):
        questions = self.tokenizer.sequences_to_texts(np.array(inp))
        answers = self.tokenizer.sequences_to_texts(np.array(targ))
        for j in range(len(questions)):
          question = re.sub("(<OOV>)+|(<start>)+|(<end>)+", "", questions[j]).strip()
          answer = re.sub("(<OOV>)+|(<start>)+|(<end>)+", "", answers[j]).strip()
          answer_words = answer.split()
          response_words = []
          top_n_response_words = []
          counted_words = []
          if beam:
            predictions, beam_scores = self.beam_evaluate_question(question, beam_width=5)
            predictions = predictions[0]
            beam_scores = beam_scores[0]
            predictions = self.tokenizer.sequences_to_texts(predictions)
            beam_scores = [a.sum() for a in beam_scores]
            for i in range(len(predictions)):
              predictions[i] = predictions[i][:predictions[i].index('<end>')]
              for word in predictions[i].split():
                if word not in counted_words:
                  if i == 0:
                    response_words.append(word)
                  top_n_response_words.append(word)
                  counted_words.append(word)
              if answer == predictions[i].strip():
                top_n_correct += 1
            if answer == predictions[0].strip():
              correct += 1
            for word in answer_words:
              if word in top_n_response_words:
                top_n_TP += 1
              else:
                top_n_FN += 1
            for word in top_n_response_words:
              if word not in answer_words:
                top_n_FP += 1
            top_n_precision = top_n_TP/(top_n_TP+top_n_FP)
            top_n_recall = top_n_TP/(top_n_TP+top_n_FN)
            if top_n_precision+top_n_recall != 0:
              top_n_f1 = 2*top_n_precision*top_n_recall/(top_n_precision+top_n_recall)
          else:
            response = self.evaluate_question(question)
            response = self.tokenizer.sequences_to_texts(response)[0]
            response = re.sub("(<OOV>)+|(<start>)+|(<end>)+", "", response).strip()
            if response == answer:
              correct += 1    
            response_words = response.split()          
          for word in answer_words:
            if word in response_words:
              TP += 1
            else:
              FN += 1
          for word in response_words:
            if word not in answer_words:
              FP += 1
          # print("answer_words:", answer_words)
          # print("response_words:", response_words)
          # print("TP, FP, FN:", TP, FP, FN)
          # print("precision, recall:", precision, recall)
          # exit()
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        if precision+recall != 0:
          f1 = 2*precision*recall/(precision+recall)
        print('\nQuestion: %s' % (question))
        print('Answer: {}'.format(answer))          
        if beam:
          for i in range(len(predictions)):
            print('Prediction {}: {}  {}'.format(i+1, predictions[i],
                                                beam_scores[i]))
          accuracy = correct/((batchno*self.BATCH_SIZE)+j+1)
          top_n_accuracy = top_n_correct/((batchno*self.BATCH_SIZE)+j+1)
          print(f"Accuracy: {accuracy} ({correct}/{(batchno*self.BATCH_SIZE)+j+1} correct)")
          print(f"Top {beam_width} Accuracy: {top_n_accuracy} ({top_n_correct}/"
                f"{(batchno*self.BATCH_SIZE)+j+1} correct)")
          print(f"F1 Score: {f1} Precision: {precision} Recall: {recall}")
          print(f"Top {beam_width} Recall: {top_n_recall}")
        else:
          print('Response: {}'.format(response))
          print(f"Accuracy: {accuracy} ({correct}/{(batchno*self.BATCH_SIZE)+j+1} correct)")
          print(f"F1 Score: {f1} Precision: {precision} Recall: {recall}")