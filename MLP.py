import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import util
import numpy as np

class MLP(keras.Model):    
    def __init__(self, embedding_matrix, num_classes, **kwargs):
        super().__init__(**kwargs)
        tokenizer = util.object_load("tokenizer.json", int_keys=True)
        embedding_dim = embedding_matrix.shape[1]
        embedding_matrix = util.embedding_matrix(tokenizer, embedding_dim, extension="_MLP_top")
        self.embedding = tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=embedding_dim, 
                                                   embeddings_initializer=keras.initializers.Constant(embedding_matrix), 
                                                   trainable=True, mask_zero=True)
        # self.dropout1 = tf.keras.layers.Dropout(0.05, name='dropout1')
        self.flatten = tf.keras.layers.Flatten()
        # self.dense1 = tf.keras.layers.Dense(128, activation='elu', name='dense1')
        # self.dropout2 = tf.keras.layers.Dropout(0.2, name='dropout2')
        self.dense2 = tf.keras.layers.Dense(64, activation='elu', kernel_initializer=tf.keras.initializers.HeNormal(), 
                                            name='dense2')
        self.soft_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='soft_output')


    def call(self, inputs):
        inputA = inputs[0]
        inputB = inputs[1]
        inputAB = tf.keras.layers.Concatenate()([inputA, inputB])
        # dropout1 = self.dropout1(inputAB)
        embedding = self.embedding(inputAB)
        flatten = self.flatten(embedding)
        # dense2 = self.dense2(flatten)
        soft_output = self.soft_output(flatten)
        return soft_output
    
    
    def model(self):
        inputA = keras.layers.Input(shape=(266))
        # inputB = keras.layers.Input(shape=(29853))
        inputB = keras.layers.Input(shape=(150))
        return keras.Model(inputs=[inputA, inputB], outputs=self.call([inputA, inputB]))


    def train(self, train_dataset, val_dataset, n_epochs):
        mlp = self.model()
        mlp.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

        n_epochs = 3
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(f"MLP_i",save_best_only=True)
        history = mlp.fit(train_dataset, epochs=n_epochs, verbose=2, validation_data=(val_dataset), callbacks=[checkpoint_cb])
        
        # plot_model(mlp, to_file=f'MLP.png', show_layer_names=True, show_shapes=True)
        self.plot_metrics(history, n_epochs)
        
        return mlp
        
        
    def plot_metrics(self, history, n_epochs):
        metrics = {'loss': 'Training Loss', 'accuracy': 'Training Accuracy', 'val_loss': 'Validation Loss', 
                   'val_accuracy': 'Validation Accuracy'}
        for metric, metric_name in metrics.items():
            x = plt.figure(1)
            plt.plot(range(1, n_epochs+1), history.history[metric], label=f'MLP_{n_epochs}')
            plt.title(f'{metric_name}')
            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plt.legend()
            plt.savefig(f'MLP_{metric_name}.svg', format='svg')
            plt.show()
            plt.clf()
            
     
            
def test(mlp, X, X_int, Xw_int, Y, num_test_examples, rev_intlabels, top_n):
    if num_test_examples == -1:
        num_test_examples = len(X_int)
    pred_probs = mlp.predict([X_int[:num_test_examples], Xw_int[:num_test_examples]])
    # pred_prob = np.max(pred_probs, axis=1)
    pred_classes = np.argmax(pred_probs, axis=1)
    pred_n_classes = np.argsort(pred_probs, axis=1)[:,-top_n:]
    pred_answers = [rev_intlabels[pred_class] for pred_class in list(pred_classes)]
    pred_n_answers = []
    for pred_n_classes_i in pred_n_classes:
        pred_n_answers_i = [rev_intlabels[pred_class] for pred_class in list(pred_n_classes_i)]
        pred_n_answers.append(pred_n_answers_i)
    questions = X[:num_test_examples]
    answers = [ans.lower() for ans in Y[:num_test_examples]]
    correct = 0
    top_n_correct = 0
    for i in range(0, num_test_examples):
        if i % 50 == 0:
            print(f"\nQuestion: {questions[i]}")
            print(f"Guess: {pred_answers[i]}")
            print(f"Guesses: {pred_n_answers[i]}")
            print(f"Answer: {answers[i]}")
        if answers[i] in pred_n_answers[i]:
            top_n_correct += 1
            if answers[i] == pred_n_answers[i][-1]:
                correct += 1
        accuracy = correct/(i+1)
        top_n_accuracy = top_n_correct/(i+1)
        print(f"Accuracy: {accuracy} ({correct}/{i+1} correct)")
        print(f"Top {top_n} accuracy: {top_n_accuracy} ({top_n_correct}/{i+1} correct)\n")
        