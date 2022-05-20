import MLP
import numpy as np
import util
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.sequence import pad_sequences as pad
import os
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from subprocess import call
from Seq2SeqRNN import Seq2SeqRNN 
import time

def main():
    run_Seq2SeqRNN = 1
    run_MLP = 0
    run_LogReg = 0
    
    BUFFER_SIZE = 1000
    BATCH_SIZE = 20
    
    if run_Seq2SeqRNN:
        training = 0
        respond = 0
        EPOCHS = 2
        pretrained = 1
        beam = 1
        version = "bahdanau_traindev"
        attention_type = "bahdanau"
        train_path = "RNN_data/RNN_train_dev.txt"
        dev_path = "RNN_data/RNN_dev.txt"
        test_path = "RNN_data/RNN_test.txt"  
        num_examples = -1
        embedding_dim = 100
        encoder_units = 256
        decoder_units = 256
        
        rnn = Seq2SeqRNN(version, num_examples, BUFFER_SIZE, BATCH_SIZE, attention_type, train_path, 
                dev_path, test_path, embedding_dim, encoder_units, decoder_units)

        print("\nModel version:", version)
        if training:
            print("\nTraining")
            print(f"Train dataset: {train_path}")
            print(f"Using {attention_type} attention")
            print(f"Runninng {EPOCHS} passes")
            if num_examples == -1:
                steps_per_epoch = rnn.train_dataset.cardinality().numpy()
            else:
                steps_per_epoch = num_examples
            if pretrained:
                print(f"Loading pre-trained checkpoint {tf.train.latest_checkpoint(rnn.checkpoint_dir)}")
                # restoring the latest checkpoint in checkpoint_dir
                rnn.checkpoint.restore(tf.train.latest_checkpoint(rnn.checkpoint_dir)).expect_partial()
            rnn.train(EPOCHS, steps_per_epoch)
        else:
            # restoring the latest checkpoint in checkpoint_dir
            rnn.checkpoint.restore(tf.train.latest_checkpoint(rnn.checkpoint_dir)).expect_partial()
            if beam:
                if respond:
                    rnn.beam_respond("first number", beam_width=5)
                    rnn.beam_respond("famous author", beam_width=5)
                    rnn.beam_respond("big state", beam_width=5)
                    rnn.beam_respond("reactive element", beam_width=5)
                    rnn.beam_respond("One work set in this country describes an affair between Alex Thomas "
                                    "and Iris Chase. ||| In another work set in this country, a young orphan "
                                    "girl lives with Matthew and Marilla Cuthbert. ||| Author Lucy Maud "
                                    "Montgomery hails from this country, which provides the setting for The "
                                    "Blind Assassin. ||| Writers from this country include the authors of Dance "
                                    "of the Happy Shades, Life of Pi, The English Patient, and The Handmaid's Tale. "
                                    "||| For 10 points, name this country, the home of Alice Munro, Yann Martel, "
                                    "Michael Ondaatje and Margaret Atwood.", beam_width=5)
                else:
                    rnn.test(beam=True, beam_width=5)
            else:
                if respond:
                    rnn.respond("the")
                    # rnn.respond("With the assistence of his chief minister, the Duc de Sully, he lowered taxes on peasantry, promoted economic recovery, and instituted a tax on the Paulette. ||| Victor at Ivry and Arquet, he was excluded from succession by the Treaty of Nemours, but won a great victory at Coutras. ||| His excommunication was lifted by Clement VIII, but that pope later claimed to be crucified when this monarch promulgated the Edict of Nantes. ||| For 10 points, name this French king, the first Bourbon who admitted that ""Paris is worth a mass"" when he converted following the War of the Three Henrys.")
                        # checkpoint.restore(f"./{checkpoint_dir}/ckpt-3")
                else:
                    rnn.test()
            
        

    if run_MLP:
        X_train, Y_train, X_dev, Y_dev, X_test, Y_test = util.read_questions()
        X_wiki, Y_wiki = util.read_wiki()
        # X_test_wtop = util.object_load_JSON("X/X_test_wtop2.json")
        X_test_wtopb = util.object_load("X/X_test_wtopb4.json")
        dataset = util.object_load("dataset.json")
        
        if os.path.exists("tokenizer.json"):
            tokenizer = util.object_load("tokenizer.json", int_keys=True)
        else:
            tokenizer = keras.preprocessing.text.Tokenizer(filters='', oov_token='<OOV>', split=' ')
            tokenizer.fit_on_texts(dataset)
            util.object_save(tokenizer, "tokenizer.json")
        
        X_test_int = pad(np.array(tokenizer.texts_to_sequences(X_test), dtype=object), maxlen=266, padding='post')
        X_test_wtopb_int = pad(np.array(tokenizer.texts_to_sequences(X_test_wtopb), dtype=object), maxlen=150, padding='post')

        # X_wiki_int = pad(np.array(tokenizer.texts_to_sequences(X_wiki), dtype=object), maxlen=29853, padding='post')
        # X_wiki_top_int = pad(np.array(tokenizer.texts_to_sequences(X_wiki_top), dtype=object), maxlen=266, padding='post')

        intlabels = util.int_labels(Y_train + Y_dev + Y_test + Y_wiki)
        rev_intlabels = {v:k for k, v in intlabels.items()}
        Y_test_int = np.array([intlabels[answer.lower()] for answer in Y_test])
        # Y_wiki_int = np.array([intlabels[answer.lower()] for answer in Y_wiki])
        
        if os.path.exists(f"MLP_twtop3"):
            print(f"Loading MLP")
            mlp = tf.keras.models.load_model(f"MLP_twtop3")
        else:
            print(f"Building MLP")
            num_train_examples = -1
            embedding_matrix = util.embedding_matrix(tokenizer, embedding_dim, extension="_MLP_top")
            num_classes = len(intlabels)
            MLP1 = MLP.MLP(embedding_matrix, num_classes)

            X_train_wtop = util.object_load("X/X_train_wtop2.json")
            X_dev_wtop = util.object_load("X/X_dev_wtop2.json")
            
            X_train_int = pad(np.array(tokenizer.texts_to_sequences(X_train), dtype=object), maxlen=266, padding='post')
            X_dev_int = pad(np.array(tokenizer.texts_to_sequences(X_dev), dtype=object), maxlen=266, padding='post')
            
            X_train_wtop_int = pad(np.array(tokenizer.texts_to_sequences(X_train_wtop), dtype=object), maxlen=150, padding='post')
            X_dev_wtop_int = pad(np.array(tokenizer.texts_to_sequences(X_dev_wtop), dtype=object), maxlen=150, padding='post')
            
            Y_train_int = np.array([intlabels[answer.lower()] for answer in Y_train])
            Y_dev_int = np.array([intlabels[answer.lower()] for answer in Y_dev])
            
            # train_dataset_X = tf.data.Dataset.from_tensor_slices((tf.constant(X_train_int), tf.constant(X_train_w_int)))
            train_dataset_X = tf.data.Dataset.from_tensor_slices((tf.constant(X_train_int[:num_train_examples]), tf.constant(X_train_wtop_int[:num_train_examples])))
            train_dataset_Y = tf.data.Dataset.from_tensor_slices(tf.constant(Y_train_int[:num_train_examples]))
            train_dataset = tf.data.Dataset.zip((train_dataset_X, train_dataset_Y))
            train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
            
            # val_dataset_X = tf.data.Dataset.from_tensor_slices((tf.constant(X_dev_int), tf.constant(X_dev_w_int)))
            val_dataset_X = tf.data.Dataset.from_tensor_slices((tf.constant(X_dev_int[:num_train_examples]), tf.constant(X_dev_wtop_int[:num_train_examples])))
            val_dataset_Y = tf.data.Dataset.from_tensor_slices(tf.constant(Y_dev_int[:num_train_examples]))
            val_dataset = tf.data.Dataset.zip((val_dataset_X, val_dataset_Y))
            val_dataset = val_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
            
            mlp = MLP1.train(train_dataset, val_dataset, embedding_dim, intlabels)
            
            print("Saving MLP")
            mlp.save(f"MLP_twtop4")

        # mlp.summary()
        num_test_examples = -1
        top_n = 5
                
        title = "X_test_pos5"
        X_test = util.object_load(f"X/{title}.json")
        X_test_int = pad(np.array(tokenizer.texts_to_sequences(X_test), dtype=object), maxlen=266, padding='post')

        MLP.test(mlp, X_test, X_test_int, X_test_wtopb_int, Y_test, num_test_examples, rev_intlabels, top_n)
        
      
        
    if run_LogReg:
        call("run.sh")
        
        

if __name__ == '__main__':
    main()

