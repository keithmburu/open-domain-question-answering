import csv
import numpy as np
import jsonpickle as jspkl
import bz2
import os
import random
import regex as re
from nltk.util import bigrams

def read_questions():
    if os.path.exists("X/X_train.json") and os.path.exists("Y/Y_train.json"):
        X_train = object_load("X/X_train.json")
        Y_train = object_load("Y/Y_train.json")
        X_dev = object_load("X/X_dev.json")
        Y_dev = object_load("Y/Y_dev.json")
        X_test = object_load("X/X_test.json")
        Y_test = object_load("Y/Y_test.json")
    else:    
        print("Parsing questions")
        with open("questions.csv", 'r') as file:
            reader = csv.reader(file)
            # examples = []
            X_train, Y_train, X_dev, Y_dev, X_test, Y_test = ([],[],[],[],[],[])
            for i, line in enumerate(reader):
                if i == 0: continue
                id, form, answer, category, question = line[:5]
                x = question
                y = answer
                # example = Example(id, category, question, answer)
                # examples.append(example)
                if form == "train":
                    X_train.append(x)
                    Y_train.append(y)
                if form == "dev":
                    X_dev.append(x)
                    Y_dev.append(y)
                elif form == "test":
                    X_test.append(x)
                    Y_test.append(y)
        object_save(X_train, "X/X_train.json")
        object_save(Y_train, "Y/Y_train.json")
        object_save(X_dev, "X/X_dev.json")
        object_save(Y_dev, "Y/Y_dev.json")
        object_save(X_test, "X/X_test.json")
        object_save(Y_test, "Y/Y_test.json")
    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test


def read_wiki():
    if os.path.exists("X/X_wiki.json") and os.path.exists("Y/Y_wiki.json"):
        X_wiki = object_load("X/X_wiki.json")
        Y_wiki = object_load("Y/Y_wiki.json")
    else:
        print("Parsing wiki")
        wikipath = "wiki"
        X_wiki, Y_wiki = ([],[])
        for filepath in os.listdir("wiki"):
            filepath = os.path.join(wikipath, filepath)
            with open(filepath, 'r') as file:
                X_wiki.append(file.read())
                Y_wiki.append(filepath[5:-4])
        object_save(X_wiki, "X/X_wiki.json")
        object_save(Y_wiki, "Y/Y_wiki.json")
    return X_wiki, Y_wiki


def object_save(object, filepath, int_keys=False):
    print(f"Saving {filepath}")
    if "X_wiki_tfidf" in filepath or "embedding" in filepath or "glove" in filepath:  
        with bz2.open(f"{filepath}.zip", 'wt') as file:
            file.write(jspkl.encode(object, keys=int_keys))
    else:
        with open(f"{filepath}", 'w') as file:
            file.write(jspkl.encode(object, keys=int_keys))


def object_load(filepath, int_keys=False):
    print(f"Loading {filepath}")
    if "X_wiki_tfidf" in filepath or "embedding" in filepath or "glove" in filepath:  
        with bz2.open(f"{filepath}.zip", 'rt') as file:
            object = jspkl.decode(file.read(), keys=int_keys)
    else:
        with open(f"{filepath}", 'r') as file:
            object = jspkl.decode(file.read(), keys=int_keys)
    return object


def examples_to_VWfiles(title, X, Y):
    print(f"Converting {title} examples to VW files")
    if not os.path.exists(title):
        os.mkdir(title)
    for x, y in zip(X, Y):
        try:
            with open(f'{title}/{y}.txt', 'w') as file:
                    file.write(x) 
        except:
            y = y.replace("/", " ")
            with open(f'{title}/{y}.txt', 'w') as file:
                file.write(x)


def examples_to_VWfile(title, X, Y):
    print(f"Converting {title} examples to VW file")
    text = ""
    for x, y in zip(X, Y):
        stripped = re.sub("[^ 'a-zA-Z0-9]+", " ", re.sub("(<br \/>)", " ", x))
        line = f"{y} | {stripped}".lower()
        text += line + '\n'
    with open(f'VW/{title}.txt', 'w') as file:
        file.write(text) 
        
        
def examples_to_VWfile_wtop(title, X, Xw, Y):
    print(f"Converting {title} examples to VW file")
    text = ""
    for x, xw, y in zip(X, Xw, Y):
        stripped = re.sub("[^ 'a-zA-Z0-9]+", " ", re.sub("(<br \/>)", " ", x))
        line = f"{y} |{title} {stripped} |wiki_top {xw}".lower()
        text += line + '\n'
        # print(line+"\n")
    with open(f'VW/{title}.txt', 'w') as file:
        file.write(text) 
        
        
def examples_to_RNNfile(title, X, Y):
    print(f"Writing {title} examples to RNN file")
    text = ""
    with open(f'RNN data/{title}.txt', 'w') as file:
      for x, y in zip(X, Y):
        x = re.sub('\n', ' ', re.sub("\t", " ", x))
        line = f"{x}\t{y}"
        text += line + '\n'
      file.write(text) 


def int_labels(answers):
    int_labels = {}
    num = 1
    covered = []
    for answer in answers:
        ans = answer.lower()
        if ans not in covered:
            int_labels[ans] = num
            num += 1
            covered.append(ans)
    return int_labels


def assign_VW_int_labels(int_labels, filepath):
    print(f"Adding integer labels to {filepath}")
    with open(f"./{filepath}", "r") as infile:
        with open(f"./int_{filepath}", "w") as outfile:
            lines = infile.readlines()
            for i in range(len(lines)):
                full = lines[i].replace('\n', '')
                words = full.split("|", maxsplit=1)
                if len(words) >= 2:
                    try:
                        tag = words[0][:-1]
                        int_label = int_labels[tag]
                        tag = tag.replace(" ", "_")
                        label = f"{int_label} '{tag}"
                        full = "|".join([label, words[1]])
                        if (i+1) < len(lines) and len(lines[i+1].split("|", maxsplit=1)) >= 2:
                            outfile.write(full + '\n')
                        else:
                            outfile.write(full)
                    except:
                        print(full)
                        exit()
                else:
                    if (i+1) < len(lines) and len(lines[i+1].split("|", maxsplit=1)) >= 2:
                        outfile.write(full + '\n')
                    else:
                        outfile.write(full) 


def vocab_docfreqs(titles_partitions):
    if os.path.exists("vocab.json") and os.path.exists("docfreqs.json"):
        vocab = object_load("vocab.json")
        docfreqs = object_load("docfreqs.json")
    else:   
        vocab = {}
        docfreqs = {}
        for title, partition in titles_partitions:
            print(f"Counting {title} document frequencies")
            docfreqs[title] = []
            for question in partition:
                question = re.sub('[^A-Za-z0-9 ]+', ' ', question).lower()
                words = question.split()
                freqs = {}
                for word in words:
                    if word in freqs:
                        freqs[word] += 1
                    else:
                        freqs[word] = 1
                        if word in vocab:
                            vocab[word] += 1
                        else:
                            vocab[word] = 1
                # freqs = dict(sorted(freqs.items(), key=lambda x:x[1], reverse=1))
                docfreqs[title].append(freqs)
        object_save(vocab, "vocab.json")
        object_save(docfreqs, "docfreqs.json")
    return vocab, docfreqs


def tfidf(title, docfreqs, vocab):
    if os.path.exists(f"X/{title}_tfidf.json"):
        X_tfidf = object_load(f"{title}_tfidf.json")
    else:
        print(f"Computing {title}_tfidf")
        X_freqs = docfreqs[title]
        print(f"{len(X_freqs)} questions")
        n_docs = 0
        for X_freqs_i in docfreqs.values():
            n_docs += len(X_freqs_i)
        X_tfidf = []
        df = np.array(list(vocab.values()))
        vocab = list(vocab.keys())
        for i in range(len(X_freqs)):
            if i % 50 == 0:
                print(f"{i}", end=" ", flush=True)
            keys = list(X_freqs[i].keys())
            values = list(X_freqs[i].values())
            freqs = np.zeros(len(vocab))
            idf = np.zeros(len(vocab))
            for word in keys:
                freqs[vocab.index(word)] = int(values[keys.index(word)])
                idf[vocab.index(word)] = np.log(n_docs/df[vocab.index(word)])
            total = sum(freqs)
            if total != 0:
                tf = freqs/total
            else:
                tf = freqs
            tfidf = tf*idf
            X_tfidf.append(tfidf)
        random.shuffle(X_tfidf)
        object_save(f"{title}_tfidf.json")
    return X_tfidf


def top_words(title, X_tfidf, thresh, vocab):
    print(f"Computing {title}_top")
    print(f"{len(X_tfidf)} questions")
    X_top = []
    for i in range(len(X_tfidf)):
        if i % 50 == 0:
            print(f"{i}", end=" ", flush=True)
        sorted_tfidfs = sorted(X_tfidf[i], reverse=1)
        top_tfidfs = []
        j = 0
        count = 0
        while count < thresh:
            if sorted_tfidfs[j] not in top_tfidfs:
                top_tfidfs.append(sorted_tfidfs[j])
                count += 1
            elif sorted_tfidfs[j] == 0:
                top_tfidfs.append(0)
                count += 1
            j += 1
        topwords = np.empty(thresh, dtype=object)
        index_lookup = {tfidf: i[0] for i, tfidf in np.ndenumerate(X_tfidf[i])}
        for j, tfidf in enumerate(top_tfidfs):
            word = ""
            if tfidf != 0:
                word_index = index_lookup[tfidf]
                word = list(vocab.keys())[word_index] 
            topwords[j] = word 
        X_top.append(topwords)
    return X_top


def bin_unigrams(filepath):
    print(f"Adding binary unigram features to {filepath}")
    with open(f"./{filepath}", "r") as infile:
        with open(f"./bin_{filepath}", "w") as outfile:
            lines = infile.readlines()
            for i in range(len(lines)):
                word_dict = {}
                tokens = lines[i].split()[:2]
                words = lines[i].split()[2:]
                for word in words:
                    if word not in word_dict:
                        tokens.append(word)
                        word_dict[word] = 1
                lines[i] = " ".join(tokens)
                outfile.write(lines[i] + '\n') 


def wc_unigrams(filepath):
    print(f"Adding word count unigram features to {filepath}")
    with open(f"{filepath}", "r") as infile:
        with open(f"./wc_{filepath}", "w") as outfile:
            lines = infile.readlines()
            for i in range(len(lines)):
                word_dict = {}
                tokens = lines[i].split()[:2]
                words = lines[i].split()[2:]
                for word in words:
                    if word in word_dict:
                        word_dict[word] += 1
                    else:
                        word_dict[word] = 1
                for word in word_dict:
                    tokens.append(f"{word}:{float(word_dict[word])}")
                lines[i] = " ".join(tokens)
                outfile.write(lines[i] + '\n') 


def bigrams(filepath):
    print(f"Adding bigram features to {filepath}")
    with open(f"./{filepath}", "r") as infile:
        with open(f"./big_{filepath}", "w") as outfile:
            lines = infile.readlines()
            for i in range(len(lines)):
                tokens = lines[i].split()
                words = lines[i].split()[2:]
                bigram_tuples = list(bigrams(words))
                bigram_strings = []
                for tuple in bigram_tuples:
                    bigram_strings.append(tuple[0] + tuple[1])
                tokens += bigram_strings
                lines[i] = " ".join(tokens)
                outfile.write(lines[i] + '\n')


def embedding_matrix(tokenizer, embedding_dim, extension=""):
    embedding_matrix = []
    if os.path.exists(f"embeddings/embedding_matrix_{embedding_dim}{extension}.json.zip"):
        embedding_matrix = object_load(f"embeddings/embedding_matrix_{embedding_dim}{extension}.json")
    else:
        print("Setting up pre-trained embedding")
        path_to_glove_file = f"embeddings/glove.6B.{embedding_dim}d.txt"
        embeddings_index = {}
        with open(path_to_glove_file, encoding="utf8") as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs

        print("Found %s word vectors." % len(embeddings_index))

        num_tokens = len(tokenizer.word_index) + 1
        hits = 0
        misses = 0
        # Prepare embedding matrix
        embedding_matrix = np.zeros((num_tokens, embedding_dim))
        for word, i in tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                misses += 1
        print("Converted %d words (%d misses)" % (hits, misses))
        object_save(embedding_matrix, f"embeddings/embedding_matrix_{embedding_dim}{extension}.json")
    return embedding_matrix


def corresponding_wiki_articles(new_title, X, Y, X_wiki, Y_wiki):
    if os.path.exists("X/{new_title}.json"):
        X_w = object_load("{new_title}.json")
    else:
        X_w = []
        Y_X_wiki_dict = {y:x for x, y in zip(X_wiki, Y_wiki)}
        for i in range(len(X)):
            try:
                X_w.append(Y_X_wiki_dict[Y[i]])
            except:
                X_w.append("")
        object_save(X_w, "{new_title}.json")
    return X_w
    

def guess_articles(X_test, X_wiki_top, X_wiki_top_tfidfs, actual):
    X_test_wtopb = []
    correct = 0
    accuracy = 0
    for i in range(len(X_test)):
        if i % 10 == 0:
            print(i, flush=True)
        arg_max = ""
        max = 0
        test = [t.lower() for t in X_test[i].split()]
        for j in range(len(X_wiki_top)):
            count = 0
            value = 0
            top = X_wiki_top[j].split()
            for k, word in enumerate(top):
                if word in test:
                    count += 1
                    value += X_wiki_top_tfidfs[j][k]
            if count+(75*value) > max:
                max = count+(75*value)
                # print(max)
                arg_max = X_wiki_top[j]
        if arg_max == actual[i]:
            correct += 1
        accuracy = correct/(i+1)
        # print(X_test[i], "\n", arg_max, "\n", true[i], '\n', Y_test[i], '\n', max, "\n\n")
        print(accuracy)
        # exit()
        X_test_wtopb.append(arg_max)
    object_save(X_test_wtopb, "X_test_wtopb5.json")
    return X_test_wtopb
    
    
def guess_accuracy(X_test_w, actual):
    count = 0
    for x1, x2 in zip(X_test_w, actual):
        if x1 == x2:
            count += 1
        # else:
        #     # print(x1)
        #     # print(x2, "\n")
    print(count)
    
    
def partial_questions(title, X, position):
    if os.path.exists(f"X/{title}_pos{position}.json"):
        X_partial = object_load(f"X/{title}_pos{position}.json")
    else:
        X_partial = []
        for question in X:
            parts = question.split('|||')[:position]
            X_partial.append(''.join(parts))
            # print(question)
            # print(" ".join(parts))
            # exit()
        object_save(X_partial, f"X/{title}_pos{position}.json")
    return X_partial



def main():
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = read_questions()
    X_wiki, Y_wiki = read_wiki()

    # X_train_w = corresponding_wiki_articles("X_train_w", X_train, Y_train, X_wiki, Y_wiki)
    # X_dev_w = corresponding_wiki_articles("X_dev_w", X_dev, Y_dev, X_wiki, Y_wiki)
    # X_test_w = corresponding_wiki_articles("X_test_w", X_test, Y_test, X_wiki, Y_wiki)
    
    # X_wiki_top = object_load("X/X_wiki_top2.json")
    # X_train_wtop = corresponding_wiki_articles("X_train_wtop", X_train, Y_train, X_wiki_top, Y_wiki)
    # X_dev_wtop = corresponding_wiki_articles("X_dev_wtop", X_dev, Y_dev, X_wiki_top, Y_wiki)
    # X_test_wtop = corresponding_wiki_articles("X_test_wtop", X_test, Y_test, X_wiki_top, Y_wiki)

    # actual = object_load("X/X_test_wtop2.json")
    # X_wiki_top = object_load("X/X_wiki_top2.json")
    # X_wiki_top_tfidfs = object_load("X/X_wiki_top_tfidfs.json")
    # X_test_w_guesses = guess_articles(X_wiki_top, X_wiki_top_tfidfs, actual)
    # guess_accuracy(X_test_w_guesses, actual)
    
    # dataset = X_train + X_dev + X_test + X_wiki_top + Y_train + Y_dev + Y_test + Y_wiki
    # object_save(dataset, "dataset.json")

    # vocab, docfreqs = vocab_docfreqs([("X_train", X_train), ("X_dev", X_dev), ("X_test", X_test), ("X_wiki", X_wiki)])

    # X_tfidf = []
    # for title in ["X_train", "X_dev", "X_test", "X_wiki"]:
        # for filepath in [f"{title}_tfidf_0_500.json", f"{title}_tfidf_500_1000.json", f"{title}_tfidf_1000_1500.json", 
    #              f"{title}_tfidf_1500_2000.json", f"{title}_tfidf_2000_2500.json"]:
    #         X_tfidf = tfidf(title, docfreqs, vocab)
    
    # X_top = []
    # for title in ["X_train", "X_dev", "X_test", "X_wiki"]:
        # for filepath in [f"{title}_tfidf_0_500.json", f"{title}_tfidf_500_1000.json", f"{title}_tfidf_1000_1500.json", 
    #              f"{title}_tfidf_1500_2000.json", f"{title}_tfidf_2000_2500.json"]:
    #     X_tfidf = object_load(filepath)
    #     X_top.append(top_words(filepath, X_tfidf, 150, vocab))
    #     object_save(X_top, f"X_wiki_top2.json")

    # X_wiki_top = object_load("X/X_wiki_top2.json")
    # X_wiki_top_q = []
    # for partition in X_wiki_top:
    #     for example in partition:
    #         X_wiki_top_q.append(" ".join(example))
    # object_save(X_wiki_top_q, "X_wiki_top2b.json")

    # X_wtop2_train = []
    # X_wtop2_dev = []
    # X_wtopb4_test = []
    # X_train_wtop2 = object_load("X/X_train_wtop2.json")
    # X_dev_wtop2 = object_load("X/X_dev_wtop2.json")
    # X_test_wtopb = object_load("X/X_test_wtopb4.json")
    # for i in range(len(X_train)):
    #     X_wtop2_train.append(f"{X_train_wtop2[i]} {X_train[i]}")
    # object_save(X_wtop2_train, "X_wtop2_train.json")
    # for i in range(len(X_dev)):
    #     X_wtop2_dev.append(f"{X_dev_wtop2[i]} {X_dev[i]}")
    # object_save(X_wtop2_dev, "X_wtop2_dev.json")
    # for i in range(len(X_test)):
    #     X_wtopb4_test.append(f"{X_test_wtopb4[i]} {X_test[i]}")
    # object_save(X_wtopb4_test, "X_wtopb4_test.json")


    # examples_to_VWfile_wtop("train", X_train, X_train_wtop, Y_train)
    # examples_to_VWfile_wtop("dev", X_dev, X_dev_wtop, Y_dev)
    
    # X_test_wtopb = object_load("X/X_test_wtopb4.json")
    # for i in range(5, 6):
    #     X_test = object_load(f"X/X_test_pos{i}.json")
    #     examples_to_VWfile_wtop(f"test_pos{i}", X_test, X_test_wtopb, Y_test)


    # examples_to_VWfile("wiki_top", X_wiki_top2, Y_wiki)

    # call("./parse.sh")
    
    # ###weighting by tfidf
    # X_wiki_top_tfidfs = []
    # X_wiki_top = object_load("X/X_wiki_top2.json")
    # j = 0
    # X_wiki_tfidf = object_load("X/X_wiki_tfidf_0_500.json")
    # for i, xwt in enumerate(X_wiki_top):
    #     if i % 10 == 0:
    #         print(i, end=' ', flush=1)
    #     if i == 500:
    #         X_wiki_tfidf = object_load("X/X_wiki_tfidf_500_1000.json")
    #         j = 500
    #     elif i == 1000:
    #         X_wiki_tfidf = object_load("X/X_wiki_tfidf_1000_1500.json")
    #         j = 1000
    #     elif i == 1500:
    #         X_wiki_tfidf = object_load("X/X_wiki_tfidf_1500_2000.json")
    #         j = 1500
    #     elif i == 2000:
    #         X_wiki_tfidf = object_load("X/X_wiki_tfidf_2000_2500.json")
    #         j = 2000
    #     xwtt = []
    #     for word in xwt.split():
    #         if word == ' ':
    #             xwtt.append(0)
    #         else:
    #             idx = list(vocab.keys()).index(word)
    #             tfidf = X_wiki_tfidf[i-j][idx]
    #             xwtt.append(tfidf)
    #     X_wiki_top_tfidfs.append(xwtt)
    # object_save(X_wiki_top_tfidfs, "X/X_wiki_top_tfidfs.json")
    

    # int_labels_dict = int_labels(Y_train+Y_dev+Y_test+Y_wiki)
    # assign_VW_int_labels(int_labels_dict, "VW/train2.txt") 
    # assign_VW_int_labels(int_labels_dict, "VW/dev2.txt") 
    # assign_VW_int_labels(int_labels_dict, "VW/test3.txt") 
    # assign_VW_int_labels(int_labels_dict, "VW/wiki_top.txt") 
    
    # bin_unigrams("VW/int_train.txt")
    # wc_unigrams("VW/int_train.txt")
    # bigrams("VW/int_train.txt")
    
    # bin_unigrams("VW/int_dev.txt")
    # wc_unigrams("VW/int_dev.txt")
    # bigrams("VW/int_dev.txt")

    # bin_unigrams("VW/int_test.txt")
    # wc_unigrams("VW/int_test.txt")
    # bigrams("VW/int_test.txt")

    # bin_unigrams("VW/int_wiki_top.txt")
    # wc_unigrams("VW/int_wiki_top.txt")
    # bigrams("VW/int_wiki_top.txt")
    
    # examples_to_RNNfile("RNN_data/RNN_dataset", dataset_X, dataset_Y)
    # examples_to_RNNfile("RNN_data/RNN_train", X_train, Y_train)
    # examples_to_RNNfile("RNN_data/RNN_wiki", X_wiki, Y_wiki)
    # examples_to_RNNfile("RNN_data/RNN_train_wiki", X_train_wiki, Y_train_wiki)
    # examples_to_RNNfile("RNN_data/RNN_dev", X_dev, Y_dev)
    # examples_to_RNNfile("RNN_data/RNN_test", X_test, Y_test)
    
    # X_wtop2_train = object_load("X/X_wtop2_train.json")
    # X_wtop2_dev = object_load("X/X_wtop2_dev.json")
    # X_wtopb4_test = object_load("X/X_wtopb4_test.json")
    # examples_to_RNNfile("RNN_train2", X_wtop2_train, Y_train)
    # examples_to_RNNfile("RNN_dev2", X_wtop2_dev, Y_dev)
    # examples_to_RNNfile("RNN_test4", X_wtopb4_test, Y_test)

    # partial_questions("X_test", X_test, position=1)
    # partial_questions("X_test", X_test, position=2)
    # partial_questions("X_test", X_test, position=3)
    # partial_questions("X_test", X_test, position=4)
    # partial_questions("X_test", X_test, position=5)
    # X_test_wtopb = object_load("X/X_test_wtopb4.json")
    # partial_questions("X_test_wtopb", X_test_wtopb, position=1)
    # partial_questions("X_test_wtopb", X_test_wtopb, position=2)
    # partial_questions("X_test_wtopb", X_test_wtopb, position=3)

    # counts = {}
    # avgcount = 0
    # totalcount = 0
    # mincount = 10
    # for example in X_test:
    #     count = len(example.split('|||'))
    #     # if count == 1:
    #     #     print("CHECK", example)
    #     #     continue
    #     totalcount += count
    #     if count in counts:
    #         counts[count] += 1
    #     else:
    #         counts[count] = 1
    #     # if count < mincount:
    #     #     mincount = count
    #     #     print(example)
    #     #     print(mincount)
    # print(counts)
    # print("avgcount:", totalcount/len(X_test))
    
    # X_test_pos1 = object_load("X/X_test_pos1.json")
    # X_test_pos2 = object_load("X/X_test_pos2.json")
    # X_test_pos3 = object_load("X/X_test_pos3.json")
    # X_test_pos4 = object_load("X/X_test_pos4.json")
    # X_test_pos5 = object_load("X/X_test_pos5.json")
    # X_test_wtopb_pos1 = object_load("X/X_test_wtopb_pos1.json")
    # X_test_wtopb_pos2 = object_load("X/X_test_wtopb_pos2.json")
    # X_test_wtopb_pos3 = object_load("X/X_test_wtopb_pos3.json")
    # examples_to_RNNfile("RNN_data/RNN_test_pos1", X_test_pos1, Y_test)
    # examples_to_RNNfile("RNN_data/RNN_test_pos2", X_test_pos2, Y_test)
    # examples_to_RNNfile("RNN_data/RNN_test_pos3", X_test_pos3, Y_test)
    # examples_to_RNNfile("RNN_data/RNN_test_pos4", X_test_pos4, Y_test)
    # examples_to_RNNfile("RNN_data/RNN_test_pos5", X_test_pos5, Y_test)
    # examples_to_RNNfile("RNN_data/RNN_test_wtopb_pos1", X_test_wtopb_pos1, Y_test)
    # examples_to_RNNfile("RNN_data/RNN_test_wtopb_pos2", X_test_wtopb_pos2, Y_test)
    # examples_to_RNNfile("RNN_data/RNN_test_wtopb_pos3", X_test_wtopb_pos3, Y_test)

    # X_test_wtopb = object_load("X/X_test_wtopb4.json")
    # X_test_wtop2 = object_load("X/X_test_wtop2.json")
    # guess_accuracy(X_test_wtopb, X_test_wtop2)


if __name__=="__main__":
    main()