"""model_factory script containing ModelFactory"""
import numpy as np
import os
import constants as c
from survey_data_parser import SurveyDoc2VecDataParser, SurveyTheanoDataParser
from doc2vec_model import Doc2VecModel
from doc2vec_to_np_data_parser import Doc2VecToNumpyDataParser
from sgd_model import SGDModel
from gbc_model import GBClassModel
from lstm_model import LSTMModel
from sklearn.cross_validation import train_test_split
from survey_util import Util

MODEL_OBJ = {
    c.SGD: SGDModel(),
    c.GBC: GBClassModel(),
    c.LSTM: LSTMModel(),
    None: None
}

DATA_PARSER_OBJ = {
    c.SGD: SurveyDoc2VecDataParser(),
    c.GBC: SurveyDoc2VecDataParser(),
    c.LSTM: SurveyTheanoDataParser(),
    None: None
}

class ModelFactory(object):
    """ModelFactory"""
    def __init__(self, model_type, remove_stopwords_flag=False, words_stemming_flag=False):
        """init"""
        self.model_type = model_type
        self.model_out = MODEL_OBJ[self.model_type]
        self.data_parser = DATA_PARSER_OBJ[self.model_type]
        if self.data_parser is not None:
            self.data_parser.remove_stopwords_flag = remove_stopwords_flag
            self.data_parser.words_stemming_flag = words_stemming_flag
        self.remove_stopwords_flag = remove_stopwords_flag
        self.words_stemming_flag = words_stemming_flag

        if self.model_out is None:
            if os.path.exists(c.LSTM_FILE):
                self.model_out = MODEL_OBJ[c.LSTM]
                self.model_out.load(c.LSTM_FILE)
            if os.path.exists(c.SGD_FILE):
                self.model_out = MODEL_OBJ[c.SGD]
                self.model_out.load(c.SGD_FILE)
            elif os.path.exists(c.GBC_FILE):
                self.model_out = MODEL_OBJ[c.GBC]
                self.model_out.load(c.GBC_FILE)

    def model_train(self, train_file):
        """train model"""
        self._delete_model_files()
        self.data_parser.load(train_file)

        if self.model_type == c.LSTM:
            print("Building RNN representations of training data")
            word2idx, idx2word, _, x_word2idx_out, y_dict, y_inverse_dict, _, y_onehot = self.data_parser.convert()
            self.data_parser.save(self.model_type+"_meta.npz", None)
            np.savez(
                c.LSTM_EMBEDDING_FILE,
                word2idx=word2idx,
                idx2word=idx2word,
                y_dict=y_dict,
                y_inverse_dict=y_inverse_dict)
            self.model_out.fit([x_word2idx_out, y_onehot])
            self.model_out.save(c.LSTM_FILE)
        else:
            print("Converting training data to LabeledSentences")
            labeled_sentences = self.data_parser.convert()
            print("Fitting Doc2Vec model")
            docvec_model_out = Doc2VecModel(epochs=1000, dim_size=400, window_size=10)
            docvec_model_out.fit(labeled_sentences)
            print("Saving Doc2Vec model")
            docvec_model_out.save(c.DOC2VEC_MODEL_FILE)
            if self.model_out is not None:
                print("Preparing for SGDClassifier model")
                doc2vec_model = docvec_model_out.model
                print("Converting from Doc2Vec vectors to numpy array")
                data_parser2 = Doc2VecToNumpyDataParser(doc2vec_model)
                x_out, y_out, y_dict, y_rev_dict = data_parser2.convert()
                print("x_out shape: ", x_out.shape)
                print("y_out shape: ", y_out.shape)
                print("y_dict: ", y_dict)
                print("y_reverse_dict: ", y_rev_dict)
                x_train, x_test, y_train, y_test = train_test_split(x_out, y_out, test_size=0.1)
                sgd_model = SGDModel()
                print("Fitting SGDClassifier model")
                sgd_model.fit([x_train, y_train])
                print("Testing against validation data")
                y_pred = sgd_model.predict(x_test)
                y_pred_max = Util.get_argmax_from_prob_array(y_pred)
                print("correctly predicted: ", (np.sum(y_pred_max == y_test)/y_test.shape[0]))
                top_classification = Util.get_np_dict_value_by_idx(y_rev_dict, y_pred, 2)
                print("top classification: ", top_classification)
                print("Saving model")
                sgd_model.save(c.MODEL_DIR+"/"+ self.model_type + "_" + c.DISCRIMINATOR_MODEL_FILE)

    def sentence_prediction(self, sentence=None, top_class_number=2, threshold=0.3):
        """print predictions"""
        """predict sentence"""
        if os.path.exists(c.LSTM_FILE):
            self.model_type = c.LSTM
            embedding_array = np.load(c.LSTM_EMBEDDING_FILE)
            word2idx = embedding_array["word2idx"].item()
            unknown = word2idx['UNKNOWN']
            sentence_list = []
            sentence_embedded = np.array([word2idx[w] if w in word2idx else unknown for w in sentence.split()])
            sentence_list.append(sentence_embedded)
            y_out = self.model_out.predict(sentence_list)
            y_inverse_dict = embedding_array["y_inverse_dict"].item()
        else:
            docvec_model_out = Doc2VecModel()
            docvec_model_out.load(c.DOC2VEC_MODEL_FILE)
            x_test_sen = sentence
            if self.remove_stopwords_flag:
                x_test_sen = Util.remove_stop_words(x_test_sen)
                x_test_sen = Util.remove_pronouns(x_test_sen)
            if self.words_stemming_flag:
                x_test_sen = Util.stem_words(x_test_sen)
            x_test, _ = docvec_model_out.predict(x_test_sen.lower())
            data_parser = Doc2VecToNumpyDataParser(docvec_model_out.model)
            _, _, _, y_inverse_dict = data_parser.convert()
            y_out = self.model_out.predict(x_test)

        print("Sentence: ", sentence)
        top_classification = Util.get_np_dict_value_by_idx(y_inverse_dict, y_out, top_class_number)
        print("Classifications: ", top_classification)
        y_out_sorted = np.sort(y_out[0])[::-1]
        print("Probability Score: ", y_out_sorted[:top_class_number])

        y_out_sorted = y_out_sorted[:top_class_number]
        for i in range(top_class_number):
            if i == 0:
                if y_out_sorted[i] < threshold:
                    y_out_sorted = np.array([])
                    break
            if y_out_sorted[i] < threshold:
                y_out_sorted = np.delete(y_out_sorted, i)
                top_classification = np.delete(top_classification, i)
        if y_out_sorted.shape[0] <= 0:
            print("The Probability Scores do not meet the threshold of ", threshold, " \n")    
        else:
            print("Top Classifications: ", top_classification)
            print("Top Probability Score: ", y_out_sorted, "\n")

    def _delete_model_files(self):
        """delete model files"""
        if os.path.exists(c.SGD_FILE):
            os.remove(c.SGD_FILE)
        if os.path.exists(c.GBC_FILE):
            os.remove(c.GBC_FILE)
        if os.path.exists(c.LSTM_FILE):
            os.remove(c.LSTM_FILE)
        if os.path.exists(c.LSTM_EMBEDDING_FILE):
            os.remove(c.LSTM_EMBEDDING_FILE)      
