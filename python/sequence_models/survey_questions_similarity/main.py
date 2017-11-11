"""program main"""
import warnings
warnings.filterwarnings("ignore")
import sys, os
import numpy as np
import constants as c
from survey_data_parser import SurveyDoc2VecDataParser, SurveyTheanoDataParser
from doc2vec_model import Doc2VecModel
from doc2vec_to_np_data_parser import Doc2VecToNumpyDataParser
from sgd_model import SGDModel
from gbc_model import GBClassModel
from lstm_model import LSTMModel
from sklearn.cross_validation import train_test_split
from survey_util import Util
from model_factory import ModelFactory

def main_menu():
    """exec_menu"""
    os.system('cls')
    print("Welcome,\n")
    print("Please choose the menu you want to start:")
    print("1. Vectorise training data and train a new model")
    print("2. Test the model against a test file (default data/test_question.txt)")
    print("0. Quit")
    print("Or enter some text to return the most similar reference questions")
    choice = input(" >>")
    exec_menu(choice)


def train_action():
    """train_action method that triggers the model training"""
    print("Select the model. (Doc2Vec by default): ")
    print("1. Stochastic Gradient Descent")
    print("2. Gradient Boosting Classification")
    print("3. Simple LSTM")
    choice = input(" >>")
    model_type = None
    model_options = {
        '1': c.SGD,
        '2': c.GBC,
        '3': c.LSTM
    }
    model_type = model_options[choice]

    if model_type is None:
        print("No such option. Hit the Enter key to continue")
        MENU_ACTIONS['main_menu']()

    print("Enter a filename (or hit enter to use data/labeled_data.csv):")
    choice = input(" >>")
    train_file = "data/labeled_data.csv"
    if choice != '':
        train_file = choice
    if os.path.exists(train_file) is False:
        print("File does not exist. Going back to the main menu")
        print("Hit the Enter key to continue")
        input(">>")
        MENU_ACTIONS['main_menu']()
    model_factory = ModelFactory(model_type=model_type)
    model_factory.model_train(train_file)
    print("Training complete. Hit the Enter key to continue")
    choice = input(" >>  ")
    MENU_ACTIONS['main_menu']()

def predict_test_file_action():
    """train_action"""
    print("Enter a filename (or hit enter to use data/test_questions.txt):")
    choice = input(" >>  ")
    test_file = "data/test_questions.txt"
    if choice != '':
        test_file = choice
    if os.path.exists(test_file) is False:
        print("File does not exist. Going back to the main menu")
        print("Hit the Enter key to continue")
        input(">>")
        MENU_ACTIONS['main_menu']()
    data_parser1 = SurveyDoc2VecDataParser()
    print("Loading test file")
    x_out, _ = data_parser1.load(test_file)
    os.system('cls')
    
    model_factory = ModelFactory(model_type=None)
    for sentence in x_out:
        model_factory.sentence_prediction(sentence)

    print("Hit the Enter key to continue")
    input(">>")
    MENU_ACTIONS['main_menu']()

# Back to main menu
def back():
    """back"""
    MENU_ACTIONS['main_menu']()

# Exit program
def exit_program():
    """exit"""
    sys.exit()

# Menu definition
MENU_ACTIONS = {
    'main_menu': main_menu,
    '1': train_action,
    '2': predict_test_file_action,
    '0': exit_program
}

# Execute menu
def exec_menu(choice):
    """exec_menu"""
    os.system('cls')
    choice = choice.lower()
    if choice == '':
        MENU_ACTIONS['main_menu']()
    elif choice.isdigit():
        try:
            MENU_ACTIONS[choice]()
        except KeyError:
            print("Invalid selection, please try again.\n")
            MENU_ACTIONS['main_menu']()
    else:
        model_factory = ModelFactory(model_type=None)
        model_factory.sentence_prediction(choice)
        print("Hit the Enter key to continue")
        input(">>")
        MENU_ACTIONS['main_menu']()
    return

# Main Program
if __name__ == "__main__":
    main_menu()
