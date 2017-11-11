# Prerequisites
	Python 3.5
	## Libaries
		- numpy
		- sklearn
		- gensim
		- nltk

# Program Execution
The program can be run with the following command:
	python main.py

The below menu should be presented on the console:
	Welcome,

	Please choose the menu you want to start:
	1. Vectorise training data and train a new model
	2. Test the model against a test file (default data/test_question.txt)
	0. Quit
	Or enter some text to return the most similar reference questions
	 >>
	
	## 1. Vectorise training data and train a new model
	This option vectorises and trains the training data. It will prompt you to select a classification model:
	
		Select the model. (Doc2Vec by default):
		1. Stochastic Gradient Descent
		2. Gradient Boosting Classification
		3. Simple LSTM

	It then prompts you to enter a training csv file, or if you press enter, the program will use the default data/labeled_data.csv file. 
	
	## 2. Test the model against a test file (default data/test_question.txt)
	This option attempts to predict the classifications of sentences in a test file. It prompts you to enter a test txt file, or if you press enter, the program will use the default data/test_questions.txt file. Below is an example classification output from the program. It will print the Sentence, the top two classifications, and their respective probabilities, as well as the classifications that are higher than the threshold (default 0.3).
		
		Sentence:  we hold ourselves and our team members accountable for results
		Classifications:  [['TEA.2', 'INN.2']]
		Probability Score:  [ 0.96869981  0.06744113]
		Top Classifications:  ['TEA.2']
		Top Probability Score:  [ 0.96869981]

	
	## or enter some text to return the most similar reference questions
	You can also type in a sentence and hit enter at the main menu screen. The above mentioned sentence classification output in the previous section will be presented on the console for the input sentence.
	
# Algorithm Description
The program's algorithm is broken into 2 stages -- Sentence Vectorization and Classification. Note that training data is not processed in the Sentence Vectorization stage, as it has it own Word Embeddings implemented for that Model.

	## Sentence Vectorization
	 Doc2Vec is a method similar to Word2Vec. It generalises the Word2Vec by adding a paragraph/document vector. Like Word2Vec, there are two methods: Distributed Memory (DM) and Distributed Bag of Words (DBOW). DM attempts to predict a word given its previous words and a paragraph vector. Even though the context window moves across the text, the paragraph vector does not (hence distributed memory) and allows for some word-order to be captured.
	 
	 The objective of this part of the algorithm is to obtain a mathematical representation of the text, and group the text with similar numerical properties together.
	 
	 - The algorithm is currently configured to use DM. 
	 - It removes stop words from the sentence.
	 
	 The sentences are preprocessed at this stage prior to forwarding it to the Classification stage of the algorithm.
	 
	## Classification
	There are two classifiers implemented for the algorithm.
		### Stochastic Gradient Descent Classifier
		The Stochastic Gradient Descent classifier is a stochastic approximation of the gradient descent optimization method for minimizing an objective function that is written as a sum of differentiable functions. The method attempts to find minima or maxima by iteration.
		
		### Gradient Boosting Classifier
		Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It builds the model in a stage-wise fashion like other boosting methods do, and it generalizes them by allowing optimization of an arbitrary differentiable loss function.

		### LSTM
		Long short-term memory (LSTM) is a recurrent neural network (RNN) architecture that remembers values over arbitrary intervals. Stored values are not modified as learning proceeds. RNNs allow forward and backward connections between neurons.
		An LSTM is well-suited to classify, process and predict time series given time lags of unknown size and duration between important events. Relative insensitivity to gap length gives an advantage to LSTM over alternative RNNs, hidden Markov models and other sequence learning methods in numerous applications.
		The output of the network is a sigmoid function with 4 output nodes (containing the label classifications by probability), and the loss function used is binary cross entropy with an Adam optimiser - a SGD algorithm in which the gradient used in each iteration is updated from the previous using a technique based in momenta.
		
		### Classification Output
		A probability of each classes (ALI.5, ENA.3, INN.2, TEA.2) is returned from the two models.
		
		### Training, Validation and Test data
		There is a holdout on validation data for each of the models, as well as a holdout test data set in the main program.


## Software Project Description
Below is a description of the project directory structure and file description
	# Parent Direectory
	* abstract_data_parser.py - Abstract base class for data parsers.
	* abstract_model.py - Abstract base class for models.
	* doc2vec_model.py - Script containing the Doc2VecModel class.
	* doc2vec_to_np_data_parser.py - Script with the Doc2VecToNumpyDataParser class, that converts a Doc2Vec vector into a numpy array.
	* gbc_model.py - GradientBoostingClassifier model class script
	* lstm_model.py - LSTM model class script
	* model_factory.py - Factory class for training and predicting of models
	* main.py - The main script
	* README.txt - readme file
	* sgd_model.py - Stochastic Gradient Descent model
	* survey_data_parser.py - Parses CSV survey label file into a TaggedDocument format for Doc2Vec to consume
	* survey_util.py - Script containing Util methods
	
	# test
	The test directory contains the unit test scripts
	* doc2vec_model_test.py - Unit tests for the Doc2VecModel class
	* doc2vec_to_np_data_parser_test.py - Unit tests for the Doc2VecToNumpyDataParser class
	* sgd_model_test.py - Unit tests for the SGDModel class
	* survey_data_parser_test.py - Unit tests for the SGDModel class
	* TestDoc2VecModel.doc2vec - Unit tests for the SurveyDoc2VecDataParser class
	* TSNE_visualise.png  - A png file with a TSNE plot generated by the doc2vec_model_test.py test script. It contains a TSNE transformation of the Doc2Vec sentence vectors.
	
	# model_files
	The model_files directory is where the program will save and load models.
	
	# data
	THe data directory contains the default training and testing files.

