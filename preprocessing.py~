from nltk import word_tokenize
from nltk.corpus import reuters 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from collections import defaultdict
import re
from nltk.corpus import stopwords
import csv

all_words_train = {}
all_words_test = {}
cachedStopWords = stopwords.words("english")

def tokenize(text):
	min_length = 3
	words = map(lambda word: word.lower(), word_tokenize(text));
	words = [word for word in words if word not in cachedStopWords]
	tokens =(list(map(lambda token: PorterStemmer().stem(token), words)));
	p = re.compile('[a-zA-Z]+');
	filtered_tokens = list(filter(lambda token: p.match(token) and len(token)>=min_length, tokens));
	return filtered_tokens

# Return the representer, without transforming
def tf_idf(docs):	
	tfidf = TfidfVectorizer(tokenizer=tokenize, min_df=3, max_df=0.90, max_features=1000, use_idf=True, sublinear_tf=True);
	tfidf.fit(docs);
	return tfidf;

def feature_values(doc, representer):
	doc_representation = representer.transform([doc])
	features = representer.get_feature_names()
	return [(features[index], doc_representation[0, index]) for index in doc_representation.nonzero()[1]]

def collection_stats():
	# List of documents
	documents = reuters.fileids()
	print(str(len(documents)) + " documents");
	
	train_docs = list(filter(lambda doc: doc.startswith("train"), documents));
	print(str(len(train_docs)) + " total train documents");
	
	test_docs = list(filter(lambda doc: doc.startswith("test"), documents));	
	print(str(len(test_docs)) + " total test documents");

	# List of categories 
	categories = reuters.categories();
	print(str(len(categories)) + " categories");

	# Documents in a category
	category_docs = reuters.fileids("acq");

	# Words for a document
	document_id = category_docs[0]
	document_words = reuters.words(category_docs[0]);
	print(document_words);	

	# Raw document
	print(reuters.raw(document_id));



# Initial functions which gets the required data for indexing
def build_index_test(doc_data,doc_id):
	for i in doc_data:
		if i in all_words_test:
			if doc_id in all_words_test[i]:
				continue
			else:
				x = all_words_test[i]
				all_words_test[i] += doc_id
				all_words_test[i] += ' '				
		else:
			all_words_test[i] = defaultdict(list)
			all_words_test[i] = doc_id
			all_words_test[i] += ' '	

def build_index_train(doc_data,doc_id):
	for i in doc_data:
		if i in all_words_train:
			if doc_id in all_words_train[i]:
				continue
			else:
				x = all_words_train[i]
				all_words_train[i] += doc_id
				all_words_train[i] += ' '				
		else:
			all_words_train[i] = defaultdict(list)
			all_words_train[i] = doc_id
			all_words_train[i] += ' '	
		

def main():
	train_docs = []
	test_docs = []
	order_test_docs = []
	test_list_data = []
	with open("Frequent_train.csv","wb") as f:
		writer = csv.writer(f,quoting=csv.QUOTE_ALL)	
		for doc_id in reuters.fileids():
			if doc_id.startswith("train"):
				writer.writerow(tokenize(reuters.raw(doc_id)))
				#order_train_docs.append(doc_id)
				print "######"		
				train_docs.append(reuters.raw(doc_id))
				doc_number = doc_id.split('/')[1]
				build_index_train(tokenize(reuters.raw(doc_id)),doc_number)
				
			else:
				#writer.writerow(tokenize(reuters.raw(doc_id)))
				#order_test_docs.append(doc_id)
				#test_list_data.append(tokenize(reuters.raw(doc_id)))
				#print "######"
				#doc_number = doc_id.split('/')[1]
				#build_index_test(tokenize(reuters.raw(doc_id)),doc_number)
				test_docs.append(reuters.raw(doc_id))
	
	
		#print(all_words)
	exit()

	inverted_file_test = {}
	for i in all_words_test:
		e = all_words_test[i].split(' ')[:-1]
		if len(e)>= 3 and len(e) <= 90:
			inverted_file_test[i] = e

	inverted_file_train = {}
	for i in all_words_train:
		e = all_words_train[i].split(' ')[:-1]
		if len(e)>= 3 and len(e) <= 90:
			inverted_file_train[i] = e
			
	print(inverted_file_train)	
	exit()	


main()
