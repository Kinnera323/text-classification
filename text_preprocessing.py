from nltk import word_tokenize
from nltk.corpus import reuters 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from collections import defaultdict
import re
from nltk.corpus import stopwords
import csv
import operator

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



inverted_index_train = {}
inverted_index_test = {}
inverted_index_train_pruned = {}
inverted_index_test_pruned = {}
frequent_item_list = []

final_inverted_train_index = {}
final_category_docs = {}
trained_categories = {}


def build_index_train(doc_data,doc_id):
	for i in doc_data:
		doc_list  = []
		if i in inverted_index_train:
			doc_list = inverted_index_train[i]
			if doc_id in doc_list:
				continue
			else:
				doc_list.append(doc_id)				
		else:
			doc_list.append(doc_id)
		inverted_index_train[i] = doc_list	

def build_index_test(doc_data,doc_id):
	for i in doc_data:
		doc_list  = []
		if i in inverted_index_test:
			doc_list = inverted_index_test[i]
			if doc_id in doc_list:
				continue
			else:
				doc_list.append(doc_id)				
		else:
			doc_list.append(doc_id)
		inverted_index_test[i] = doc_list	

def generate_train_csv():

	with open("sentences_train.csv","wb") as f:
		writer = csv.writer(f,quoting=csv.QUOTE_ALL)	
		for doc_id in reuters.fileids():
			if doc_id.startswith("train"):
				raw_data = reuters.raw(doc_id).split('.')
				for sentence in raw_data:
					#sentence_list=[]
					if  len(tokenize(sentence)) >= 3:
						writer.writerow(tokenize(sentence))


def extract_csv():

	with open('frequent_train_set.csv','rb') as f:
		reader = csv.reader(f)
		for word in list(reader):
			frequent_item_list.append(word)	

def extract_category_csv():
	with open('category_train_docs.csv','rb') as f:
		reader = csv.reader(f)
		for word in list(reader):
			final_category_docs[word[0]] = word[1:]	


def extract_index_csv():

	with open('inverted_train_index.csv','rb') as f:
		reader = csv.reader(f)
		for word in list(reader):
			final_inverted_train_index[word[0]] = word[1:]	


def main():
	train_docs = [] # contains train document numbers
	test_docs = []  # contains test document numbers
	train_category_docs = {}  # contains category corresponding train documents
	test_category_docs = {}   # contains category corresponding test documents
	train_data = {}  # contains train document numbers corresponding data
	test_data = {}	 # contains test document numbers corresponding data
	

	categories = reuters.categories() # Total categories list

	#print categories

	#print "Category Name" + " <------------------> " +  "No of Train documents in each Category"
	with open("category_train_docs.csv","wb") as f:
		writer = csv.writer(f,quoting=csv.QUOTE_ALL)
		for category_name in categories:
			category_docs = reuters.fileids(category_name)
			#print category_name + " <------------------> " + str(len(category_docs))
			train_list = []
			test_list = []
			for category_id in category_docs:	
				if category_id.startswith("train"):
					train_list.append(category_id)
					
				else:
					test_list.append(category_id)
			writer.writerow([category_name] + train_list)

			#test_category_docs[category_name] = test_list
			#train_category_docs[category_name] = train_list


		

	for doc_id in reuters.fileids():
		if doc_id.startswith("train"):		
			train_docs.append(doc_id)
			train_data[doc_id] = tokenize(reuters.raw(doc_id))
			doc_number = doc_id.split('/')[1]
			build_index_train(tokenize(reuters.raw(doc_id)),doc_number)
			#train_docs.append(reuters.raw(doc_id))
		else:
			test_docs.append(doc_id)
			test_data[doc_id] = tokenize(reuters.raw(doc_id))
			doc_number = doc_id.split('/')[1]
			build_index_test(tokenize(reuters.raw(doc_id)),doc_number)

	#print train_data	
	

	with open("inverted_train_index.csv","wb") as f:
		writer = csv.writer(f,quoting=csv.QUOTE_ALL)
		for words in inverted_index_train:
			if len(inverted_index_train[words]) >= 3:
				inverted_index_train_pruned[words] = (inverted_index_train[words])
				writer.writerow([words] + inverted_index_train_pruned[words])



	for words in inverted_index_test:
		if len(inverted_index_test[words]) >= 3:
			inverted_index_test_pruned[words] = inverted_index_test[words] 

	#print len(inverted_index_train_pruned)
	#print len(inverted_index_test_pruned)
	#print len(train_docs)
	#print len(test_docs)


#extract_csv()
#generate_train_csv()
#main()
#print frequent_item_list


def test_weight_computation(test_doc_tokens):
	category_weights = {}
	remove_duplicates = []
	for tt in test_doc_tokens:
		if tt not in remove_duplicates:
			remove_duplicates.append(tt)

	test_doc_tokens = []
	test_doc_tokens = remove_duplicates

	for category_named in trained_categories:
		x = 0 
		for each_itemset in trained_categories[category_named]:
				if len(each_itemset) == 1:
					if each_itemset[0] in test_doc_tokens:
						x = x + len(each_itemset)

				elif len(each_itemset) == 2:
					if each_itemset[0] or  each_itemset[1] in test_doc_tokens:
						x = x + len(each_itemset)

				elif len(each_itemset) == 3:
					if each_itemset[0]  or each_itemset[1]  or each_itemset[2] in test_doc_tokens:
						x = x + len(each_itemset)

				elif len(each_itemset) == 4:
					if each_itemset[0]  or each_itemset[1]  or each_itemset[2] or each_itemset[3] in test_doc_tokens:
						x = x + len(each_itemset)
			
		print x		
		category_weights[category_named] = x

	val = max(category_weights.iteritems(),key=operator.itemgetter(1))
	#print val
	return val[0]





if __name__ == '__main__':
	
	#main()
	extract_csv()
	extract_index_csv()
	extract_category_csv()
	#print final_category_docs
	#print final_inverted_train_index['said']
	#print final_inverted_train_index['month']

	b =  list(set(final_inverted_train_index['said']) & set(final_inverted_train_index['month']))
	#print list(set(final_category_docs['acq']) & set(b))
	

	for item_set in frequent_item_list:
		
		#print item_set
		
		document_item_set = final_inverted_train_index[item_set[0]]
		weight_category = {}
		
		if len(item_set) == 1:
			for category_name in final_category_docs:
				match_length = len(list(set(final_category_docs[category_name]) & set(document_item_set)))
				category_name_length = len(final_category_docs[category_name])
				weight_category[category_name] = round(match_length/float(category_name_length),3)
		else:
			for word in item_set[1:]:
				document_item_set = list(set(final_inverted_train_index[word]) & set(document_item_set))
			#print document_item_set
			
			for category_name in final_category_docs:
				match_length = len(list(set(final_category_docs[category_name]) & set(document_item_set)))
				category_name_length = len(final_category_docs[category_name])
				weight_category[category_name] = round(match_length/float(category_name_length),3)

				#weight_category[category_name] = float(len(list(set(final_category_docs[category_name]) & set(document_item_set)))/len(final_category_docs[category_name]))
				#print weight_category[category_name]
		#print weight_category
		val = max(weight_category.iteritems(),key=operator.itemgetter(1))[0]
		for i in weight_category:
			if weight_category[i] == weight_category[val]:
				if i not in trained_categories:
						trained_categories[i] = []
						trained_categories[i].append(item_set)
				else:
						trained_categories[i].append(item_set)

	final_test_categories = {}

	for doc_id in reuters.fileids():
		if doc_id.startswith("test"):		
			#test_data[doc_id] = tokenize(reuters.raw(doc_id))
			doc_number = doc_id.split('/')[1]
			if test_weight_computation(tokenize(reuters.raw(doc_id))) not in final_test_categories:
				final_test_categories[test_weight_computation(tokenize(reuters.raw(doc_id)))] = []
				final_test_categories[test_weight_computation(tokenize(reuters.raw(doc_id)))].append(doc_number)
			else:
				final_test_categories[test_weight_computation(tokenize(reuters.raw(doc_id)))].append(doc_number)
			
			#exit()
	

	for tt in final_test_categories:
		print tt + ' ' + str(final_test_categories[tt])

	

	test_category_list = {}
	categories = reuters.categories() # Total categories list
	for category_name in categories:
			category_docs = reuters.fileids(category_name)
			test_category_list[category_name] = []
			for category_id in category_docs:	
				if category_id.startswith("test"):
					test_category_list[category_name].append(category_id.split('/')[1])


	for tt in final_test_categories:
		true_positive_list = list(set(final_test_categories[tt]) & set(test_category_list[tt]))
		
		print tt + ' ' + str(len(true_positive_list))
		false_positive_list = []
		
		for mk in final_test_categories[tt]:
			if mk not in true_positive_list:
				false_positive_list.append(mk)

		false_negative_list = []

		for mk in test_category_list[tt]:
			if mk not in true_positive_list:
				false_negative_list.append(mk)

		print len(true_positive_list)/float(len(true_positive_list) + len(false_positive_list))
