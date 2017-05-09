import sys, getopt
from sklearn.externals import joblib
from Word2VecUtility import Word2VecUtility
import pickle
import numpy as np

def predict(messages, threshold):
	sentiment_model = joblib.load('sentiment.pkl') 
	sentiment_vectorizer = pickle.load(open("sentiment_vec.pickle", "rb"))

	polarity_model = joblib.load('polarity.pkl') 
	polarity_vectorizer = pickle.load(open("polarity_vec.pickle", "rb"))

	def create_data_features(dset, vectorizer):
	    clean_reviews = []
	    for i in range( 0, len(dset)):
	        clean_reviews.append(" ".join(Word2VecUtility.review_to_wordlist(
	            str(list(dset)[i]), False)))
	    data_features = vectorizer.transform(clean_reviews)
	    np.asarray(data_features)
	    return data_features        

	sentiment_feat = create_data_features(messages, sentiment_vectorizer)
	polarity_feat = create_data_features(messages, polarity_vectorizer)

	sr = sentiment_model.predict_proba(sentiment_feat)
	pr = polarity_model.predict_proba(polarity_feat)

	print("Sentiment detection: spam - %i%%, sentiment - %i%%"%(100*sr[0][0],100*sr[0][1]))
	print("Polarity detection: negative - %i%%, positive - %i%%"%(100* pr[0][0],100*pr[0][1]))

	output_file = open("output.csv", "w")
	output_file.write("message\tsentiment\tpolarity\n")
	for i in range(len(messages)):
		if sr[i][1] > threshold:
			polarity = "negative"
			if pr[i][0] < 0.5:
				polarity = "positive"
			line = "%s\t%i%%\t%s\n" % (messages[i], 100*sr[i][1], polarity)
			output_file.write(line)
	return 0

def main(argv):
   messages = []
   filename = ''
   threshold = 0
   try:
      opts, args = getopt.getopt(argv,"hi:f:t:",["i=", "f=", "t="])
   except getopt.GetoptError:
      print ('test.py -i <chatmessage> OR -f <filename> -t <threshold>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
        print ('test.py -i <chatmessage>')
        sys.exit()
      elif opt in ("-i"):
        messages = [ arg ]
      elif opt in ("-f"):
      	filename = arg
      	print(filename)
      	with open(filename, "r") as f:
      		lines = f.readlines()
      	messages = [x.strip() for x in lines]	
      elif opt in ("-t"):
      	threshold = float(arg) / 100
   predict(messages, threshold)
if __name__ == "__main__":
   main(sys.argv[1:])

