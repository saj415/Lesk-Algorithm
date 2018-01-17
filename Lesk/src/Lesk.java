/**
 * Implement the Lesk algorithm for Word Sense Disambiguation (WSD)
 * @author Sachin Joshi
 * Project 4 
 * NLP CSE 498 Fall 2017
 * Prof. Sihong Xie
 */
 
import java.util.*;
import java.io.*;
import javafx.util.Pair;

import edu.mit.jwi.*;
import edu.mit.jwi.Dictionary;
import edu.mit.jwi.item.*; 

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.logging.RedwoodConfiguration;

public class Lesk {

	/** 
	 * Each entry is a sentence where there is at least a word to be disambiguate.
	 * E.g.,
	 * 		testCorpus.get(0) is Sentence object representing
	 * 			"It is a full scale, small, but efficient house that can become a year' round retreat complete in every detail."
	 **/
	private ArrayList<String> testCorpus = new ArrayList<String>();
	
	/** Each entry is a list of locations (integers) where a word needs to be disambiguate.
	 * The index here is in accordance to testCorpus.
	 * E.g.,
	 * 		ambiguousLocations.get(0) is a list [13]
	 * 		ambiguousLocations.get(1) is a list [10, 28]
	 **/
	private ArrayList<ArrayList<Integer> > ambiguousLocations = new ArrayList<ArrayList<Integer> >();
	
	/**
	 * Each entry is a list of pairs, where each pair is the lemma and POS tag of an ambiguous word.
	 * E.g.,
	 * 		ambiguousWords.get(0) is [(become, VERB)]
	 * 		ambiguousWords.get(1) is [(take, VERB), (apply, VERB)]
	 */
	private ArrayList<ArrayList<Pair<String, String> > > ambiguousWords = new ArrayList<ArrayList<Pair<String, String> > > (); 
	
	/**
	 * Each entry is a list of maps, each of which maps from a sense key to similarity(context, signature)
	 * E.g.,
	 * 		predictions.get(1) = [{take%2:30:01:: -> 0.9, take%2:38:09:: -> 0.1}, {apply%2:40:00:: -> 0.1}]
	 */
	private ArrayList<ArrayList<HashMap<String, Double> > > predictions = new ArrayList<ArrayList<HashMap<String, Double> > >();
	
	/**
	 * Each entry is a list of ground truth senses for the ambiguous locations.
	 * Each String object can contain multiple synset ids, separated by comma.
	 * E.g.,
	 * 		groundTruths.get(0) is a list of strings ["become%2:30:00::,become%2:42:01::"]
	 * 		groundTruths.get(1) is a list of strings ["take%2:30:01::,take%2:38:09::,take%2:38:10::,take%2:38:11::,take%2:42:10::", "apply%2:40:00::"]
	 */
	private ArrayList<ArrayList<String> > groundTruths = new ArrayList<ArrayList<String> >();
	
	/* This section contains the NLP tools */
	
	private Set<String> POS1 = new HashSet<String>(Arrays.asList("ADJECTIVE", "ADVERB", "NOUN", "VERB"));
	
	private Dictionary wordnetdict;
	
	private StanfordCoreNLP pipeline;

	private Set<String> stopwords;
	
	/* Constructor to intialise the class variables*/
	public Lesk() {
		try{
			/* Open the WordNet Dictionary */
			wordnetdict = new Dictionary(new File("data/dict/"));
			wordnetdict.open();
			
			/* Constructing the CoreNLP object from a given set of properties (tokenization, lemmatization and POS tagging) using the annotators*/
			Properties props = new Properties();
			props.setProperty("annotators", "tokenize, ssplit, pos, lemma");
			pipeline = new StanfordCoreNLP(props);
			
			/* Reading the stopwords.txt file and storing the stopwords in the given set*/
			stopwords = new HashSet<String>(); // Construct the stopwords set object
			BufferedReader br = new BufferedReader(new FileReader("data/stopwords.txt")); // Open the file
			String s;
			while((s = br.readLine()) != null){ // Reading the file
				stopwords.add(s); // Add the stopwords in the set
			}
			br.close();
		}
		catch(Exception e){
			e.printStackTrace();
		}
	}
	
	/**
	 * Convert a pos tag in the input file to a POS tag that WordNet can recognize (JWI needs this).
	 * We only handle adjectives, adverbs, nouns and verbs.
	 * @param pos: a POS tag from an input file.
	 * @return JWI POS tag.
	 */
	private String toJwiPOS(String pos) {
		if (pos.equals("ADJ")) {
			return "ADJECTIVE";
		} else if (pos.equals("ADV")) {
			return "ADVERB";
		} else if (pos.equals("NOUN") || pos.equals("VERB")) {
			return pos;
		} else {
			return pos;
		}
	}

	/**
	 * This function fills up testCorpus, ambiguousLocations and groundTruths lists
	 * @param filename
	 */
	public void readTestData(String filename) throws IOException {
		
		/* Read a single text file and extract the sentences, ambiguous words, its clocation and associated lemma and POS tag */
		BufferedReader br = new BufferedReader(new FileReader(filename)); // Open the file
		
		/* Temporary Variables to perform data extraction*/
		String s; ArrayList<Integer> loc = null; ArrayList<Pair<String, String>> lemma_POS = null; ArrayList<String> gt = null;
		
		while((s = br.readLine())!= null) { // Read the contents of the file
			
			if(s.charAt(0)!= '#' && ((s.length()) != 1 && (s.length()) != 2)) { // check if the string is an ambiguous sentence
				
				testCorpus.add(s); // add the ambiguous sentence to the testCorpus
				
				if(testCorpus.size()!= 1){
					ambiguousLocations.add(loc); // adding the ambiguous locations
					ambiguousWords.add(lemma_POS); // adding the (lemma, POS) Pair
					groundTruths.add(gt); // adding all the possible senses associated with the ambiguous word 
				}
				
				loc = new ArrayList<Integer>();
				lemma_POS = new ArrayList<Pair<String, String>>();
				gt = new ArrayList<String>();
			}
			
			if(s.charAt(0) == '#') { // check if the string contains information about the target words (all strings that start with #)
				
				String temp[] = s.split(" ");
				loc.add(Integer.parseInt(temp[0].substring(1))); // extract the ambiguous location
				String lemma = temp[1]; // extract the associated lemma
				String POS = toJwiPOS(temp[2]); // extract the POS tag and convert it to the Wordnet dict format by calling the function toJwiPOS()
				lemma_POS.add(new Pair(lemma, POS)); 
				
				ArrayList<String> sen = parseSenseKeys(temp[3]); // Extract each of the senses separated by comma by calling the function parseSenseKeys() 
				for(String senses : sen) {
					gt.add(senses); // Add all the senses to the temporary variable
				}	
			}
		}
		br.close();
		ambiguousLocations.add(loc); // adding the ambiguous locations
		ambiguousWords.add(lemma_POS); // adding the (lemma, POS) Pair
		groundTruths.add(gt); // adding all the possible senses associated with the ambiguous word
	}
	
	/**
	 * Create signatures of the senses of a pos-tagged word.
	 * 
	 * 1. use lemma and pos to look up IIndexWord using Dictionary.getIndexWord()
	 * 2. use IIndexWord.getWordIDs() to find a list of word ids pertaining to this (lemma, pos) combination.
	 * 3. Each word id identifies a sense/synset in WordNet: use Dictionary's getWord() to find IWord
	 * 4. Use the getSynset() api of IWord to find ISynset
	 *    Use the getSenseKey() api of IWord to find ISenseKey (such as charge%1:04:00::)
	 * 5. Use the getGloss() api of the ISynset interface to get the gloss String
	 * 6. Use the Dictionary.getSenseEntry(ISenseKey).getTagCount() to find the frequencies of the synset.d
	 * 
	 * @param args
	 * lemma: word form to be disambiguated
	 * pos_name: POS tag of the wordform, must be in {ADJECTIVE, ADVERB, NOUN, VERB}.
	 * 
	 */
	private Map<String, Pair<String, Integer> > getSignatures(String lemma, String pos_name) {
		
		/* Get the indexWord corresponding to a particular (lemma, POS) pair from the WordNet Dictionary */
		IIndexWord idxWord = null;
		if(pos_name.equals("NOUN")){
			idxWord = wordnetdict.getIndexWord(lemma, POS.NOUN);
		}
		else if(pos_name.equals("VERB")){
			idxWord = wordnetdict.getIndexWord(lemma, POS.VERB);
		}	
		else if(pos_name.equals("ADVERB")){
			idxWord = wordnetdict.getIndexWord(lemma, POS.ADVERB);
		}
		else{
			idxWord = wordnetdict.getIndexWord(lemma, POS.ADJECTIVE);
		}
		
		/* For each wordID of the idxWord extract the all the senses, corresponding signatures of the senses and the TagCount and store it in a HashMap*/
		HashMap<String, Pair<String, Integer>> signatures = new HashMap<String, Pair<String, Integer>>();
		for(int i = 0; i < idxWord.getWordIDs().size(); i++){
			IWordID wordID = idxWord.getWordIDs().get(i); // Get the wordID of an idxWord
			IWord word = wordnetdict.getWord(wordID); // Get the word corresponding to the wordID 
			
			String key = word.getSenseKey().toString(); // Get the senseKey
			signatures.put(key, new Pair(word.getSynset().getGloss(), wordnetdict.getSenseEntry(word.getSenseKey()).getTagCount())); // Fill the HashMap
		}
	
		return signatures;
	}
	
	/**
	 * Create a bag-of-words representation of a document (a sentence/phrase/paragraph/etc.)
	 * @param str: input string
	 * @return a list of strings (words, punctuation, etc.)
	 */
	private ArrayList<String> str2bow(String str) {
		String text = str.toLowerCase();
		
		// create an empty Annotation just with the given text
        Annotation document = new Annotation(text);

        // run all Annotators on this text
        pipeline.annotate(document);
		
		return(new ArrayList<String>(Arrays.asList(text.split(" "))));
		
	}
	
	/**
	 * compute similarity between two bags-of-words.
	 * @param bag1 first bag of words
	 * @param bag2 second bag of words
	 * @param sim_opt COSINE or JACCARD similarity
	 * @return similarity score
	 */
	private double similarity(ArrayList<String> bag1, ArrayList<String> bag2, String sim_opt) {
		
		/* JACCARD Similarity between the Context and the Signature */
		Set<String> common = new HashSet<>(bag1);
		
		/* Get all the common words between the Context and the Signature*/
		common.retainAll(new HashSet<>(bag2));
		
		/* Get the Union of the  Context and the Signature*/
		ArrayList<String> merged = new ArrayList(bag1);
		merged.addAll(bag2);
		
		/* Compute and Return the JACCARD Similarity */
		return((double)common.size() / merged.size());
		
	}
	
	/**
	 * This is the WSD function that prediction what senses are more likely.
	 * @param context_option: one of {ALL_WORDS, ALL_WORDS_R, WINDOW, POS}
	 * @param window_size: an odd positive integer > 1
	 * @param sim_option: one of {COSINE, JACCARD}
	 */
	public void predict(String context_option, int window_size, String sim_option) {
		
		/* For each ambiguous sentence*/
		for(int i = 0; i < ambiguousWords.size(); i++) {
			
			/* Temporary Variable to store the similarity for all the targets words in a sentence  */
			ArrayList<HashMap<String, Double>> pred = new ArrayList<HashMap<String, Double>>();
			
			/* For each ambiguous word*/
			for(int j = 0; j < ambiguousWords.get(i).size(); j++) {
				
				String lem = ambiguousWords.get(i).get(j).getKey(); // Get the lemma of the target word
				String pos = ambiguousWords.get(i).get(j).getValue(); // Get the POS tag of the target word
				
				if(pos.equals("NOUN") || pos.equals("VERB") || pos.equals("ADVERB") || pos.equals("ADJECTIVE")){
					
					/* Get all the signatures corresponding to the (lemma, POS) pair*/
					Map<String, Pair<String, Integer>> sign = getSignatures(lem, pos);
					
					/* Call the Lesk algorithm that disambiguates the senses and also calculates the similarity score*/
					pred.add((Lesk(testCorpus.get(i), sign, lem, context_option)));
				}
			}
			
			predictions.add(pred); // Add the similarity score of each target word to the predictions data set
			
		}
		//System.out.println(predictions.get(1).get(0).keySet());
	}
	
	/**
	 * Lesk algorithm to perform WSD that predicts which sense is most likely for a target word
	 * @param ambiguousSentence: Context of the target word
	 * @param signature: A HashMap that contains signature correspoding to a particular sense
	 * @param lemma: lemma of the target word
     * @return: HashMap with the similarity score for the all the senses of a target word 	 
	*/
	
	public HashMap<String, Double> Lesk(String ambiguousSent, Map<String, Pair<String, Integer>> signature, String lemma, String context_opt) {
		
		/* Get all the senses from the HashMap and store it in an arraylist*/
		Set<String> keySet = signature.keySet();
		ArrayList<String> keySenses = new ArrayList<String>(keySet);
		
		/* Convert the Contect of the target word into an arraylist by calling the function str2bow() */
		ArrayList<String> sent = str2bow(ambiguousSent);
		
		/* Removing the stopwords from the context if context option = ALL_WORDS_R */
		if(context_opt.equals("ALL_WORDS_R")) {
			for(String str : sent) {
				if(stopwords.contains(str)) {
					sent.remove(str);
				}
			}
		}
		
		/* Set the Lesk variables to find the best sense*/
		String best_sense = "";
		int maxOverlap = 0;
		
		/* HashMap to store similarity score of all the senses of a target word */
		HashMap<String, Double> sim = new HashMap<String, Double>();
		
		/* For each sense in keySenses*/
		for(String senses : keySenses) {
			
			/* Get the signature of the corresponding sense and convert it into an arraylist */
			ArrayList<String> signat = str2bow(signature.get(senses).getKey());
			
			/* Removing the stopwords from the signature if context option = ALL_WORDS_R */
			if(context_opt.equals("ALL_WORDS_R")) {
				for(String str : signat) {
					if(stopwords.contains(str)) {
						signat.remove(str);
					}
				}
			}
			
			/* Compute the number of overlapping words between the context and the signature by calling the function Overlap() */
			int overlap = Overlap(sent, signat);
			
			/* Calculate the best sense*/
			if(overlap > maxOverlap) {
				maxOverlap = overlap;
				best_sense = senses;
			}
			
			/* Get the JACCARD similarity between the context and the signature*/
			double sim_score = similarity(sent, signat, "JACCARD");
			sim.put(senses, sim_score);	
			
		}
		//System.out.println(best_sense);
		return sim;
	}
	
	/* Computes the number of overapping words between the context and the signature for a particular sense*/
	public int Overlap(ArrayList<String> c, ArrayList<String> s) {
		Set<String> common = new HashSet<>(c);
		common.retainAll(new HashSet<>(s));
		return common.size();
	}

	/**
	 * Multiple senses are concatenated using comma ",". Separate them out.
	 * @param senses
	 * @return
	 */
	private ArrayList<String> parseSenseKeys(String senseStr) {
		ArrayList<String> senses = new ArrayList<String>();
		String[] items = senseStr.split(",");
		for (String item : items) {
			senses.add(item);
		}
		return senses;
	}
	
	/**
	 * Precision/Recall/F1-score at top K positions
	 * @param groundTruths: a list of sense id strings, such as [become%2:30:00::, become%2:42:01::]
	 * @param predictions: a map from sense id strings to the predicted similarity
	 * @param K
	 * @return a list of [top K precision, top K recall, top K F1]
	 */
	private ArrayList<Double> evaluate(ArrayList<String> groundT, HashMap<String, Double> predict, int K) {
		
		/* Variable to store the Precision, recall and F1 score */
		ArrayList<Double> scores = new ArrayList<Double>();
		
		/* Get the senses of the target word*/
		Set<String> keys = predict.keySet();
		ArrayList<String> senses = new ArrayList<String>(keys);
		
		/* Calculate the number of correctly predicted senses with highest K Similarity*/
		int correctly_pred_sen = 0;
		for(String str : senses) {
			if(groundT.contains(str)) {
				correctly_pred_sen++;
			}
		}
		
		/* Calculate the Precision, Recall and F1 Scores */
		double precision = (double)correctly_pred_sen / K;
		
		double recall = 0.0;
		if(groundT.size()!= 0){
			recall = (double)correctly_pred_sen / groundT.size();
		}
		else{
			recall = 0.0;
		}
		
		double f1 = 0.0;
		if(recall == 0 && precision == 0) {
			f1 = 0;
		}
		else{
			f1 = (2 * recall * precision) / (recall + precision);
		}
		
		/* Add the Scores to the ArrayList */
		scores.add(precision);
		scores.add(recall);
		scores.add(f1);
		
		return scores;
	}
	
	/**
	 * Test the prediction performance on all test sentences
	 * @param K Top-K precision/recall/f1
	 */
	//public ArrayList<Double> evaluate(int K) {
	//}
	
	/**
     * Calculates the senses with highest K similarity
	 * @param key: a list of all senses for a target word
     * @param val: corresponding similarity score for each sense
     * @return kSim: K senses with highest similarity	 
	*/
	public HashMap<String, Double> kPositions(ArrayList<String> key, ArrayList<Double> val) {
		
		/* Get the sense with the higest similarity*/
		double highest_sim = Collections.max(val);
		
		/* HashMap to store K senses with highest similarity*/
		HashMap<String, Double> kSim = new HashMap<String, Double>();
		
		/* Check if any other sense has a similairty equal to the highest similarity and add it to the HashMap*/
		for(int i = 0; i < val.size(); i++) {
			if(val.get(i) == highest_sim) {
				kSim.put(key.get(i), val.get(i));
			}
		}
		
		return kSim;
	}
	/**
	 * @param args[0] file name of a test corpus
	 */
	public static void main(String[] args)throws IOException {
		
		/* Open the Output File to store the results */
		BufferedWriter br = new BufferedWriter(new FileWriter("results/metrices.txt"));
		
		String context_opt = "ALL_WORDS";
		int window_size = 3;
		String sim_opt = "JACCARD";
		
		/* Open the file directory that contains all the test files */
		File directory = new File("data/semcor_txt");
		File[] listOfFiles = directory.listFiles();
		
		/* For each test file */
		for (File file : listOfFiles){
			
			/* Create an object of the class*/
			Lesk model = new Lesk();
			
			/* Call the function readTestData for each test file */
			model.readTestData("data/semcor_txt/"+file.getName());		
			
			/* Call the predict function */
			model.predict(context_opt, window_size, sim_opt);
			
			/* ArrayList the stores the Precision, Recall and F1 score over all target words*/
			ArrayList<Double> metrices = new ArrayList<Double>();
			
			/* Variables to store the average of the 3 metrices over all targets*/
			double avg_prec = 0.0;
			double avg_reca = 0.0;
			double avg_f1 = 0.0;
		
			/* For each sentence*/
			for(int i = 0; i < model.predictions.size(); i++) {
				
				/* for each target word*/
				for(int j = 0; j < model.predictions.get(i).size(); j++) {
					
					/* get all the senses of the target word*/
					Set<String> key = model.predictions.get(i).get(j).keySet();
					ArrayList<String> listofkeys = new ArrayList<String>(key);
					
					/* Get the similarity scores for each sense*/
					ArrayList<Double> values = new ArrayList<Double>();
					for(String keys : listofkeys) {
						values.add(model.predictions.get(i).get(j).get(keys));
					}
				
					/* Get the senses with the highest K similarity*/
					HashMap<String, Double> kpos = model.kPositions(listofkeys, values);
					
					int k = kpos.size();
					listofkeys.retainAll(model.groundTruths.get(i));
					
					/* Get the scores of a target word */
					ArrayList<Double> temp = (model.evaluate(listofkeys, kpos, k));
					
					/* Add the scores to the arrayList */
					metrices.add(temp.get(0));
					metrices.add(temp.get(1));
					metrices.add(temp.get(2));
				}
			}
			
			/* Compute the average of the 3 metrices over all targets*/
			for(int i = 0; i < metrices.size(); i++) {
				if(i % 3 == 0) { // Precision score
					avg_prec = avg_prec + metrices.get(i);
				}
				if(i % 3 == 1) { // Recall Score
					avg_reca = avg_reca + metrices.get(i);
				}
				if(i % 3 == 2) { // F1 score
					avg_f1 = avg_f1 + metrices.get(i);
				}
			}
			
			/* Average Scores*/
			avg_prec = (avg_prec / (metrices.size()/3));
			avg_reca = (avg_reca / (metrices.size()/3));
			avg_f1 = (avg_f1 / (metrices.size()/3));
			
			/* Write the result to the file metrices.txt */
			br.write("data/semcor_txt/"+file.getName() + "\t" + avg_prec+"" + "\t" + avg_reca+"" + "\t" + avg_f1+"" + "\n");
		}
		br.close();
		/*ArrayList<Double> res = model.evaluate(1);
		System.out.print(args[0]);
		System.out.print("\t");
		System.out.print(res.get(0));
		System.out.print("\t");
		System.out.print(res.get(1));
		System.out.print("\t");
		System.out.println(res.get(2));*/
		
		//System.out.println(model.testCorpus);
		//System.out.println(model.ambiguousLocations);
		//System.out.println(model.ambiguousWords);
		//System.out.println(model.groundTruths);
	}
}
