package by.bsuir.kslisenko;

import java.io.IOException;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.clustering.kmeans.Cluster;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.vectorizer.collocations.llr.Gram;

import bbuzz2011.stackoverflow.preprocess.xml.PostWritable;
import by.bsuir.kslisenko.util.ReaderHandler;
import by.bsuir.kslisenko.util.SequenceFileReaderUtil;
import by.bsuir.kslisenko.util.handler.ConsoleReaderHandler;
import by.bsuir.kslisenko.util.handler.SimpleConsoleReaderHandler;
import by.bsuir.kslisenko.util.handler.TextFileOutputReaderHandler;

/**
 * Preparing and clustering data process produces several binary sequence files.
 * This utility reads and output this files data to console for better understanding what's going on.
 * 
 * @author cloudera
 */
public class MySequenceFileReader {

	private static final String CLUSTERED_POINTS_PATH = "target/stackoverflow-output-base/kmeans/clusteredPoints";

	private static final String INITIAL_CLUSTER_PATH = "target/stackoverflow-kmeans-initial-clusters";

	private static final int RECORDS_TO_OUT_TEXT_FILE = 1000000; //0

	private static final int DOCUMENTS_COUNT = 10;
	
	private static final String SUBGRAMS_TXT_PATH = "target/stackoverflow-output-base/sparse/wordcount/subgrams/subgrams.txt";
	private static final String SUBGRAMS_PATH = "target/stackoverflow-output-base/sparse/wordcount/subgrams";
	private static final String NGRAMS_PATH = "target/stackoverflow-output-base/sparse/wordcount/ngrams";
	private static final String FREQUENCY_FILE_PATH = "target/stackoverflow-output-base/sparse/frequency.file-0";
	private static final String DICTIONARY_FILE_TXT_PATH = "target/stackoverflow-output-base/sparse/dictionary.file-0.txt";
	private static final String TF_IDF_VECTORS_PATH = "target/stackoverflow-output-base/sparse/tfidf-vectors";
	private static final String TF_VECTORS_PATH = "target/stackoverflow-output-base/sparse/tf-vectors";
	
	private static final String TOKENIZED_DOCUMENTS_PATH = "target/stackoverflow-output-base/sparse/tokenized-documents";
	private static final String DICTIONARY_PATH = "target/stackoverflow-output-base/sparse/dictionary.file-0";
	private static final String POSTS_WRITABLE_PATH = "target/stackoverflow-output-base/posts";
	private static final String POSTS_TO_TEXT_PATH = "target/stackoverflow-output-base/posts-text";
	private static final String FINAL_CLUSTERS_PATH = "target/stackoverflow-output-base/kmeans/clusters-2-final/part-r-00000";
	private static final String FINAL_CLUSTERS_PATH2 = "target/stackoverflow-output-base/kmeans/clusters-2-final";
	private static final String CLUSTERED_POSTS_PATH = "target/stackoverflow-output-base/clusteredPosts";

	public static Map<Integer, String> dictionary;
	
	public static void main(String[] args) throws IOException {
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(conf);
		
		// 1. Preprocess data
		
		// 1.1 StackOverflowPostXMLParserJob
		// posts.xml -> sequence file [PostId] [PostWritable(Title Text)]
		readProcessedPostsToPostWritable(conf);
		
		// 1.2 StackOverflowPostTextExtracterJob 
		// sequence file [PostId] [PostWritable(Title Text)] -> sequence file [PostId] [Title+Text]
		readProcessedPosts(conf);
		
		// 2. Vectorize data

		// 2.1 Read tokenized documents
		// For document 
		// "What is the best way to micro-adjust a lens? I have a Canon 7D with a 50mm f/1.4 lens, and I think the auto-focus of the lens is off. How can I test and adjust this reliably?"
		// Result would be [micro, adjust, lens, canon, lens, think, auto, focus, lens, test, adjust, reliably, will, approach, work, lenses, different, camera, body, other, different, options]
		// TODO (check) So first we tokenize documents, then prepare dictionary, then do vectorizing
		readTokenizedDocuments(conf);
		
		// 2.2 Output generated dictionary from tokenized documents
		dictionary = readDictionary(conf);

		// 2.3 Output frequency file
		// TODO what does this files represent?
		// TODO at which stage does this file created?
		readFrequencyFile(conf);
		readDfCount(conf);		
		
		// 2.4 Output generated vectors
		readTfVectors(conf, dictionary);
		readTfIdfVectors(conf, dictionary);
		
		// 2.5 Read ngrams and subgrams
		readNGrams(conf);
		readSubgrams(conf);
		
		// 3. Cluster data		
		
		// Clustering
		// 3.1 Read initial clusters
		readInitialClusters(conf);
		
		readClusteredPoints(conf);
		readFinalClusters(conf);
	}

	private static void readFinalClusters(Configuration conf) throws IOException {
		readClusters("target/stackoverflow-output-base/kmeans/clusters-2-final", conf);
	}	
	
	// TODO I really do not much understand what are this clustered points. Is it a full output?
	private static void readClusteredPoints(Configuration conf) throws IOException {
		SequenceFileReaderUtil.readPartFilesInDirToConsole(CLUSTERED_POINTS_PATH, 10, conf);
		
		// Write to text file
		SequenceFileReaderUtil.readPartFilesInDir(CLUSTERED_POINTS_PATH, RECORDS_TO_OUT_TEXT_FILE, conf, new TextFileOutputReaderHandler<IntWritable, WeightedVectorWritable>(CLUSTERED_POINTS_PATH + "/points.txt"));
	}

	// TODO May be there are clusters with words and their values
	// TODO read this file using clusterdump for better interpreting results
	private static void readInitialClusters(Configuration conf) throws IOException {
		readClusters(INITIAL_CLUSTER_PATH, conf);
	}

	private static void readClusters(String path, Configuration conf) throws IOException {
		ReaderHandler<Text, Cluster> handler = new ReaderHandler<Text, Cluster>() {
			@Override
			public void before() throws IOException {
			}

			@Override
			public void read(Text key, Cluster value, PrintStream out) throws IOException {
				out.println("Cluster id: " + key);
				out.println("Num points: " + value.getNumPoints());
				out.println("Count: " + value.count());
				out.println("Centroid: " + printVectorWithDictionary(value.computeCentroid()));
				out.println("");
			}

			@Override
			public void after() throws IOException {
			}
		};
		SequenceFileReaderUtil.readPartFilesInDir(path, 10, conf, new ConsoleReaderHandler<Text, Cluster>(handler));
		
		SequenceFileReaderUtil.readPartFilesInDir(path, RECORDS_TO_OUT_TEXT_FILE, conf, new TextFileOutputReaderHandler<Text, Cluster>(path + ".txt", handler));
	}

	// TODO are there any 2-grams in this file?
	private static void readNGrams(Configuration conf) throws IOException {
		SequenceFileReaderUtil.readPartFilesInDirToConsole(NGRAMS_PATH, 20, conf);
		SequenceFileReaderUtil.readPartFilesInDir(NGRAMS_PATH, RECORDS_TO_OUT_TEXT_FILE, conf, new TextFileOutputReaderHandler(NGRAMS_PATH + "/ngrams.txt"));
	}
	
	// This is a some statistics for n-grams with weight of each world in n-gram
	private static void readSubgrams(Configuration conf) throws IOException {
		ReaderHandler<Gram, Gram> gramHandler = new ReaderHandler<Gram, Gram>() {
			@Override
			public void before() throws IOException {
			}

			@Override
			public void read(Gram key, Gram value, PrintStream out) throws IOException {
				out.println("gram: " + outGram(key) + "\t has gram: " + outGram(value));
			}

			@Override
			public void after() throws IOException {
			}
			
			public String outGram(Gram gram) {
				return gram.getString() + " [frq=" + gram.getFrequency() + ", type=" + gram.getType() + "]";
			}
		};
		
		// Output first 20 grams to console
		SequenceFileReaderUtil.readPartFilesInDir(SUBGRAMS_PATH, 20, conf, new ConsoleReaderHandler<Gram, Gram>(gramHandler));
		
		// Print all grams to file
		SequenceFileReaderUtil.readPartFilesInDir(SUBGRAMS_PATH, RECORDS_TO_OUT_TEXT_FILE, conf, new TextFileOutputReaderHandler<Gram, Gram>(SUBGRAMS_TXT_PATH, gramHandler));		
	}	

	// TODO I do not understand why we need this file?
	private static void readDfCount(Configuration conf) throws IOException {
		SequenceFileReaderUtil.readPartFilesInDirToConsole("target/stackoverflow-output-base/sparse/df-count", 20, conf);
	}

	private static void readTokenizedDocuments(Configuration conf) throws IOException {
		SequenceFileReaderUtil.readPartFilesInDirToConsole(TOKENIZED_DOCUMENTS_PATH, DOCUMENTS_COUNT, conf);
	}

	// TODO which information does this frequency file contains?
	private static void readFrequencyFile(Configuration conf) throws IOException {
		SequenceFileReaderUtil.readPartFileToConsole(FREQUENCY_FILE_PATH, 20, conf);
	}

	private static Map<Integer, String> readDictionary(Configuration conf) throws IOException {
		SequenceFileReaderUtil.readPartFileToConsole(DICTIONARY_PATH, 20, conf);	
		
		// Output dictionary to text file
		ReaderHandler<Text, IntWritable> dictionaryToTextFileHandler = new TextFileOutputReaderHandler<Text, IntWritable>(DICTIONARY_FILE_TXT_PATH);
		SequenceFileReaderUtil.readPartFile(DICTIONARY_PATH, RECORDS_TO_OUT_TEXT_FILE, conf, dictionaryToTextFileHandler);
		
		// Output dictionary to Map for vectors visualizing
		final Map<Integer, String> dictionary = new HashMap<Integer, String>();
		ReaderHandler<Text, IntWritable> dictionaryToMapHandler = new ReaderHandler<Text, IntWritable>() {
			@Override
			public void before() throws IOException {
			}
			
			@Override
			public void read(Text key, IntWritable value, PrintStream myout) throws IOException {
				dictionary.put(Integer.parseInt(value.toString()), key.toString());
			}

			@Override
			public void after() throws IOException {
			}
		};
		
		SequenceFileReaderUtil.readPartFile(DICTIONARY_PATH, RECORDS_TO_OUT_TEXT_FILE, conf, dictionaryToMapHandler);
		return dictionary;
	}	
	
	static SimpleConsoleReaderHandler<Text, VectorWritable> vectorHandler = new SimpleConsoleReaderHandler<Text, VectorWritable>() {
		@Override
		public void read(Text key, VectorWritable value, PrintStream myout) {
			System.out.println("Key: " + key);
			System.out.println("Vector: " + value.get().asFormatString());
			System.out.println("Vector + dictionary: " + printVectorWithDictionary(value.get()));
			// Here is possible to iterate vector elements
			// TODO What does the key represent?
			// TODO what dows element values represent?
		}
	};
	
	public static String printVectorWithDictionary(Vector vector) {
		StringBuilder result = new StringBuilder();
		for (Element element: vector) {
			if (element.get() > 0) {
				result.append(dictionary.get(element.index()) + "[" + element.index() + "]:" + element.get() + ",");
			}
		}
		return result.toString();
	}
	
	private static void readTfVectors(Configuration conf, Map<Integer, String> dictionary) throws IOException {
		SequenceFileReaderUtil.readPartFilesInDir(TF_VECTORS_PATH, DOCUMENTS_COUNT, conf, vectorHandler);
	}	
	
	private static void readTfIdfVectors(Configuration conf, Map<Integer, String> dictionary) throws IOException {
		SequenceFileReaderUtil.readPartFilesInDir(TF_IDF_VECTORS_PATH, DOCUMENTS_COUNT, conf, vectorHandler);
	}


	private static void readProcessedPostsToPostWritable(Configuration conf) throws IOException {
		SimpleConsoleReaderHandler<LongWritable, PostWritable> handler = new SimpleConsoleReaderHandler<LongWritable, PostWritable>() {
			@Override
			public void read(LongWritable key, PostWritable value, PrintStream myout) {
				System.out.println("Post key: " + key);
				System.out.println("Post title: " + value.getTitle());
				System.out.println("Post content: " + value.getContent());
			}
		};
		
		SequenceFileReaderUtil.readPartFilesInDir(POSTS_WRITABLE_PATH, DOCUMENTS_COUNT, conf, handler);
	}
	
	private static void readProcessedPosts(Configuration conf) throws IOException {
		SimpleConsoleReaderHandler<Text, Text> handler = new SimpleConsoleReaderHandler<Text, Text>() {
			@Override
			public void read(Text key, Text value, PrintStream myout) {
				System.out.println("Post key: " + key);
				System.out.println("Post value (title+text): " + value);
			}
		};
		
		SequenceFileReaderUtil.readPartFilesInDir(POSTS_TO_TEXT_PATH, DOCUMENTS_COUNT, conf, handler);
	}	
}