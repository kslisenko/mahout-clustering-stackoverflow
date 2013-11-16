package by.bsuir.kslisenko;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.clustering.kmeans.Cluster;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.vectorizer.collocations.llr.Gram;

import bbuzz2011.stackoverflow.preprocess.xml.PostWritable;
import by.bsuir.kslisenko.util.ReaderHandler;
import by.bsuir.kslisenko.util.SequenceFileReaderUtil;

/**
 * Preparing and clustering data process produces several binary sequence files.
 * This utility reads and output this files data to console for better understanding what's going on.
 * 
 * @author cloudera
 */
public class MySequenceFileReader {

	private static final String TF_IDF_VECTORS_PATH = "target/stackoverflow-output-base/sparse/tfidf-vectors";
	private static final String TF_VECTORS_PATH = "target/stackoverflow-output-base/sparse/tf-vectors";
	private static final int DOCUMENTS_COUNT = 10;
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
		
		// 2.3 Output generated vectors
		readTfVectors(conf, dictionary);
		readTfIdfVectors(conf, dictionary);
		
//		// 3.1 Output frequency file
//		readFrequencyFile(conf, fs);
		
		
//		
//		readDfCount(conf, fs);
//		
//		// 4. Read ngrams and subgrams
//		readNGrams(conf, fs);
//		
//		readSubgrams(conf, fs);
//		
//		// Clustering
//		// 5. Read initial clusters
//		readInitialClusters(conf, fs);
//		
//		readClusteredPoints(conf, fs);
//		readFinalClusters(conf, fs);
//		
//		readPartFiles(FINAL_CLUSTERS_PATH, 2, conf, Text.class, Cluster.class);
//		
//		// 6. Pring clustered posts
//		readPartFiles(CLUSTERED_POSTS_PATH, 2, conf, LongWritable.class, ClusteredDocument.class);
	}

	private static void readFinalClusters(Configuration conf, FileSystem fs) throws IOException {
		System.out.println("target/stackoverflow-output-base/kmeans/clusters-2-final/part-r-00000\n");
		SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(FINAL_CLUSTERS_PATH),conf);
		
		Cluster dicKey = new Cluster();
		Text text = new Text();
        int numOut = 0;
        while (reader.next(text, dicKey)) {
        	if (numOut++ >= 2) break;
        	System.out.println(text.toString() + "\t" + dicKey.asFormatString());
        }
        reader.close();			
	}	
	
	// TODO I really do not much understand what are this clustered points. Is it a full output?
	private static void readClusteredPoints(Configuration conf, FileSystem fs) throws IOException {
		System.out.println("target/stackoverflow-output-base/kmeans/clusteredPoints/part-m-0\n");
		SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path("target/stackoverflow-output-base/kmeans/clusteredPoints/part-m-0"),conf);
		
		WeightedVectorWritable dicKey = new WeightedVectorWritable();
		IntWritable text = new IntWritable();
        int numOut = 0;
        while (reader.next(text, dicKey)) {
        	if (numOut++ >= 10) break;
        	System.out.println(text.toString() + "\t" + dicKey.toString());
        }
        reader.close();			
	}

	// May be there are clusters with words and their values
	// TODO read this file using clusterdump for better interpreting results
	private static void readInitialClusters(Configuration conf, FileSystem fs) throws IOException {
		System.out.println("target/stackoverflow-kmeans-initial-clusters/part-randomSeed\n");
		SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path("target/stackoverflow-kmeans-initial-clusters/part-randomSeed"),conf);
		
		Cluster dicKey = new Cluster();
		Text text = new Text();
        int numOut = 0;
        while (reader.next(text, dicKey)) {
        	if (numOut++ >= 10) break;
        	System.out.println(text.toString() + "\t" + dicKey.asFormatString());
        }
        reader.close();			
	}

	// TODO are there any 2-grams in this file?
	private static void readNGrams(Configuration conf, FileSystem fs) throws IOException {
		System.out.println("target/stackoverflow-output-base/sparse/wordcount/ngrams/part-r-00000\n");
		SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path("target/stackoverflow-output-base/sparse/wordcount/ngrams/part-r-00000"),conf);
		
		DoubleWritable dicKey = new DoubleWritable();
		Text text = new Text();
        int numOut = 0;
        while (reader.next(text, dicKey)) {
        	if (numOut++ >= 10) break;
        	System.out.println(text.toString() + "\t" + dicKey.toString());
        }
        reader.close();	
	}
	
	// This is a some statistics for n-grams with weight of each world in n-gram
	private static void readSubgrams(Configuration conf, FileSystem fs) throws IOException {
		System.out.println("target/stackoverflow-output-base/sparse/wordcount/subgrams/part-r-00000\n");
		SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path("target/stackoverflow-output-base/sparse/wordcount/subgrams/part-r-00000"),conf);
		
		Gram dicKey = new Gram();
		Gram text = new Gram();
        int numOut = 0;
        while (reader.next(text, dicKey)) {
        	if (numOut++ >= 10) break;
        	System.out.println(text.toString() + "\t" + dicKey.toString());
        }
        reader.close();	
	}	

	// TODO I do not understand why we need this file?
	private static void readDfCount(Configuration conf, FileSystem fs) throws IOException {
		System.out.println("target/stackoverflow-output-base/sparse/df-count/part-r-00000\n");
		SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path("target/stackoverflow-output-base/sparse/df-count/part-r-00000"),conf);
		
		LongWritable dicKey = new LongWritable();
		IntWritable text = new IntWritable();
        int numOut = 0;
        while (reader.next(text, dicKey)) {
        	if (numOut++ >= 10) break;
        	System.out.println(text.toString() + "\t" + dicKey.toString());
        }
        reader.close();	
	}

	private static void readTokenizedDocuments(Configuration conf) throws IOException {
		SequenceFileReaderUtil.readPartFilesInDir(TOKENIZED_DOCUMENTS_PATH, DOCUMENTS_COUNT, conf);
	}

	// TODO which information does this frequency file contains?
	private static void readFrequencyFile(Configuration conf, FileSystem fs) throws IOException {
		System.out.println("target/stackoverflow-output-base/sparse/frequency.file-0\n");
		SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path("target/stackoverflow-output-base/sparse/frequency.file-0"),conf);
		
		LongWritable dicKey = new LongWritable();
		IntWritable text = new IntWritable();
        int numOut = 0;
        while (reader.next(text, dicKey)) {
        	if (numOut++ >= 10) break;
        	System.out.println(text.toString() + "\t" + dicKey.toString());
        }
        reader.close();	
	}

	private static Map<Integer, String> readDictionary(Configuration conf) throws IOException {
		SequenceFileReaderUtil.readPartFile(DICTIONARY_PATH, 20, conf);	
		
		// Output dictionary to text file
		ReaderHandler<Text, IntWritable> dictionaryToTextFileHandler = new ReaderHandler<Text, IntWritable>() {
			
			File dictionaryTextFile = new File("target/stackoverflow-output-base/sparse/dictionary.file-0.txt");
			FileWriter out = new FileWriter(dictionaryTextFile);
			
			@Override
			public void read(Text key, IntWritable value) throws IOException {
				out.write(key + " " + value + "\n");
			}
			
			@Override
			public void after() throws IOException {
				out.flush();
			}			
		};
		
		SequenceFileReaderUtil.readPartFile(DICTIONARY_PATH, 1000000, conf, dictionaryToTextFileHandler);
		
		// Output dictionary to Map
		final Map<Integer, String> dictionary = new HashMap<Integer, String>();
		ReaderHandler<Text, IntWritable> dictionaryToMapHandler = new ReaderHandler<Text, IntWritable>() {
			
			@Override
			public void read(Text key, IntWritable value) throws IOException {
				dictionary.put(Integer.parseInt(value.toString()), key.toString());
			}
		};
		
		SequenceFileReaderUtil.readPartFile(DICTIONARY_PATH, 1000000, conf, dictionaryToMapHandler);
		return dictionary;
	}	
	
	static ReaderHandler<Text, VectorWritable> vectorHandler = new ReaderHandler<Text, VectorWritable>() {
		@Override
		public void read(Text key, VectorWritable value) {
			System.out.println("Key: " + key);
			System.out.println("Vector: " + value.get().asFormatString());
			
			StringBuilder result = new StringBuilder();
			for (Element element: value.get()) {
				if (element.get() > 0) {
					result.append(dictionary.get(element.index()) + "[" + element.index() + "]:" + element.get() + ",");
				}
			}
			
			System.out.println("Vector + dictionary: " + result.toString());
			// Here is possible to iterate vector elements
			// TODO What does the key represent?
			// TODO what dows element values represent?
		}
	};
	
	private static void readTfVectors(Configuration conf, Map<Integer, String> dictionary) throws IOException {
		SequenceFileReaderUtil.readPartFilesInDir(TF_VECTORS_PATH, DOCUMENTS_COUNT, conf, vectorHandler);
	}	
	
	private static void readTfIdfVectors(Configuration conf, Map<Integer, String> dictionary) throws IOException {
		SequenceFileReaderUtil.readPartFilesInDir(TF_IDF_VECTORS_PATH, DOCUMENTS_COUNT, conf, vectorHandler);
	}


	private static void readProcessedPostsToPostWritable(Configuration conf) throws IOException {
		ReaderHandler<LongWritable, PostWritable> handler = new ReaderHandler<LongWritable, PostWritable>() {
			@Override
			public void read(LongWritable key, PostWritable value) {
				System.out.println("Post key: " + key);
				System.out.println("Post title: " + value.getTitle());
				System.out.println("Post content: " + value.getContent());
			}
		};
		
		SequenceFileReaderUtil.readPartFilesInDir(POSTS_WRITABLE_PATH, DOCUMENTS_COUNT, conf, handler);
	}
	
	private static void readProcessedPosts(Configuration conf) throws IOException {
		ReaderHandler<Text, Text> handler = new ReaderHandler<Text, Text>() {
			@Override
			public void read(Text key, Text value) {
				System.out.println("Post key: " + key);
				System.out.println("Post value (title+text): " + value);
			}
		};
		
		SequenceFileReaderUtil.readPartFilesInDir(POSTS_TO_TEXT_PATH, DOCUMENTS_COUNT, conf, handler);
	}	
}