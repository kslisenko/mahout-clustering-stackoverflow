package by.bsuir.kslisenko.util;

import java.io.IOException;

import org.apache.hadoop.io.Writable;

public class ReaderHandler<K extends Writable, V extends Writable> {
	
	public void before() throws IOException {
	}
	
	public void read(K key, V value) throws IOException {
		System.out.println(key + "\t" + value);
	}
	
	public void after() throws IOException {
	}	
}
