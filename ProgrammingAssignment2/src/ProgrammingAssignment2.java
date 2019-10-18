//@authors Josephine Lamp (jl4rj) and Trey Woodlief (adw8dm)

import java.awt.BorderLayout;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

public class ProgrammingAssignment2
{
	
	public static void main(String[] args) throws Exception
	{
		//Feature order: x mean, x std dev, x median, x mean root square, y mean, y std dev, y median, y mean root square, z mean, z std dev, z median, z mean root square, label
    	//int[] features = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}; // all the features, index 12 is
		
		/*
		 * Part 1 - Add more training samples and run Weka without the GUI
		 */
		
////		//Original Data
////		//get features first and output to CSV file
////		//getFeatures("features0.csv", 1);
//		
//        //  first get the full features file
//    	String[][] csvData = readCSV("features0.csv");  
//		int[] features = {0, 1, 4, 5, 8, 9}; // use mean, std dev for x, y, z and label
//        String arffData = csvToArff(csvData, features);
//        System.out.println(arffData);
//        
//        double accuracy = classify(arffData, 1);
//		
//
//        //Compare to larger data size, excluding original handwash data
//        //get features first and output to CSV file
//      	getFeatures("features1.csv", 1);
//      		
//        //  first get the full features file
//      	String[][] csvData1 = readCSV("features1.csv");  
//  		int[] features1 = {0, 1, 4, 5, 8, 9}; // use mean, std dev for x, y, z and label
//        String arffData1 = csvToArff(csvData1, features1);
//        System.out.println(arffData1);
//        double accuracy1 = classify(arffData1, 1);
//        
//        System.out.println("Accuracy of Original Data is "  + accuracy + " \nAccuracy of New Data is " + accuracy1);
        
		/*
		 * Part 2 - Change Sliding window size
		 */

//		//sliding window 1
//        getFeatures("featuresWindow1.csv", 1);
//       	String[][] csvDataWindow1 = readCSV("featuresWindow1.csv");  
//  		int[] featuresWindow1 = {0, 1, 4, 5, 8, 9}; // use mean, std dev for x, y, z and label
//        String arffDataWindow1 = csvToArff(csvDataWindow1, featuresWindow1);
//        double accuracyWindow1 = classify(arffDataWindow1, 1);
//        
//        //sliding window 2
//        getFeatures("featuresWindow2.csv", 2);
//       	String[][] csvDataWindow2 = readCSV("featuresWindow2.csv");  
//  		int[] featuresWindow2 = {0, 1, 4, 5, 8, 9}; // use mean, std dev for x, y, z and label
//        String arffDataWindow2 = csvToArff(csvDataWindow2, featuresWindow2);
//        double accuracyWindow2 = classify(arffDataWindow2, 1);
//        
//        //sliding window 3
//        getFeatures("featuresWindow3.csv", 3);
//       	String[][] csvDataWindow3 = readCSV("featuresWindow3.csv");  
//  		int[] featuresWindow3 = {0, 1, 4, 5, 8, 9}; // use mean, std dev for x, y, z and label
//        String arffDataWindow3 = csvToArff(csvDataWindow3, featuresWindow3);
//        double accuracyWindow3 = classify(arffDataWindow3, 1);
//        
//        //sliding window 4
//        getFeatures("featuresWindow4.csv", 4);
//       	String[][] csvDataWindow4 = readCSV("featuresWindow4.csv");  
//  		int[] featuresWindow4 = {0, 1, 4, 5, 8, 9}; // use mean, std dev for x, y, z and label
//        String arffDataWindow4 = csvToArff(csvDataWindow4, featuresWindow4);
//        double accuracyWindow4 = classify(arffDataWindow4, 1);
//        
//        System.out.println("Accuracy of Sliding Window 1 "  + accuracyWindow1 + " \nAccuracy of Sliding Window 2 " + accuracyWindow2 + " \nAccuracy of Sliding Window 3 " + accuracyWindow3 + " \nAccuracy of Sliding Window 4 " + accuracyWindow4);
//        
        /*
         * Part 3 - Add features using the best sliding window of 3
         */
        
//        getFeatures("featuresAllFeatures.csv", 3);
//       	String[][] csvDataAllFeatures = readCSV("featuresAllFeatures.csv");  
//  		int[] featuresAllFeatures = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
//        String arffAllFeatures = csvToArff(csvDataAllFeatures, featuresAllFeatures);
//        double accuracyAllFeatures = classify(arffAllFeatures, 1);
//        System.out.println("Accuracy of using All Features "  + accuracyAllFeatures);
		
		
		/*
		 * Part 4 - Find best features
		 */
		
		//find best features for decision tree classifer
		findBestFeatures(12, 1);
		
		/*
		 * Part 5 - Compare Classifiers
		 */
		
		//Use Random Forest classifier
		findBestFeatures(12, 2);
		
		//Use SVM classifer
		findBestFeatures(12, 3);
	}
	

    private static int[] findBestFeatures(int numFeatures, int clNumber) throws Exception
    {

    	
//    	find the classification accuracies for each of the features individually
    	String[][] csvDataWindow3 = readCSV("featuresWindow3.csv");
    	for (int i = 0; i < numFeatures; i++) {
			String arffDataWindow3 = csvToArff(csvDataWindow3, new int[] {i});
        	double accuracyWindow3 = classify(arffDataWindow3, clNumber);
    		System.out.println("Accuracy with just feature " + i + " is " + accuracyWindow3);
    	}
    	ArrayList<Integer> features = new ArrayList<>();
    	Set<Integer> left = new HashSet<>();
    	for (int i  =  0; i < 12; i++) {
    		left.add(i);
    	}
    	double lastAcc = -1;
    	for(int i = 0; i < numFeatures; i++)
    	{
        	int best = -1;
        	double accuracy=-1;
    		int[] featureArr = new int[i+1];
    		for (int j=0; j < featureArr.length - 1; j++) {
    			featureArr[j] = features.get(j);	
    		}
    		for (int feature : left) {
    			featureArr[i] = feature;
    			String arffDataWindow3 = csvToArff(csvDataWindow3, featureArr);
            	double accuracyWindow3 = classify(arffDataWindow3, clNumber);
            	if (accuracyWindow3 > accuracy) {
            		best = feature;
            		accuracy = accuracyWindow3;
            	}
    		}
    		features.add(best);
    		Collections.sort(features);
    		left.remove(best);
    		System.out.println("Current feature set: Size: " + (i + 1) + " " + features.toString());
    		System.out.println("Current accuracy: " + accuracy);
    		if (accuracy < lastAcc + 1) {
    			break;
    		}
    		lastAcc = accuracy;
    	}
		int[] featureArr = new int[features.size()];
		for (int j=0; j < featureArr.length; j++) {
			featureArr[j] = features.get(j);	
		}
		return featureArr;
    }
	
	public static void getFeatures(String finalOutputFilename, int slidingWindow)
	{
		try 
		{
			//original dataset
//			extractFeatures("src/CSVData/G6NZCJ00326722F-subject1-left-handwash-none-2019-09-26-13-13-04.csv", "handwash", "src/FeaturesData/1.csv", finalOutputFilename, slidingWindow);
//			extractFeatures("src/CSVData/G6NZCJ00326722F-subject1-left-handwash-none-2019-09-26-13-13-59.csv", "handwash", "src/FeaturesData/2.csv", finalOutputFilename, slidingWindow);
//			extractFeatures("src/CSVData/G6NZCJ00326722F-subject1-left-handwash-none-2019-09-26-13-14-26.csv", "handwash", "src/FeaturesData/3.csv", finalOutputFilename, slidingWindow);
//			extractFeatures("src/CSVData/G6NZCJ00326722F-subject1-left-handwash-none-2019-09-26-13-15-09.csv", "handwash", "src/FeaturesData/4.csv", finalOutputFilename, slidingWindow);
//			extractFeatures("src/CSVData/G6NZCJ00326722F-subject1-left-handwash-none-2019-09-26-13-15-40.csv", "handwash", "src/FeaturesData/5.csv", finalOutputFilename, slidingWindow);
//			extractFeatures("src/CSVData/G6NZCJ00326722F-subject1-left-handwash-none-2019-09-26-13-16-56.csv", "handwash", "src/FeaturesData/6.csv", finalOutputFilename, slidingWindow);
//			extractFeatures("src/CSVData/G6NZCJ00326722F-subject1-left-handwash-none-2019-09-26-13-17-26.csv", "handwash", "src/FeaturesData/7.csv", finalOutputFilename, slidingWindow);
//			extractFeatures("src/CSVData/G6NZCJ00326722F-subject1-left-handwash-none-2019-09-26-13-17-53.csv", "handwash", "src/FeaturesData/8.csv", finalOutputFilename, slidingWindow);
//			extractFeatures("src/CSVData/G6NZCJ00326722F-subject1-left-handwash-none-2019-09-26-13-18-30.csv", "handwash", "src/FeaturesData/9.csv", finalOutputFilename, slidingWindow);
//			extractFeatures("src/CSVData/G6NZCJ00326722F-subject1-left-handwash-none-2019-09-26-13-19-01.csv", "handwash", "src/FeaturesData/10.csv", finalOutputFilename, slidingWindow);
			
			extractFeatures("src/CSVData/G6NZCJ00326722F-subject1-left-nonhandwash-waving-2019-09-26-13-20-07.csv", "nonhandwash", "src/FeaturesData/11.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/CSVData/G6NZCJ00326722F-subject1-left-nonhandwash-walking-2019-09-26-13-20-32.csv", "nonhandwash", "src/FeaturesData/12.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/CSVData/G6NZCJ00326722F-subject1-left-nonhandwash-fakechopping-2019-09-26-13-20-57.csv", "nonhandwash", "src/FeaturesData/13.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/CSVData/G6NZCJ00326722F-subject1-left-nonhandwash-throwing-2019-09-26-13-21-24.csv", "nonhandwash", "src/FeaturesData/14.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/CSVData/G6NZCJ00326722F-subject1-left-nonhandwash-jumpingjacks-2019-09-26-13-21-44.csv", "nonhandwash", "src/FeaturesData/15.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/CSVData/G6NZCJ00326722F-subject1-left-nonhandwash-mixing-2019-09-26-13-22-07.csv", "nonhandwash", "src/FeaturesData/16.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/CSVData/G6NZCJ00326722F-subject1-left-nonhandwash-knocking-2019-09-26-13-22-30.csv", "nonhandwash", "src/FeaturesData/17.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/CSVData/G6NZCJ00326722F-subject1-left-nonhandwash-writing-2019-09-26-13-22-54.csv", "nonhandwash", "src/FeaturesData/18.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/CSVData/G6NZCJ00326722F-subject1-left-nonhandwash-handshake-2019-09-26-13-25-33.csv", "nonhandwash", "src/FeaturesData/19.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/CSVData/G6NZCJ00326722F-subject1-left-nonhandwash-running-2019-09-26-13-25-54.csv", "nonhandwash", "src/FeaturesData/20.csv", finalOutputFilename, slidingWindow);

//			extractFeatures("src/CSVData/G6NZCJ00326722F-subject2-left-handwash-none-2019-09-26-13-27-41.csv", "handwash", "src/FeaturesData/21.csv", finalOutputFilename, slidingWindow);
//			extractFeatures("src/CSVData/G6NZCJ00326722F-subject2-left-handwash-none-2019-09-26-13-28-01.csv", "handwash", "src/FeaturesData/22.csv", finalOutputFilename, slidingWindow);
//			extractFeatures("src/CSVData/G6NZCJ00326722F-subject2-left-handwash-none-2019-09-26-13-28-26.csv", "handwash", "src/FeaturesData/23.csv", finalOutputFilename, slidingWindow);
//			extractFeatures("src/CSVData/G6NZCJ00326722F-subject2-left-handwash-none2019-09-26-13-29-29.csv", "handwash", "src/FeaturesData/24.csv", finalOutputFilename, slidingWindow);
//			extractFeatures("src/CSVData/G6NZCJ00326722F-subject2-left-handwash-none-2019-09-26-13-29-47.csv", "handwash", "src/FeaturesData/25.csv", finalOutputFilename, slidingWindow);
//			extractFeatures("src/CSVData/G6NZCJ00326722F-subject2-left-handwash-none-2019-09-26-13-29-58.csv", "handwash", "src/FeaturesData/26.csv", finalOutputFilename, slidingWindow);
//			extractFeatures("src/CSVData/G6NZCJ00326722F-subject2-left-handwash-none-2019-09-26-13-30-17.csv", "handwash", "src/FeaturesData/27.csv", finalOutputFilename, slidingWindow);
//			extractFeatures("src/CSVData/G6NZCJ00326722F-subject2-left-handwash-none-2019-09-26-13-30-32.csv", "handwash", "src/FeaturesData/28.csv", finalOutputFilename, slidingWindow);
//			extractFeatures("src/CSVData/G6NZCJ00326722F-subject2-left-handwash-none-2019-09-26-13-30-43.csv", "handwash", "src/FeaturesData/29.csv", finalOutputFilename, slidingWindow);
//			extractFeatures("src/CSVData/G6NZCJ00326722F-subject2-left-handwash-none-2019-09-26-13-30-52.csv", "handwash", "src/FeaturesData/30.csv", finalOutputFilename, slidingWindow);

			extractFeatures("src/CSVData/G6NZCJ00326722F-subject2-left-nonhandwash-waving-2019-09-26-13-32-15.csv", "nonhandwash", "src/FeaturesData/31.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/CSVData/G6NZCJ00326722F-subject2-left-nonhandwash-walking-2019-09-26-13-32-38.csv", "nonhandwash", "src/FeaturesData/32.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/CSVData/G6NZCJ00326722F-subject2-left-nonhandwash-fakechopping-2019-09-26-13-33-00.csv", "nonhandwash", "src/FeaturesData/33.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/CSVData/G6NZCJ00326722F-subject2-left-nonhandwash-throwing-2019-09-26-13-33-24.csv", "nonhandwash", "src/FeaturesData/34.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/CSVData/G6NZCJ00326722F-subject2-left-nonhandwash-jumpingjacks-2019-09-26-13-33-34.csv", "nonhandwash", "src/FeaturesData/35.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/CSVData/G6NZCJ00326722F-subject2-left-nonhandwash-mixing-2019-09-26-13-33-51.csv", "nonhandwash", "src/FeaturesData/36.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/CSVData/G6NZCJ00326722F-subject2-left-nonhandwash-knocking-2019-09-26-13-34-11.csv", "nonhandwash", "src/FeaturesData/37.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/CSVData/G6NZCJ00326722F-subject2-left-nonhandwash-writing-2019-09-26-13-34-36.csv", "nonhandwash", "src/FeaturesData/38.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/CSVData/G6NZCJ00326722F-subject2-left-nonhandwash-handshake-2019-09-26-13-35-02.csv", "nonhandwash", "src/FeaturesData/39.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/CSVData/G6NZCJ00326722F-subject2-left-nonhandwash-running-2019-09-26-13-35-15.csv", "nonhandwash", "src/FeaturesData/40.csv", finalOutputFilename, slidingWindow);
		
			//extract new data
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject1-left-activity1-info1-2019-10-18-10-12-36.csv", "nonhandwash", "src/FeaturesData/41.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject1-left-activity1-info2-2019-10-18-10-13-11.csv", "nonhandwash", "src/FeaturesData/42.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject1-left-activity1-info3-2019-10-18-10-14-28.csv", "nonhandwash", "src/FeaturesData/43.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject1-left-activity1-info4-2019-10-18-10-15-16.csv", "nonhandwash", "src/FeaturesData/44.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject1-left-activity2-info1-2019-10-18-10-17-33.csv", "nonhandwash", "src/FeaturesData/45.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject1-left-activity2-info2-2019-10-18-10-18-49.csv", "nonhandwash", "src/FeaturesData/46.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject1-left-activity2-info3-2019-10-18-10-19-53.csv", "nonhandwash", "src/FeaturesData/47.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject1-left-activity2-info4-2019-10-18-10-21-09.csv", "nonhandwash", "src/FeaturesData/48.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject1-left-activity2-info4-2019-10-18-10-21-34.csv", "nonhandwash", "src/FeaturesData/49.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject1-left-activity2-none-2019-10-18-10-16-47.csv", "nonhandwash", "src/FeaturesData/50.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject1-left-activity3-info1-2019-10-18-10-23-28.csv", "handwash", "src/FeaturesData/51.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject1-left-activity3-info2-2019-10-18-10-23-53.csv", "handwash", "src/FeaturesData/52.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject1-left-activity3-info3-2019-10-18-10-24-21.csv", "handwash", "src/FeaturesData/53.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject1-left-activity3-info4-2019-10-18-10-24-53.csv", "handwash", "src/FeaturesData/54.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject1-left-activity3-none-2019-10-18-10-22-52.csv", "handwash", "src/FeaturesData/55.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject1-left-activity4-info1-2019-10-18-10-26-27.csv", "handwash", "src/FeaturesData/56.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject1-left-activity4-info2-2019-10-18-10-26-56.csv", "handwash", "src/FeaturesData/57.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject1-left-activity4-info3-2019-10-18-10-28-12.csv", "handwash", "src/FeaturesData/58.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject1-left-activity4-info4-2019-10-18-10-28-34.csv", "handwash", "src/FeaturesData/59.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject1-left-activity4-none-2019-10-18-10-25-59.csv", "handwash", "src/FeaturesData/60.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject2-left-activity1-none-2019-10-18-10-30-14.csv", "nonhandwash", "src/FeaturesData/61.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject2-left-activity1-none-2019-10-18-10-30-31.csv", "nonhandwash", "src/FeaturesData/62.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject2-left-activity1-none-2019-10-18-10-30-50.csv", "nonhandwash", "src/FeaturesData/63.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject2-left-activity1-none-2019-10-18-10-31-19.csv", "nonhandwash", "src/FeaturesData/64.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject2-left-activity1-none-2019-10-18-10-31-43.csv", "nonhandwash", "src/FeaturesData/65.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject2-left-activity1-none-2019-10-18-10-32-06.csv", "nonhandwash", "src/FeaturesData/66.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject2-left-activity1-none-2019-10-18-10-32-42.csv", "nonhandwash", "src/FeaturesData/67.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject2-left-activity1-none-2019-10-18-10-33-35.csv", "nonhandwash", "src/FeaturesData/68.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject2-left-activity1-none-2019-10-18-10-33-56.csv", "nonhandwash", "src/FeaturesData/69.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject2-left-activity1-none-2019-10-18-10-34-40.csv", "nonhandwash", "src/FeaturesData/70.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-10-36-23.csv", "handwash", "src/FeaturesData/71.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-10-36-40.csv", "handwash", "src/FeaturesData/72.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-10-36-55.csv", "handwash", "src/FeaturesData/73.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-10-37-11.csv", "handwash", "src/FeaturesData/74.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-10-37-25.csv", "handwash", "src/FeaturesData/75.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-10-37-39.csv", "handwash", "src/FeaturesData/76.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-10-37-54.csv", "handwash", "src/FeaturesData/77.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-10-38-08.csv", "handwash", "src/FeaturesData/78.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-10-38-22.csv", "handwash", "src/FeaturesData/79.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-10-38-35.csv", "handwash", "src/FeaturesData/80.csv", finalOutputFilename, slidingWindow);
			
			
			//New handwash data
			extractFeatures("src/NewNewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-11-26-58.csv", "handwash", "src/FeaturesData/81.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewNewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-11-27-11.csv", "handwash", "src/FeaturesData/82.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewNewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-11-27-30.csv", "handwash", "src/FeaturesData/83.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewNewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-11-27-48.csv", "handwash", "src/FeaturesData/84.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewNewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-11-28-03.csv", "handwash", "src/FeaturesData/85.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewNewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-11-28-18.csv", "handwash", "src/FeaturesData/86.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewNewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-11-28-35.csv", "handwash", "src/FeaturesData/87.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewNewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-11-28-50.csv", "handwash", "src/FeaturesData/88.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewNewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-11-29-07.csv", "handwash", "src/FeaturesData/89.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewNewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-11-29-22.csv", "handwash", "src/FeaturesData/90.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewNewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-11-29-40.csv", "handwash", "src/FeaturesData/91.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewNewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-11-30-16.csv", "handwash", "src/FeaturesData/92.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewNewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-11-30-43.csv", "handwash", "src/FeaturesData/93.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewNewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-11-31-09.csv", "handwash", "src/FeaturesData/94.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewNewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-11-31-36.csv", "handwash", "src/FeaturesData/95.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewNewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-11-32-05.csv", "handwash", "src/FeaturesData/96.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewNewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-11-32-22.csv", "handwash", "src/FeaturesData/97.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewNewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-11-32-39.csv", "handwash", "src/FeaturesData/98.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewNewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-11-33-00.csv", "handwash", "src/FeaturesData/99.csv", finalOutputFilename, slidingWindow);
			extractFeatures("src/NewNewCSVData/G6NZCJ00326722F-subject2-left-activity2-none-2019-10-18-11-33-22.csv", "handwash", "src/FeaturesData/100.csv", finalOutputFilename, slidingWindow);

			
		} 
		catch (IOException e) 
		{
			e.printStackTrace();
		}
	}
	
	//extract 12 features + label from each second of data and save in new CSV file
	//Feature order: x mean, x std dev, x median, x mean root square, y mean, y std dev, y median, y mean root square, z mean, z std dev, z median, z mean root square, label
	public static void extractFeatures(String filename, String label, String outputFilename, String finalOutputFilename, int slidingWindow) throws IOException
	{
		slidingWindow = slidingWindow * 1000;
		
		//Calculate features from line by line reading from CSV data
        ArrayList<String[]> features = new ArrayList();
		
        ArrayList<Double> xArray = new ArrayList();
        ArrayList<Double> yArray = new ArrayList();
        ArrayList<Double> zArray = new ArrayList();
        int count = 0;        
		BufferedReader csvReader = new BufferedReader(new FileReader(filename));
		String line = csvReader.readLine();
		String[] data = line.split(",");
		double startSeconds = Double.valueOf(data[0]) / slidingWindow;
		
		while (line != null) 
		{
		    data = line.split(",");
		    int time = Integer.valueOf(data[0]);
		    double seconds = Double.valueOf(time) / slidingWindow;
		    
		    if(seconds >= (startSeconds + 1))
		    {
		    	double[] xVals = getValues(xArray, count);
		    	double[] yVals = getValues(yArray, count);
		    	double[] zVals = getValues(zArray, count);
		    	
		    	String[] f = {Double.toString(xVals[0]), Double.toString(xVals[1]), Double.toString(xVals[2]), Double.toString(xVals[3]), Double.toString(yVals[0]), Double.toString(yVals[1]), Double.toString(yVals[2]), Double.toString(yVals[3]), Double.toString(zVals[0]), Double.toString(zVals[1]), Double.toString(zVals[2]), Double.toString(zVals[3]), label};
		    	features.add(f);
		    	
		    	startSeconds = seconds;
		    	count = 0;
		    	xArray.clear();
		    	yArray.clear();
		    	zArray.clear();
		    	xArray.add(Double.valueOf(data[1]));
			    yArray.add(Double.valueOf(data[2]));
			    zArray.add(Double.valueOf(data[3]));
			    count += 1;
		    }
		    else
		    {
		    	xArray.add(Double.valueOf(data[1]));
			    yArray.add(Double.valueOf(data[2]));
			    zArray.add(Double.valueOf(data[3]));
			    count += 1;
		    }
		    
		    line = csvReader.readLine();
		}
		csvReader.close();
		
		//print newly made features for this file
		//printFeatures(features);
		
		//save features to individual file and final features.csv file
		saveToCSV(features, outputFilename);
		saveToCSV(features, finalOutputFilename);
	}
	
	//Returns mean, std dev, mean and root mean square foe given value
	public static double[] getValues(ArrayList<Double> values, int count)
	{		
		double mean = 0;
		double stdDev = 0;
		double sum = 0;
		double sd = 0;
		
		//mean
		for (double v : values)
		{
			sum += v;
		}
		mean = sum / count;
		
		//std dev
		for(double v : values)
		{
            sd += Math.pow(v - mean, 2);
		}
		stdDev = Math.sqrt(sd/count);
		
		//median
		Collections.sort(values);
	    double median = 0;
	    if (values.size() % 2 == 0) //even num elements
	    {
	    	int sumMiddle = (int) (values.get(values.size() / 2) + values.get(values.size() / 2 - 1));
	    	median = ((double) sumMiddle) / 2;
	    }
	    else //odd num elements
	    {
	    	median = (double) values.get(values.size() / 2);
	    }
	    
	    //root mean square
	    double sq = 0;
	    for(int i = 0; i < values.size(); i++) 
	    { 
	        sq += Math.pow(values.get(i), 2); 
	    } 
	    double m = (sq / values.size()); 
	    double rsm = Math.sqrt(m); 

		double[] vals = {mean, stdDev, median, rsm};
		return vals;
	}
	
	public static void printFeatures(ArrayList<String[]> features)
	{
		for (String[] f : features)
		{
			System.out.println(" ");
			System.out.print(f[0] + " ");
			System.out.print(f[1] + " ");
			System.out.print(f[2] + " ");
			System.out.print(f[3] + " ");
			System.out.print(f[4] + " ");
			System.out.print(f[5] + " ");
			System.out.print(f[6] + " ");
			System.out.print(f[7] + " ");
			System.out.print(f[8] + " ");
			System.out.print(f[9] + " ");
			System.out.print(f[10] + " ");
			System.out.print(f[11] + " ");
			System.out.print(f[12] + " ");
		}
	}
	
	public static void saveToCSV(ArrayList<String[]> features, String outputFilename) throws IOException
	{
		BufferedWriter br = new BufferedWriter(new FileWriter(outputFilename, true));
		
		// Append strings from array
		
		for (String[] f : features)
		{
			StringBuilder sb = new StringBuilder();
			for (String element : f) {
				 sb.append(element);
				 sb.append(",");
			}
			br.write(sb.toString());
			br.write("\n");
		}
		
		br.flush();
		br.close();
	}
	
    public static String[][] readCSV(String filePath) throws Exception {
        StringBuilder sb = new StringBuilder();
        BufferedReader br = new BufferedReader(new FileReader(filePath));
        ArrayList<String> lines = new ArrayList();
        String line;

        while ((line = br.readLine()) != null) {
            lines.add(line);;
        }


        if (lines.size() == 0) {
            System.out.println("No data found");
            return null;
        }

        int lineCount = lines.size();

        String[][] csvData = new String[lineCount][];
        String[] vals;
        int i, j;
        for (i = 0; i < lineCount; i++) {            
                csvData[i] = lines.get(i).split(",");            
        }
        
        return csvData;

    }
	
    public static String csvToArff(String[][] csvData, int[] featureIndices) throws Exception 
    {
        int total_rows = csvData.length;
        int total_cols = csvData[0].length;
        int fCount = featureIndices.length;

        String[] attributeList = new String[fCount + 1];
        
        int i, j;
        for (i = 0; i < fCount; i++) {
            attributeList[i] = csvData[0][featureIndices[i]];
        }
        
        attributeList[i] = csvData[0][total_cols - 1];
        String[] classList = new String[1];
        classList[0] = csvData[1][total_cols - 1];

        for (i = 1; i < total_rows; i++) {
            classList = addClass(classList, csvData[i][total_cols - 1]);
        }
        StringBuilder sb = getArffHeader(attributeList, classList);

        for (i = 1; i < total_rows; i++) {
            for (j = 0; j < fCount; j++) {
                sb.append(csvData[i][featureIndices[j]]);
                sb.append(",");
            }            
            sb.append(csvData[i][total_cols - 1]);
            sb.append("\n");
        }

        return sb.toString();
    }

    public static double classify(String arffData, int option) throws Exception {
		StringReader strReader = new StringReader(arffData);
		Instances instances = new Instances(strReader);
		strReader.close();
		instances.setClassIndex(instances.numAttributes() - 1);
		
		Classifier classifier;
		if(option==1)
			classifier = new J48(); // Decision Tree classifier
		else if(option==2)			
			classifier = new RandomForest();
		else if(option == 3)
			classifier = new SMO();  //This is a SVM classifier
		else 
			return -1;
		
		classifier.buildClassifier(instances); // build classifier
		
		Evaluation eval = new Evaluation(instances);
		eval.crossValidateModel(classifier, instances, 10, new Random(1));
		
		
//		if (option==1)
//		{
//	        final javax.swing.JFrame jf = new javax.swing.JFrame("Weka Classifier Tree Visualizer: J48");
//	        jf.setSize(500,400);
//	        jf.getContentPane().setLayout(new BorderLayout());
//	        TreeVisualizer tv = new TreeVisualizer(null, ((J48) classifier).graph(), new PlaceNode2());
//	        jf.getContentPane().add(tv, BorderLayout.CENTER);
//	        jf.addWindowListener(new java.awt.event.WindowAdapter() {
//	        public void windowClosing(java.awt.event.WindowEvent e) {
//	        	jf.dispose();
//	        }
//	        });
//	
//	        jf.setVisible(true);
//	        tv.fitToScreen();
//        
//		}
		
		return eval.pctCorrect();
	}
    
    private static StringBuilder getArffHeader(String[] attributeList, String[] classList) {
        StringBuilder s = new StringBuilder();
        s.append("@RELATION wada\n\n");

        int i;
        for (i = 0; i < attributeList.length - 1; i++) {
            s.append("@ATTRIBUTE ");
            s.append(attributeList[i]);
            s.append(" numeric\n");
        }

        s.append("@ATTRIBUTE ");
        s.append(attributeList[i]);
        s.append(" {");
        s.append(classList[0]);

        for (i = 1; i < classList.length; i++) {
            s.append(",");
            s.append(classList[i]);
        }
        s.append("}\n\n");
        s.append("@DATA\n");
        return s;
    }

    private static String[] addClass(String[] classList, String className) {
        int len = classList.length;
        int i;
        for (i = 0; i < len; i++) {
            if (className.equals(classList[i])) {
                return classList;
            }
        }

        String[] newList = new String[len + 1];
        for (i = 0; i < len; i++) {
            newList[i] = classList[i];
        }
        newList[i] = className;

        return newList;
    }

    private static void printArray(String[][] x)
    {
    	for (int i = 0; i < x.length; i++)
    	{
    		for (int j = 0; j < x[0].length; j++)
    		{
    			System.out.print(x[i][j]+" ");
    		}
    		System.out.println(" ");
    	}
    }



}
