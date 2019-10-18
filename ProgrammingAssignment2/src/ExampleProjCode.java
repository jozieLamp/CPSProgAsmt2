//   Example code on how to send a subset of features to a classifier
//   and it includes first needing to create an aarf file with 
//   these specific features and then activating the classifier
//   by calling the MyWekaUtils class

//   This example is for a csv file with six features labeled 0 thru 5
//   You will have 12 features

//   This example is choosing three of the features (0, 2 and 5) upon which to
//   train the classifier - you will be using increasing sets of features
//   incrementally and finding the best - see the tutorial

//   You will also not be printing the answers but collecting them and choosing
//   the best features 

public class ExampleProjCode
{

	public static void main(String[] args)
	{
	
		try {
		            //  first get the full features file
			    	String[][] csvData = MyWekaUtils.readCSV("features.csv");           
		            //int[] features = {0, 1, 2, 3, 4, 5}; // all the features
					int[] features = {0, 2, 5}; // use mean_x, mean_y, std_z 
		            String arffData = MyWekaUtils.csvToArff(csvData, features);
		            System.out.println(arffData);
		            
		            double accuracy = MyWekaUtils.classify(arffData, 1);
		            System.out.println(accuracy);
		                        
		} catch (Exception ex) 
		{
		            System.out.println(ex.toString());
		}
	
	}
}