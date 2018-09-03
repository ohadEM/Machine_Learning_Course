package HomeWork1;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class MainHW1 {
	
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}
		
	/**
	 * Sets the class index as the last attribute.
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException{
		BufferedReader datafile = readDataFile(fileName);

		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	
	public static void main(String[] args) throws Exception {
		//load data
		Instances data1 = loadData("wind_training.txt");
	
		//find best alpha and build classifier with all attributes
		LinearRegression lRegression1 = new LinearRegression();
		lRegression1.buildClassifier(data1);
		
		System.out.println("Test on wind_training.txt file:");
		System.out.println("The best alpha is: " + lRegression1.getAlpha());
		System.out.println("The best Error is: " + lRegression1.getError());
		System.out.println();
		
		//load data
		Instances data2 = loadData("wind_testing.txt");
		
		//find best alpha and build classifier with all attributes
		LinearRegression lRegression2 = new LinearRegression();
		lRegression2.buildClassifier(data2);
		
		System.out.println("Test on wind_testing.txt file:");
		System.out.println("The best Alpha is: " + lRegression2.getAlpha());
		System.out.println("The best Error is: " + lRegression2.getError());
		
		threeBestFeatures(lRegression1, data1);
		
		threeBestFeatures(lRegression2, data2);	
	}
	
	public static void threeBestFeatures(LinearRegression lRegression, Instances data) throws Exception {
		//build classifiers with all 3 attributes combinations
				int truNumAttributes = data.numAttributes() - 1;
				int[] bestAttIndex = new int[3];
				double bestError = Double.MAX_VALUE;
				double tempError;
				int i, j = 0, k = 0;
				for(i = 0; i < truNumAttributes; i++) {
					for(j = i + 1; j < truNumAttributes; j++) {
						for(k = j + 1; k < truNumAttributes; k++) {
							Instances tempInstances = new Instances(data);	
							for (int z = truNumAttributes - 1; z >= 0; z--) {
								if ((z != i) && (z != j) && (z != k)) {
									tempInstances.deleteAttributeAt(z);
								}	
							}
							LinearRegression tempLR = new LinearRegression();
							tempLR.buildClassifier(tempInstances, lRegression.getAlpha());
							tempError = tempLR.calculateMSE(tempInstances);
							System.out.println(tempError);
							if(bestError > tempError) {
								bestError = tempError;
								bestAttIndex[0] = i;
								bestAttIndex[1] = j;
								bestAttIndex[2] = k;			
							}	
						}
					}
				}
				System.out.println("**********************************************************************");
				System.out.println("> Best error: " + bestError);
				System.out.println("> Best features: " + bestAttIndex[0] + " " + bestAttIndex[1] 
														+ " " + bestAttIndex[2]);
				System.out.println("**********************************************************************");
	}

}
