package HomeWork3;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Collections;

import HomeWork3.DistanceCheck;
import HomeWork3.Knn.WeightingScheme;
import weka.core.Instance;
import weka.core.Instances;

public class MainHW3 {
	
	//static int numFolds = 10;

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void main(String[] args) throws Exception {
		
		Instances data = loadData("auto_price.txt");
		int[] folds ={data.size(), 50, 10, 5, 3};
		
		Collections.shuffle(data);
		
		Knn knn = new Knn();
		ChooseHyperParameters(data, knn);
        
		//print
		System.out.println("----------------------------");
		System.out.println("Results for original dataset:");
		System.out.println("----------------------------");
		
		//todo: if infinity lp.
		if (knn.getP() == 4) {
			System.out.println("Cross validation error with K = " + knn.getK() + ", lp = Infinity, majority function");
			System.out.println("= " + knn.getWeight().toString() + " for auto_price data is: <cross_vaidation_error>");
			System.out.println();
		} 
		else {
			System.out.println("Cross validation error with K = " + knn.getK() + ", lp = " + knn.getP() + ", majority function");
			System.out.println("= " + knn.getWeight().toString() + " for auto_price data is: " + knn.getBestError());
			System.out.println();
		}
		
		
		data = FeatureScaler.scaleData(data);
		
		System.out.println("----------------------------");
		System.out.println("Results for scaled dataset:");
		System.out.println("----------------------------");
		
		Knn scaledKnn = new Knn();
		scaledKnn.buildClassifier(data);
		
		
		System.out.println("Cross validation error with K = " + scaledKnn.getK() + ", lp = " + scaledKnn.getP() + ", majority function");
		System.out.println("= " + scaledKnn.getWeight().toString() + " for auto_price data is: " + scaledKnn.getBestError());
		System.out.println();
		
		double start, stop, crossValError;
		for (int i = 0; i < folds.length; i++) {
			System.out.println("----------------------------");
			System.out.println("Results for " + folds[i] +" folds: ");
			System.out.println("----------------------------");
			
			start = System.nanoTime();
			crossValError = scaledKnn.crossValidationError(data, folds[i]);
			stop = System.nanoTime();
			scaledKnn.setDistance(DistanceCheck.Regular);
			System.out.println("Cross validation error of regular knn on auto_price dataset is " + crossValError + " and");
			System.out.println("the average elapsed time is " + (stop - start) /  folds[i]);
			System.out.println("The total elapsed time is: " + (stop - start));
			System.out.println();
			
			start = System.nanoTime();
			crossValError = scaledKnn.crossValidationError(data, folds[i]);
			stop = System.nanoTime();
			scaledKnn.setDistance(DistanceCheck.Efficient);
			System.out.println("Cross validation error of efficient knn on auto_price dataset is " + crossValError + " and");
			System.out.println("the average elapsed time is " + (stop - start) /  folds[i]);
			System.out.println("The total elapsed time is: " + (stop - start));
			System.out.println();
		}
		
	}
	
	public static void ChooseHyperParameters(Instances instances, Knn knn) throws Exception {
		double bestError = Double.MAX_VALUE, tempError;
		int bestP = 0, tempP = 0, bestK = 0, tempK = 0;
		WeightingScheme tempMethod = null, bestMethod = null;
		
		for (int i = 1; i <= 20; i++) {
			knn.setK(i);
			//p
			for (int j = 1; j <= 4; j++) {
				knn.setP(j);
				for (WeightingScheme method : WeightingScheme.values()) {
					knn.setWeight(method);
					tempError = knn.crossValidationError(instances, 10);
					if (tempError < bestError) {
						bestError = tempError;
						bestK = tempK;
						bestP = tempP;
						bestMethod = tempMethod;
					}
				}
			}	
		};
		knn.setK(bestK);
		knn.setP(bestP);
		knn.setWeight(bestMethod);
		knn.setBestError(bestError);
	}

	
	
}
