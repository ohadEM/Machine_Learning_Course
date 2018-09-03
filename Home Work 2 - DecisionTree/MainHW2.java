package HomeWork2;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Instances;

public class MainHW2 {

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
		Instances trainingCancer = loadData("cancer_train.txt");
		Instances testingCancer = loadData("cancer_test.txt");
		Instances validationCancer = loadData("cancer_validation.txt");
		double errorEntropy = 0;
		double errorGini = 0;
		double bestPvalue = 1;
		double bestError = Double.MAX_VALUE;
		
		//build tree
		DecisionTree tree = new DecisionTree();
		tree.buildClassifier(trainingCancer);
		
		
		errorEntropy = tree.calcAvgError(validationCancer);
		System.out.println("Validation error using Entropy: " + errorEntropy);
		
		tree.getTree("gini");
		errorGini = tree.calcAvgError(validationCancer);
		System.out.println("Validation error using Gini: " + errorGini);
		
		if (errorGini > errorEntropy) {
			tree.setImpurity("entropy");
		} 
		
		System.out.println("---------------------------------------------------");
		
		DecisionTree treePrune1 =new DecisionTree();
		treePrune1.setPruneValue(1);
		treePrune1.buildClassifier(trainingCancer);
		System.out.println("Decision Tree with p_value of: 1");
		
		errorGini = treePrune1.calcAvgError(trainingCancer);
		System.out.println("The train error of the decision tree is " + errorGini);
		
		errorGini = treePrune1.calcAvgError(validationCancer);
		if (bestError > errorGini) {
			bestError = errorGini;
			bestPvalue = 1;
		}
		System.out.println("Max height on validation data: " + treePrune1.getMaxHeight());
		System.out.println("Average hight on validation data: " + treePrune1.getAverageHeight());
		System.out.println("The validation error of the decision tree is " + errorGini);
		
		System.out.println("---------------------------------------------------");
		
		DecisionTree treePrune075 =new DecisionTree();
		treePrune075.setPruneValue(0.75);
		treePrune075.buildClassifier(trainingCancer);
		System.out.println("Decision Tree with p_value of: 0.75");

		errorGini = treePrune075.calcAvgError(trainingCancer);
		System.out.println("The train error of the decision tree is " + errorGini);
		
		errorGini = treePrune075.calcAvgError(validationCancer);
		if (bestError > errorGini) {
			bestError = errorGini;
			bestPvalue = 0.75;
		}
		System.out.println("Max height on validation data: " + treePrune075.getMaxHeight());
		System.out.println("Average height on validation data: " + treePrune075.getAverageHeight());
		System.out.println("The validation error of the decision tree is " + errorGini);
		
		System.out.println("---------------------------------------------------");
		
		DecisionTree treePrune05 =new DecisionTree();
		treePrune05.setPruneValue(0.5);
		treePrune05.buildClassifier(trainingCancer);
		System.out.println("Decision Tree with p_value of: 0.5");

		errorGini = treePrune05.calcAvgError(trainingCancer);
		System.out.println("The train error of the decision tree is " + errorGini);
		
		errorGini = treePrune05.calcAvgError(validationCancer);
		if (bestError > errorGini) {
			bestError = errorGini;
			bestPvalue = 0.5;
		}
		System.out.println("Max height on validation data: " + treePrune05.getMaxHeight());
		System.out.println("Average height on validation data: " + treePrune05.getAverageHeight());
		System.out.println("The validation error of the decision tree is " + errorGini);
		
		System.out.println("---------------------------------------------------");
		
		DecisionTree treePrune025 =new DecisionTree();
		treePrune025.setPruneValue(0.25);
		treePrune025.buildClassifier(trainingCancer);
		System.out.println("Decision Tree with p_value of: 0.25");

		errorGini = treePrune025.calcAvgError(trainingCancer);
		System.out.println("The train error of the decision tree is " + errorGini);
		
		errorGini = treePrune025.calcAvgError(validationCancer);
		if (bestError > errorGini) {
			bestError = errorGini;
			bestPvalue = 0.25;
		}
		System.out.println("Max height on validation data: " + treePrune025.getMaxHeight());
		System.out.println("Average height on validation data: " + treePrune025.getAverageHeight());
		System.out.println("The validation error of the decision tree is " + errorGini);
		
		System.out.println("---------------------------------------------------");
		
		DecisionTree treePrune005 =new DecisionTree();
		treePrune005.setPruneValue(0.05);
		treePrune005.buildClassifier(trainingCancer);
		System.out.println("Decision Tree with p_value of: 0.05");

		errorGini = treePrune005.calcAvgError(trainingCancer);
		System.out.println("The train error of the decision tree is " + errorGini);
		
		errorGini = treePrune005.calcAvgError(validationCancer);
		if (bestError > errorGini) {
			bestError = errorGini;
			bestPvalue = 0.05;
		}
		System.out.println("Max height on validation data: " + treePrune005.getMaxHeight());
		System.out.println("Average height on validation data: " + treePrune005.getAverageHeight());
		System.out.println("The validation error of the decision tree is " + errorGini);
		
		System.out.println("---------------------------------------------------");

		DecisionTree treePrune0005 =new DecisionTree();
		treePrune0005.setPruneValue(0.005);
		treePrune0005.buildClassifier(trainingCancer);
		System.out.println("Decision Tree with p_value of: 0.005");

		errorGini = treePrune0005.calcAvgError(trainingCancer);
		if (bestError > errorGini) {
			bestError = errorGini;
			bestPvalue = 0.005;
		}
		System.out.println("The train error of the decision tree is " + errorGini);
		System.out.println("Max height on validation data: " + treePrune0005.getMaxHeight());
		System.out.println("Average height on validation data: " + treePrune0005.getAverageHeight());
		System.out.println("The validation error of the decision tree is " + errorGini);
		
		System.out.println("---------------------------------------------------");
		
		System.out.println("Best validation error at p_value = " + bestPvalue);
		
		if (bestPvalue == 1) {
			System.out.println("Test error with best tree: " + treePrune1.calcAvgError(testingCancer));
			System.out.println("Representation of the best tree by ‘if statements’");
			tree.printTree();
			
		} else if (bestPvalue == 0.75) {
			System.out.println("Test error with best tree: " + treePrune075.calcAvgError(testingCancer));
			System.out.println("Representation of the best tree by ‘if statements’");
			tree.printTree();
			
		} else if (bestPvalue == 0.5) {
			System.out.println("Test error with best tree: " + treePrune05.calcAvgError(testingCancer));
			System.out.println("Representation of the best tree by ‘if statements’");
			tree.printTree();
			
		} else if (bestPvalue == 0.25) {
			System.out.println("Test error with best tree: " + treePrune025.calcAvgError(testingCancer));
			System.out.println("Representation of the best tree by ‘if statements’");
			tree.printTree();
			
		} else if (bestPvalue == 0.05) {
			System.out.println("Test error with best tree: " + treePrune005.calcAvgError(testingCancer));
			System.out.println("Representation of the best tree by ‘if statements’");
			tree.printTree();
			
		} else if (bestPvalue == 0.005) {
			System.out.println("Test error with best tree: " + treePrune0005.calcAvgError(testingCancer));
			System.out.println("Representation of the best tree by ‘if statements’");
			tree.printTree();
			
		}
		
	}
}
