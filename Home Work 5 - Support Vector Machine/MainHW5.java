package HomeWork5;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instances;

public class MainHW5 {
	
	public static enum KernelType {
		Polynomial_Kernel,
		RBF_Kernel
	}

	final private static double alpha = 1.5;
	final private static double[] polyKernel = {2, 3, 4};
	final private static double[] rbfKernel = {0.005, 0.05, 0.5};
	final private static int[] cValueI = {1, 0, -1, -2, -3, -4};
	final private static int[] cValueJ = {3, 2, 1};

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
		
		Instances data = new Instances(loadData("cancer.txt"));
		Random rand = new Random();
		data.randomize(rand);
		int per80 =(int) (data.size() * 0.8);
		//best = {degree or gamma, alpha(TPR-FPR)}. 
		double[] best = {0, Double.MIN_VALUE};
		double temp;
		
		//1. Divide the data to training and test set – 80% training and 20% test.
		Instances trainingData = new Instances(data, 0, per80 - 1);
		Instances testData = new Instances(data, per80, data.size() - per80);
		
		//2. For each kernel, build the SVM classifier on the training set using the SMO WEKA class.
		//3. Calculate & print to the console the TPR and FPR on the test set.
		SVM svm;
		PolyKernel kernelP;
		RBFKernel kernelR;
		
		for (double degree : polyKernel) {
			svm = new SVM();
			kernelP = new PolyKernel();
			kernelP.setExponent(degree);
			svm.setKernel(degree);
			svm.setKernelType(KernelType.Polynomial_Kernel);
			svm.setKernel(kernelP);
			svm.buildClassifier(trainingData);
			svm.calcPositiveRates(testData);
			temp = (alpha * svm.getTPR()) - svm.getFPR();
			
			if (best[1] < temp) {
				
				best[0] = degree;
				best[1] = temp;
			}
			
			System.out.println("For PolyKernel with degree " + degree + " the rates");
			System.out.println("are:");
			System.out.println("TPR = " + svm.getTPR());
			System.out.println("FPR = " + svm.getFPR());
			System.out.println();
		}
		
		for (double gamma : rbfKernel) {
			svm = new SVM();
			kernelR = new RBFKernel();
			kernelR.setGamma(gamma);
			svm.setKernel(gamma);
			svm.setKernelType(KernelType.RBF_Kernel);
			svm.setKernel(kernelR);
			svm.buildClassifier(trainingData);
			svm.calcPositiveRates(testData);
			temp = (alpha * svm.getTPR()) - svm.getFPR();
			
			if (best[1] < temp) {
				
				best[0] = gamma;
				best[1] = temp;
			}
			
			System.out.println("For RBFKernel with gamma " + gamma + " the rates are:");
			System.out.println("TPR = " + svm.getTPR());
			System.out.println("FPR = " + svm.getFPR());
			System.out.println();
		}
		
		//4. Select the best kernel according to the best alpha*TPR-FPR (with alpha=1.5).
		
		svm = new SVM();
		if (best[0] > 1) {
			svm.setKernelType(KernelType.Polynomial_Kernel);
		} else {
			svm.setKernelType(KernelType.RBF_Kernel);
		}
		
		svm.setKernel(best[0]);
		svm.buildClassifier(trainingData);
		svm.calcPositiveRates(testData);
		
		System.out.println("The best kernel is: "+ svm.getKernelType().toString() + " " + best[0] + " " + best[1]);
		System.out.println();
	
		//-----------------------------------------------------------------------------------
		//Finding the best C value.
		//-----------------------------------------------------------------------------------
		
		double c;
		
		//1. For the selected kernel, try different C values.
		for (int i : cValueI) {
			for (int j : cValueJ) {
				svm = new SVM();
				c = (Math.pow(10, i)) * ((double)j / 3);
				
				//2. build the SVM classifier with the selected kernel on the training set.
				svm.setC(c);
				PolyKernel kernel = new PolyKernel();
				kernel.setExponent(2);
				svm.setKernel(kernel);
				svm.buildClassifier(trainingData);
				//3. Calculate & print to the console the TPR and FPR on the test set.
				svm.calcPositiveRates(testData);
				
				System.out.println("For C " + c + " the rates are:");
				System.out.println("TPR = " + svm.getTPR());
				System.out.println("FPR = " + svm.getFPR());
				System.out.println();
			}
		}
	}
	
}
