package HomeWork2;

import java.security.KeyStore.Entry.Attribute;
import java.util.ArrayDeque;
import java.util.Queue;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.InstanceComparator;
import weka.core.Instances;

class Node {
	Node[] children;
	Node parent;
	Instances instances;
	int attributeIndex;
	double returnValue;
	int Height;

}

/**
 * @author Ohad
 *
 */
public class DecisionTree implements Classifier {
	private Node rootNodeGini;
	private Node rootNodeEntropy;
	private String impurity;
	private double prune = 1;
	private int maxHeight;
	private double avgHeight;
	
	private double[][] chiSquaredDistributionTable = new double[][] {
		// 0.75
		{ 0.102, 0.575, 1.213, 1.923, 2.675, 3.455, 4.255, 5.071, 5.899, 6.737, 7.584, 8.438, 9.299, 10.165,
				11.037 },
		// 0.5
		{ 0.455, 1.386, 2.366, 3.357, 4.351, 5.348, 6.346, 7.344, 8.343, 9.342, 10.341, 11.340, 12.340, 13.339,
				14.339 },
		// 0.25
		{ 1.323, 2.773, 4.108, 5.385, 6.626, 7.841, 9.037, 10.219, 11.389, 12.549, 13.701, 14.845, 15.984, 17.117,
				18.245 },
		// 0.05
		{ 3.841, 5.991, 7.815, 9.488, 11.070, 12.592, 14.067, 15.507, 16.919, 18.307, 19.675, 21.026, 22.362,
				23.685, 24.996 },
		// 0.005
		{ 7.879, 10.597, 12.838, 14.860, 16.750, 18.548, 20.278, 21.955, 23.589, 25.188, 26.757, 28.300, 29.819,
				31.319, 32.801 } };
	
	@Override
	/*
	 * Builds a decision tree from the training data. 
	 * buildClassifier is separated from buildTree in order to allow you to do 
	 * extra preprocessing before calling buildTree method or post processing after.
	 */
	public void buildClassifier(Instances instances) throws Exception {
		rootNodeGini = new Node();
		rootNodeEntropy = new Node();
		maxHeight = 0;
		avgHeight = 0;
		
		rootNodeGini.instances = instances;
		rootNodeEntropy.instances = instances;
		
		//Gini tree.
		impurity = "gini";
		rootNodeGini.instances = new Instances(instances);
		buildTree(rootNodeGini);
		
		//Entropy tree.
		impurity = "entropy";
		rootNodeEntropy.instances = new Instances(instances);
		buildTree(rootNodeEntropy);
		
	}
	
	public int getMaxHeight() {
		return maxHeight;
	}
	
	public double getAverageHeight() {
		return avgHeight;
	}
	
	public void setImpurity(String i_impurity) {
		impurity = i_impurity;
	}
	
	public Node getTree(String i_impurity) {
		setImpurity(i_impurity);
		if (i_impurity.equals("gini")) {
			return rootNodeGini;
		} else {
			return rootNodeEntropy;
		}
	}
	
	public void setPruneValue(double p_value) {
		prune = p_value;
	}

	// Builds the decision tree on given data set using either a recursive or queue algorithm.
	public void buildTree(Node node) throws Exception {
		if (node.parent != null) {
			node.Height = node.parent.Height + 1;
		} else {
			node.Height = 0;
		}
		
		double classification = getClassification(node.instances);
		
		if (classification < 0.5) {
			node.returnValue = 0.0;
		}
		
		else {
			node.returnValue = 1.0;
		}
		
		node.attributeIndex = findBestAttributeIndex(node.instances);
		
		//if not perfectly classify.
		if ((classification != 1.0) && (classification != 0.0)) {
			
			//gain is not 0;
			if (node.attributeIndex != node.instances.classIndex()) {
				
				
				//Making the children.
				node.children = new Node[node.instances.attribute(node.attributeIndex).numValues()];
				
				Instances[] childrenInstances = new Instances[node.instances.attribute(node.attributeIndex).numValues()];
				for (int i = 0; i < childrenInstances.length; i++) {
					childrenInstances[i] = new Instances(node.instances, 0);
				}
				
				for(int i = 0 ; i < node.instances.size() ; i++) {
					childrenInstances[(int) node.instances.get(i).value(node.attributeIndex)].add(node.instances.get(i));
				}
				
				for (int i = 0; i < node.children.length; i++) {
					int childAtt = findBestAttributeIndex(childrenInstances[i]);
					if (childrenInstances[i].isEmpty() || (prune != 1 && calcChiSquare(childrenInstances[i], childAtt)
							< chiSquaredDistributionTable[p_index()][calcDegreeOfFreedom(childrenInstances[i], childAtt)])) {
						node.children[i] = null;
					}
					else {
						
						//making empty child.
						node.children[i] = new Node();

						//the child parent.
						node.children[i].parent = node;
						
						node.children[i].instances = childrenInstances[i];
						
						//the recursive call.
						buildTree(node.children[i]);
					}
				}
			}
		}
		
		return;
	}
	
	/*
	 * return which place in the chiSquaredDistributionTable array should we look.
	 */
	private int p_index() {
		if (prune == 0.75) {
			return 0;
		}
		if (prune == 0.5) {
			return 1;
		}
		if (prune == 0.25) {
			return 2;
		}
		if (prune == 0.05) {
			return 3;
		}
		if (prune == 0.005) {
			return 4;
		}

		return 1;
	}

	
	/*
	 * Calculate the degree of freedom.
	 * input - instances and attributeIndex.
	 * return - the degree of freedom.
	 */
	private int calcDegreeOfFreedom(Instances instances, int attributeIndex) {

		boolean[] isBelong = new boolean[instances.attribute(attributeIndex).numValues()];
		int temp = 0;

		for (int i = 0; i < instances.size(); i++) {
			isBelong[(int) instances.get(i).value(attributeIndex)] = true;
		}

		for (int i = 0; i < isBelong.length; i++) {
			if (isBelong[i]) {
				temp++;
			}
		}

		return temp - 1;
	}
	
	/*
	 * Finds the best attribute index to split by calculate thier gain.
	 * if there is no such an attribute, we return the class index.
	 * input - instances.
	 * return - attribute
	 */
	private int findBestAttributeIndex(Instances instances) {
		int bestAttributeIndex = instances.classIndex();
		double bestGain = 0;
		double tempGain = 0;
		
		if (instances.size() == 0) {
			return instances.classIndex();
		}
		
		for (int i = 0; i < instances.numAttributes() - 1; i++) {
			
			tempGain = calcGain(instances, i);
			
			if (bestGain < tempGain) {
				bestGain = tempGain;
				bestAttributeIndex = i;
			}
		}
		
		return bestAttributeIndex;
	}

	/*
	 * Returns the classification decided by the most instances.
	 */
	private double getClassification(Instances instances) {
		double purity = 0;
		
		for (int i = 0; i < instances.size(); i++) {
			purity = purity + instances.get(i).classValue();
		}
		return purity / instances.size();
	}
	
    @Override
    // Return the classification of the instance.
	public double classifyInstance(Instance instance) {
    	if (impurity.equals("gini")) {
    		return classifyInstance(instance, rootNodeGini);
		} else {
    		return classifyInstance(instance, rootNodeEntropy);
		}
    }
    
    private double classifyInstance(Instance instance, Node node) {
    	
    	if (node.children == null) {
    		avgHeight += node.Height;
    		if (maxHeight < node.Height) {
				maxHeight = node.Height;
			}
			return node.returnValue;
		} 
    	else {
			if (node.children[(int) instance.value(node.attributeIndex)] == null) {
				avgHeight += node.Height;
				if (maxHeight < node.Height) {
					maxHeight = node.Height;
				}
				return node.returnValue;
			} else {
				return classifyInstance(instance, node.children[(int) instance.value(node.attributeIndex)]);
			}
		}
	}

	/*
     * Calculate the average error on a given instances set (could be the training, test or validation set). 
     * The average error is the total number of classification mistakes on the input 
     * instances set divided by the number of instances in the input set.
     */
    public double calcAvgError(Instances instances) throws Exception {
    	double sum = 0;
    	
    	for (Instance instance : instances) {
			if (classifyInstance(instance) != instance.classValue()) {
				sum++;
			}
		}
    	avgHeight /= instances.size();
    	
    	return sum / instances.size();
    }
    
    /*
     * calculates the gain (giniGain or informationGain depending on the impurity measure) 
     * of splitting the input data according to the attribute.
     */
    public double calcGain(Instances instances, int attributeIndex) {
    	
    	double sumGain;
    	double temp = 0;
    	double tempSum = 0;
    	double[] probAttribute = getProbability(instances, attributeIndex);
    	double[] probClass = getProbability(instances, instances.classIndex());
    	
    	
    	if (impurity.equals("gini")) {
    		sumGain = calcGini(probClass);
		} else {
			sumGain = calcEntropy(probClass);
		}
    	
    	//The attribute.
    	for (int attIndexVal = 0; attIndexVal < probAttribute.length; attIndexVal++) {
    		
    		//initiates.
    		for (int i = 0; i < probClass.length; i++) {
				probClass[i] = 0;
			}
    		temp = 0;
    		tempSum = 0;
    		
    		//count.
    		for (int i = 0; i < instances.size(); i++) {
				if (instances.get(i).value(attributeIndex) == attIndexVal) {
					temp++;
					probClass[(int) instances.get(i).classValue()]++;
				}
			}
    		if(temp != 0) {
    			//|Si|/|S|.
    			for (int i = 0; i < probClass.length; i++) {
    				probClass[i] = probClass[i] / temp;
    			}
    		}
    			//|Sv|/|S|.
    			tempSum = temp / instances.size();
    				
    			if (impurity.equals("gini")) {
    				sumGain = sumGain - (tempSum * calcGini(probClass));
    			} else {
    				sumGain = sumGain - (tempSum * calcEntropy(probClass));
    			}
		}
    	
    	return sumGain;
    }
    
    
    /*
     * calculate for each attribute value the probability.
     * return array: 
     * 1. for each attribute value.
     */
    private double[] getProbability(Instances instances, int attributeIndex) {
    	
    	//array of size of the chosen attribute (number of values).
    	double[] attValues = new double[instances.attribute(attributeIndex).numValues()];
    	
    	for (int i = 0; i < instances.size(); i++) {
    		
			attValues[(int) instances.get(i).value(attributeIndex)]++;	
		}
    	
    	for (int i = 0; i < attValues.length; i++) {
    		
				//|Si|/|S|
				attValues[i] = attValues[i] / instances.size();
		}
		return attValues;
	}

	/*
     * Calculates the Entropy of a random variable.
     */
    public double calcEntropy(double[] prob) {
    	
    	double sumProb = 0;
    	  		
       	for (int i = 0; i < prob.length; i++) {
       		//page 92 lecture 2.
       		if (prob[i] != 0) {
        		sumProb = sumProb - prob[i] * Math.log(prob[i])/Math.log(2);
			}
    	}
    	
    	return sumProb;
    }
    
    /*
     * Calculates the Gini of a random variable.
     */
    public double calcGini(double[] prob) {
    	
    	double sumProb = 1;
    	
    	for (int i = 0; i < prob.length; i++) {
    		//page 82 lecture 2.
			sumProb = sumProb - Math.pow(prob[i], 2);
		}
    	
    	return sumProb;
    }
    
    /*
     * Calculates the chi square statistic of splitting the data 
     * according to the splitting attribute as learned in class.
     */
    public double calcChiSquare(Instances instances, int attributeIndex) {
    	int Df = 0, pf = 0, nf = 0;
    	double E0 = 0, E1 = 0;
    	double x = 0;
    	
    	//E0 = P(Y = 0), E1 = P(Y = 1).
    	for (int i = 0; i < instances.size(); i++) {
    		if (instances.get(i).classValue() == 0) {
				E0++;
			}
    		else {
				E1++;
			}
		}
    	E0 /= instances.size();
    	E1 /= instances.size();
    	
    	for (int i = 0; i < instances.attribute(attributeIndex).numValues(); i++) {

			for (int j = 0; j < instances.size(); j++) {
				
				//xj = f.
				if (instances.get(j).value(attributeIndex) == i) {
					
					Df++;
					
					//Y = 0.
					if (instances.get(j).classValue() == 0) {
						
						pf++;
						
					}
					//Y = 1.
					else { 
						
						nf++;
					}
				}
			}
			
			E0 = E0 * Df;
			E1 = E1 * Df;
			
			x += Math.pow(pf - E0, 2) / E0;
			x += Math.pow(nf - E1, 2) / E1;
			
			//initiating.
			//E0 = P(Y = 0), E1 = P(Y = 1).
			E0 /= Df;
			E1 /= Df;
			Df = 0;
			pf = 0;
			nf = 0;
		}
    	
    	return x;
    }
    
    
    @Override
	public double[] distributionForInstance(Instance instances) throws Exception {
		// Don't change
		return null;
	}
    
	@Override
	public Capabilities getCapabilities() {
		// Don't change
		return null;
	}
	
	/*
	 * Print the tree.
	 */
	public void printTree() {
		System.out.println("Root");
		
		if (impurity.equals("gini")) {
			
			if (rootNodeGini.returnValue == rootNodeGini.instances.classIndex()) {
				System.out.print("Leaf. Returning value: " + rootNodeGini.returnValue);
			} 
			else {
				System.out.println("Returning value: " + rootNodeGini.returnValue);
				
				for (int i = 0; i < rootNodeGini.children.length; i++) {
					printTree(rootNodeGini.children[i], 1, i);
				}
			}
		}
		else {
			
			if (rootNodeEntropy.returnValue == rootNodeEntropy.instances.classIndex()) {
				System.out.print("Leaf. Returning value: " + rootNodeEntropy.returnValue);
			} 
			else {
				System.out.println("Returning value: " + rootNodeEntropy.returnValue);
				
				for (int i = 0; i < rootNodeEntropy.children.length; i++) {
					printTree(rootNodeEntropy.children[i], 1, i);
				}
			}
		}
	}
	
	private void printTree(Node node, int depth, int index) {
		
		if (node != null) {
			
			printTabs(depth);
			System.out.println("If attribute " + node.parent.attributeIndex + "=" + index);

			printTabs(depth);
			if (node.children != null) {
				
				System.out.println("Returning value: " +  node.returnValue);
				
				for (int i = 0; i < node.children.length; i++) {
					printTree(node.children[i], depth + 1, i);
				}
			}
			else {
				
				System.out.println("Leaf. Returning value: " + node.returnValue);
			}
		}
	}
	
	/*
	 * print tabs.
	 */
	private void printTabs(int num) {
		for(int i = 0 ; i < num ; i++) {
			System.out.print('\t');
		}
	}
}
