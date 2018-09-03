package HomeWork3;
//package HomeWork3;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

// Class enum that holds two types of distance check.
enum DistanceCheck{Regular, Efficient}

class DistanceCalculator {
	
	public DistanceCheck distance = DistanceCheck.Regular; // regular as default.
	double threshold = Double.MAX_VALUE;
	boolean efficient;
	int p;
	
    /**
    * We leave it up to you wheter you want the distance method to get all relevant
    * parameters(lp, efficient, etc..) or have it has a class variables.
    */
    public double distance (Instance one, Instance two) {
    	double dist;
    	if(distance == DistanceCheck.Regular) {
    		if(p == 1 || p == 2 || p == 3) {
    			dist = lpDistance(one, two);
    		}
    		// If p is infinity.
    		else {
    			dist = lInfinityDistance(one, two);
    		}
    	}
    	// If distance is equal to Efficient.
    	else {

    		if(p == 1 || p == 2 || p == 3) {
    			dist = efficientLpDistance(one, two);
    		}
    		// If p is infinity.
    		else {
    			dist = efficientLInfinityDistance(one, two);
    		}
    	}
    	return dist;
    }

    /**
     * Returns the Lp distance between 2 instances.
     * @param one
     * @param two
     */
    public double lpDistance(Instance one, Instance two) {
    	
    	double temp = 0;
    	double sum = 0;
    	
    	// Calculating the Distance for Numeric Features
		// by L-p distance.
    	for (int i = 0; i < one.numAttributes(); i++) {
    		
    		temp = one.value(i) - two.value(i);
    		temp = Math.abs(temp);
			sum = sum + Math.pow(temp, p);
		}
    	
        return Math.pow(sum, 1.0 / p);
    }

    /**
     * Returns the L infinity distance between 2 instances.
     * @param one
     * @param two
     * @return
     */
    public double lInfinityDistance(Instance one, Instance two) {
    	
    	double curDist, maxDist = 0;
    	
    	// Calculating the Distance for Numeric Features when p is infinity.
    	for(int i = 0; i < one.numAttributes(); i++) {		
    		curDist = Math.abs(one.value(i) - two.value(i));	
    		if(maxDist < maxDist) {
    			maxDist = curDist;
    		}
    	}
        return maxDist;
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     * @param one
     * @param two
     * @return
     */
    public double efficientLpDistance(Instance one, Instance two) {
    	
    	int numberOfAttributes = one.numAttributes();
		double sum = 0;
		double currentDifference;

		for (int i = 0; i < numberOfAttributes - 1; i++) {			
			currentDifference = Math.abs(one.value(i) - two.value(i));
			sum += Math.pow(currentDifference, p);
			if(sum >= threshold) {
				return Double.MAX_VALUE;
			}
		}
		return sum;
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     * @param one
     * @param two
     * @return
     */
    public double efficientLInfinityDistance(Instance one, Instance two) {
    	
    			double maxDist = 0;
    			double sum = 0;

    			for (int i = 0; i < one.numAttributes() - 1; i++) {	
    				double currentDifference = Math.abs(one.value(i) - two.value(i));	
    				if (currentDifference > maxDist) {
    					if(sum >= threshold) {			
    						return Double.MAX_VALUE;
    					}
    					maxDist = currentDifference;
    				}
    			}
    			return maxDist;
    }
    
    // Setting the threshold.
    public void setThreshold(double threshold) {
    	this.threshold = threshold;
    }
    
    public double getThreshold() {
    	return threshold;
    }
    
    public void setP(int p) {
    	this.p = p;
    }
    
}

// Class That holds the the neighbours instances and distances.
class Neighbour{
	
	private Instance instance;
	private double distance;
	
	public Neighbour(Instance instance, double dist) {
		this.instance = instance;
		this.distance = dist;
	}
	
	public Instance getInstance() {
		return instance;
	}
	
	public double getDistance() {
		return distance;
	}
	
}

public class Knn implements Classifier {

	public enum WeightingScheme{Uniform, Weighted}
    private Instances m_trainingInstances;
    private Instances m_testiningInstances;
    private WeightingScheme weight;
    private DistanceCheck distance;
    private int k;
    private int p;
    private boolean time = false;
    private long timer = 0;
    private Neighbour[] m_Neighbours;
    private double threshold;
    private DistanceCalculator distanceCalculator;
    private double bestError;
    
    public void setWeight(WeightingScheme weight) {
    	this.weight = weight;
    }
    
    public WeightingScheme getWeight() {
    	return weight;
    }
    
    public void setDistanceCheck(DistanceCheck distance) {
		this.distance = distance;
	} 
    
    public DistanceCheck getDistanceCheck() {
		return distance;
	}
    
    public void setBestError( double error) {
		bestError = error;
	}
    
     
    @Override
    /**
     * Build the knn classifier. In our case, simply stores the given instances for 
     * later use in the prediction.
     * @param instances
     */
    public void buildClassifier(Instances instances) throws Exception {
    	m_trainingInstances = instances;
    }
    
    

    /**
     * Returns the knn prediction on the given instance.
     * @param instance
     * @return The instance predicted value.
     */
    public double regressionPrediction(Instance instance) {
    	Neighbour[] m_Neighbours = findNearestNeighbors(instance);
    	if(weight == WeightingScheme.Weighted) {
    		return getWeightedAverageValue(m_Neighbours, instance);
    	}
    	else {
    		for(int i = 0; i < m_Neighbours.length; i++)
			{
				if(distanceCalculator.distance(m_Neighbours[i].getInstance(), instance) == 0)
				{
					return m_Neighbours[i].getInstance().classValue();
				}
			}

			return getAverageValue(m_Neighbours);
    	}
    }

    /**
     * Caclcualtes the average error on a give set of instances.
     * The average error is the average absolute error between the target value and the predicted
     * value across all insatnces.
     * @param insatnces
     * @return
     */
    public double calcAvgError (Instances instances){
    	int size = instances.size();
		double sumErr = 0; 
		for (int i = 0; i < size; i++) {
			sumErr += Math.abs(regressionPrediction(instances.get(i)) - instances.get(i).classValue());
		}
		return (sumErr / size);
    }

    /**
     * Calculates the cross validation error, the average error on all folds.
     * @param insances Insances used for the cross validation
     * @param num_of_folds The number of folds to use.
     * @return The cross validation error.
     */
    public double crossValidationError(Instances instances, int num_of_folds){
    	double cvError = 0;
    	
    	distanceCalculator.setThreshold(Double.MAX_VALUE);
    	int reminder = instances.numInstances() % num_of_folds;
    	
    	Instances [] folds = new Instances[num_of_folds];
    	for (int i = 0; i < folds.length; i++) {
            folds[i] = new Instances(instances, 0);
        }
    	
    	for(int i = 0; i < instances.size(); ++i) {
    		folds[i % num_of_folds].add(instances.get(i));
    	}
    	
    	for (int i = 0; i < folds.length; i++) {
    		m_testiningInstances = folds[i];
			m_trainingInstances = new Instances(instances, 0);
			
			for (int j = 0; j < folds.length; j++) {
				if (i != j) {
					
					for (int k = 0; k < folds[j].size(); k++) {
						m_trainingInstances.add(folds[j].get(k));
					}
				}
			}
			
			cvError += calcAvgError(m_testiningInstances);
		}
    	
    	m_trainingInstances = instances;
    	
        return cvError /= num_of_folds;
    }


    /**
     * Finds the k nearest neighbors.
     * @param instance
     */
    public Neighbour[] findNearestNeighbors(Instance instance) {
    	distanceCalculator = new DistanceCalculator();
    	//calculator.setThreshold(threshold);
    	distanceCalculator.setP(p);
    	
    	Neighbour[] neighbours = new Neighbour[k];
    	
    	//double[] dist = new double[m_trainingInstances.numInstances()];
    	//double[] dist = new double[k];
    	double tempMax = -1;
    	Neighbour neighbour;
    	for(int i = 0; i < m_trainingInstances.size(); i++) {
    		neighbour = new Neighbour(m_trainingInstances.instance(i), distanceCalculator.distance(instance, m_trainingInstances.instance(i)));
    		if (i < k) {
    			
    			neighbours[i] = neighbour;
    			if(neighbour.getDistance() > tempMax) {
    				tempMax = neighbour.getDistance();
    				distanceCalculator.setThreshold(tempMax);
    			}
			} 
    		else {
    			
    			if(neighbour.getDistance() < tempMax) {
    				
    				for(int j = 0; j < neighbours.length; j++) {
    					if(tempMax == neighbours[j].getDistance()) {
    						neighbours[j] = neighbour;
    						break;
    					}
    				}

					tempMax = - 1;
    				
    				for (int j = 0; j < neighbours.length; j++) {
    					if(neighbour.getDistance() > tempMax) {
    	    				tempMax = neighbour.getDistance();
    	    				distanceCalculator.setThreshold(tempMax);
    	    			}
					}
    			}
    		}
    	}
    	
    	return neighbours;
    	
    }

    /**
     * Cacluates the average value of the given elements in the collection.
     * @param
     * @return
     */
    public double getAverageValue (Neighbour[] neighbours) {
    	
    	double sum = 0.0;
		for(int i = 0; i < neighbours.length; i++)
		{
			sum += neighbours[i].getInstance().classValue();
		}

		return sum / neighbours.length;
    }

    /**
     * Calculates the weighted average of the target values of all the elements in the collection
     * with respect to their distance from a specific instance.
     * @return
     */
    public double getWeightedAverageValue(Neighbour[] neighbours, Instance instance) {
        double distance, sum = 0, sumDistances = 0;
    	for (int i = 0; i < neighbours.length; i++) {
			if(p == 4) {
				distance = distanceCalculator.lInfinityDistance(neighbours[i].getInstance(), instance);
			}
			// If p is finite.
			else {
				distance = distanceCalculator.lpDistance(neighbours[i].getInstance(), instance); 
			}
			
			if (distance == 0) {
				return neighbours[i].getInstance().classValue();
			}
			
			sum += (neighbours[i].getInstance().classValue()) / (Math.pow(distance, 2));
			sumDistances += (1 / Math.pow(distance, 2));
		}
    	
    	return sum / sumDistances;
    }
    
    
    @Override
    public double[] distributionForInstance(Instance arg0) throws Exception {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public Capabilities getCapabilities() {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public double classifyInstance(Instance instance) {
        // TODO Auto-generated method stub - You can ignore.
        return 0.0;
    }



	public Instances getM_trainingInstances() {
		return m_trainingInstances;
	}



	public void setM_trainingInstances(Instances m_trainingInstances) {
		this.m_trainingInstances = m_trainingInstances;
	}


	/**
	 * @return the k
	 */
	public int getK() {
		return k;
	}


	/**
	 * @param k the k to set
	 */
	public void setK(int k) {
		this.k = k;
	}


	/**
	 * @return the p
	 */
	public int getP() {
		return p;
	}


	/**
	 * @param p the p to set
	 */
	public void setP(int p) {
		this.p = p;
	}


	/**
	 * @return the distance
	 */
	public DistanceCheck getDistance() {
		return distance;
	}
	
	/**
	 * @return the best error.
	 */
	public double getBestError() {
		return bestError;
	}


	/**
	 * @param distance the distance to set
	 */
	public void setDistance(DistanceCheck distance) {
		this.distance = distance;
	}


	/**
	 * @return the m_testiningInstances
	 */
	public Instances getTestiningInstances() {
		return m_testiningInstances;
	}


	/**
	 * @param m_testiningInstances the m_testiningInstances to set
	 */
	public void setTestiningInstances(Instances m_testiningInstances) {
		this.m_testiningInstances = m_testiningInstances;
	}


	
	
	public DistanceCalculator getDistanceCalculator() {
		return distanceCalculator;
	}
	
	public void calcTreshold(Instance one, Instance two) {
		DistanceCalculator calculator = new DistanceCalculator();
		if(p != 4) {
			threshold = calculator.lpDistance(one, two);
		}
		else {
			threshold = calculator.lInfinityDistance(one, two);
		}
	}
}
