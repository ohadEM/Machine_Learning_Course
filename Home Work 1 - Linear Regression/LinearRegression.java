package HomeWork1;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class LinearRegression implements Classifier {
	
    private int m_ClassIndex;
	private int m_truNumAttributes;
	private double[] m_coefficients;
	private double m_alpha;
	private double m_bestError;
	
	//the method which runs to train the linear regression predictor, i.e.
	//finds its weights.
	//@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		m_ClassIndex = trainingData.classIndex();
		m_truNumAttributes = trainingData.numAttributes();
		m_coefficients = new double[m_truNumAttributes];
		m_bestError = Double.MAX_VALUE;
		findAlpha(trainingData);
		m_coefficients = gradientDescent(trainingData);
	}
	
	public void buildClassifier(Instances trainingData, double alpha) throws Exception {
		m_alpha = alpha;
		//m_bestError = Double.MAX_VALUE;
		m_ClassIndex = trainingData.classIndex();
		m_truNumAttributes = trainingData.numAttributes();
		m_coefficients = new double[m_truNumAttributes];
		m_coefficients = gradientDescent(trainingData);
	}
	
	
	
	private void findAlpha(Instances data) throws Exception {
		
		double tempError;						// Error of the same Alpha before 100 iterations. 
		double curError; 						// The error of the same Alpha after 100 iterations.
		double bestAlpha = 0;					// Our current best Alpha.

		for(int i = -17; i <= 0; i ++) {
			
			m_alpha = Math.pow(3, i);
			tempError = Double.MAX_VALUE;
			
			for(int j = 0; j < m_truNumAttributes; j++) {
				m_coefficients[j] = 1;
			}
			
			for(int k = 0; k < 20000; k++) {
				m_coefficients = gradientDescent(data);
				
				// Comparing of the errors every 100 iterations.
				if((k + 1) % 100 == 0) {
					
					curError = calculateMSE(data);
					
					// If the current error is bigger than the previous
					// error stop the iterations and return the previous error. 
					if(tempError < curError) {
						curError = tempError;
						break;
					}
					
					// If the difference between the errors is very small, say smaller than 0.003.
					if(Math.abs(tempError - curError) < 0.003) {
						bestAlpha = m_alpha;
						m_bestError = Math.min(curError, tempError);
						break;
					}
					
					// If the current Alpha better than best Alpha
					if (m_bestError > curError) {
						m_bestError = curError;
						bestAlpha = m_alpha;
					}
					
					// Otherwise we continue
					tempError = curError;
				}
			}
		}
		m_alpha = bestAlpha;
	}
	
	/**
	 * An implementation of the gradient descent algorithm which should
	 * return the weights of a linear regression predictor which minimizes
	 * the average squared error.
     * 
	 * @param trainingData
	 * @throws Exception
	 */
	private double[] gradientDescent(Instances trainingData)
			throws Exception {
		double[] gradientArr = new double[m_truNumAttributes];

		for(int i = 0; i < m_truNumAttributes; i++) {
			gradientArr[i] = costFunc(trainingData, i);
		}
		
		for(int j = 0; j < m_truNumAttributes; j++) {
			m_coefficients[j] =  m_coefficients[j] - m_alpha * gradientArr[j];
		}
		
		return m_coefficients;
	}
	
	private double costFunc(Instances trainingData, int index) throws Exception {
		
		double sum = 0;
		double temp;
		int size = trainingData.numInstances();
		
		// Runs for each Instance.
		for(int i = 0; i < size; i++) {
			temp = 0;
			temp = temp + regressionPrediction(trainingData.get(i));
			temp = temp - trainingData.get(i).value(m_ClassIndex);			
			if (index != 0) {
				temp = temp * trainingData.get(i).value(index - 1);
			}
			sum = sum + temp;
		}
		return sum / size;
	}
	
	/**
	 * Returns the prediction of a linear regression predictor with weights
	 * given by m_coefficients on a single instance.
     *
	 * @param instance
	 * @return
	 * @throws Exception
	 */
	public double regressionPrediction(Instance instance) throws Exception {
		
		double res = m_coefficients[0];
		
		for (int i = 0; i < m_truNumAttributes - 1; i++) {
			res = res + m_coefficients[i + 1] * instance.value(i);
		}
		
	 	return res;
	}
	
	/**
	 * Calculates the total squared error over the data on a linear regression
	 * predictor with weights given by m_coefficients.
     *
	 * @param testData
	 * @return
	 * @throws Exception
	 */
	public double calculateMSE(Instances data) throws Exception {
		
		double res;
		double sum = 0;
		
		for (int i = 0; i < data.size(); i++) {
			res = 0;
			res = res + regressionPrediction(data.get(i));
			res = res - data.get(i).value(m_ClassIndex);
			res = Math.pow(res, 2);
			sum = sum + res;
		}
		
		return sum / (2 * data.size());
	}
	
	public double getAlpha() {
		return m_alpha;
	}
	
	public double getError() {
		return m_bestError;
	}
    
    @Override
	public double classifyInstance(Instance arg0) throws Exception {
		// Don't change
		return 0;
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// Don't change
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// Don't change
		return null;
	}
}
