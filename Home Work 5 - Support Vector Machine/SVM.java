package HomeWork5;

import HomeWork5.MainHW5.KernelType;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instances;
import weka.core.Instance;

public class SVM {
	
	Instances instances;
	public SMO m_smo;
	private KernelType kernelType;
	private double kernel;
	private double TPR;
	private double FPR;

	//Empty constructor.
	public SVM() {
		this.m_smo = new SMO();
	}
	
	public void buildClassifier(Instances instances) throws Exception{
		this.instances = instances;
		m_smo.buildClassifier(instances);
	}
	
	public int[] calcConfusion(Instances instances) throws Exception{
		//positive == recurrence-events.
		int TP = 0, FP = 0, TN = 0, FN = 0;
		double prediction;
		
		for (Instance instance : instances) {
			prediction = m_smo.classifyInstance(instance);
			
			if (prediction == 1) {
				
				if (prediction == instance.classValue()) {
					
					TP++;
					
				} else {
					FP++;
					
				}
			} else {
				
				if (prediction == instance.classValue()) {
					TN++;
					
				} else {
					FN++;
					
				}
			}
		}
		int[] confusionMatrix = {TP, FP, TN, FN};
		
		return confusionMatrix;
	}
	
	public void calcPositiveRates(Instances test) throws Exception {
		int[] confusionMatrix = calcConfusion(test);
		//System.err.println("matrix: " + confusionMatrix[0] + ", " + confusionMatrix[1] + ", " + confusionMatrix[2] + ", " + confusionMatrix[3]);
		
		TPR = (double) confusionMatrix[0] / (confusionMatrix[0] + confusionMatrix[3]);
		FPR = (double) confusionMatrix[1] / (confusionMatrix[1] + confusionMatrix[2]);
		
	}
	
	//-----------------------------------------------------------
	//Getters and Setters.
	//-----------------------------------------------------------

	public KernelType getKernelType() {
		return kernelType;
	}

	public void setKernelType(KernelType kernelType) {
		this.kernelType = kernelType;
	}

	public double getKernel() {
		return kernel;
	}

	//not good signature.
	public void setKernel(double kernel) {
		this.kernel = kernel;
	}
	
	//Mandatory.
	public void setKernel(PolyKernel kernel) {
		m_smo.setKernel(kernel);
	}
	
	public void setKernel(RBFKernel kernel) {
		m_smo.setKernel(kernel);
	}

	public double getFPR() {
		return FPR;
	}

	public double getTPR() {
		return TPR;
	}

	public double getC() {
		return m_smo.getC();
	}

	public void setC(double c) {
		m_smo.setC(c);
	}
	
	

}
