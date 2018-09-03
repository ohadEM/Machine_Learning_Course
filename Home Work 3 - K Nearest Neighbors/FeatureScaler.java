package HomeWork3;
//package HomeWork3;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

public class FeatureScaler {
	/**
	 * Returns a scaled version (using standarized normalization) of the given dataset.
	 * @param instances The original dataset.
	 * @return A scaled instances object.
	 * @throws Exception 
	 */
	public static Instances scaleData(Instances instances) throws Exception {
		
		// Standardize
		Standardize std = new Standardize();
		std.setInputFormat(instances);
		Instances scaleData = Filter.useFilter(instances, std);
		return scaleData;
	}
}