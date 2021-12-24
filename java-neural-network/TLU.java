import java.lang.Math;
/**
 * 
 *
 * @author (Bailey Cross)
 * @version (Winter 2021)
 */
public class TLU
{
    // instance variables - replace the example below with your own
    private double weightedSum;
    private double activationT;
    private double bias;

    /**
     * Constructor for objects of class TLU
     */
    public TLU()
    {     
        weightedSum = 0;
        activationT = 0;
    }

    
    public void calculateWeightedSum(double[] inputs,double[] weights, double bias)
    {
        if(inputs.length != weights.length){
             return;   
        }
        for(int i = 0; i < inputs.length; i++){
            weightedSum += inputs[i] * weights[i];
        }
        weightedSum += bias;
    }
    
    public void calculateSigmoidFunction(){
        activationT = 1 / (1 + Math.pow(weightedSum, 2));           
    }
    
    public double getWeightedSum(){
        return weightedSum;
    }
    
    public double getActivationT(){
        return activationT;
    }
    
    public double getBias(){
        return bias;
    }
    
    public double SG(double[] inputs, double[] weights, double bias){
        calculateWeightedSum(inputs,weights, bias);
        calculateSigmoidFunction();
        return activationT;   
    }
}
