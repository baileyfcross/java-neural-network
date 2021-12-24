import java.util.Random;
/**
 * 
 *
 * @author (Bailey Cross)
 * @version (Winter 2021)
 */
public class ConnectionMatrix
{
    private double[] weights;
    private int size;
    
    /**
     * Constructor for objects of class ConnectionMatrix
     */
    public ConnectionMatrix(int size)
    {
        weights = new double[size];
        this.size = size;
    }

    public void initRandomWeights()
    {
        Random rand = new Random();
        
        for(int i = 0; i < weights.length; i++){
            weights[i] = rand.nextDouble() * 2 - 1;
        }       
    }
    
    public void initWeights(double[] weights){
        
        for(int i = 0; i < weights.length; i++){
            this.weights[i] = weights[i];
        }
        
    }
    
    public void updateWeights(int pos, double deltaWeight){       
        weights[pos] += deltaWeight;
    }
    
    public void changeMatrix(int size){
        if(size > this.size ){
            double[] tempWeights = weights;
            weights = new double[size];
            for(int i = 0; i < size; i++){
                weights[i] = tempWeights[i];
            }
            this.size = size;
        }
        else if(size < this.size){
            double[] tempWeights = new double[size];
            for(int i = 0; i < size; i++){
                tempWeights[i] = weights[i];
            }
            weights = tempWeights;
        }
        else{
            return;
        }
    }
    
    public double[] getWeightMatrix(){
        return weights;
    }
    
    public int getMatrixSize(){
        return size;
    }
}
