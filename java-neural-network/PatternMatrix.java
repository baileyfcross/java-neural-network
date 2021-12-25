import java.io.*;
import java.util.*;
/**
 * Write a description of class PatternMatrix here.
 *
 * @author (Bailey Cross)
 * @version (Winter 2021)
 */
public class PatternMatrix
{
    // instance variables - replace the example below with your own
    private double[][] inputPatterns;
    private double[][] outputPatterns;
    private int inputPLength;
    private int outputPLength;
    private int numOfPatterns;

    /**
     * Constructor for objects of class PatternMatrix
     */
    public PatternMatrix(File pFile, int inputPLength, int outputPLength, int numOfPatterns) 
    throws FileNotFoundException
    {
        this.inputPLength = inputPLength;
        this.outputPLength = outputPLength;
        this.numOfPatterns = numOfPatterns;
        inputPatterns = new double[numOfPatterns][inputPLength];
        outputPatterns = new double[numOfPatterns][outputPLength];
        readFile(pFile);
        
    }

    /**
     * An example of a method - replace this comment with your own
     *
     * @param  y  a sample parameter for a method
     * @return    the sum of x and y
     */
    public void readFile(File pFile) throws FileNotFoundException
    {   
        Scanner sc = new Scanner(pFile);
        sc.nextLine();
        sc.nextLine();
        
        for(int i = 0; i < numOfPatterns;i++){
            String tempIn = sc.nextLine();
            String tempOut = sc.nextLine();
            sc.nextLine();
            
            StringTokenizer inputTok = new StringTokenizer(tempIn, ",");
            StringTokenizer outputTok = new StringTokenizer(tempOut, ",");
            
            for(int j = 0; j < inputPLength; j++){
                inputPatterns[i][j] = Double.parseDouble(inputTok.nextToken());
            }
            for(int k = 0; k < outputPLength; k++){
                inputPatterns[i][k] = Double.parseDouble(outputTok.nextToken());
            }
        }
    }
    
    
    public double[] getInputPatterns(int pIndex){
        return inputPatterns[pIndex];
    }
    
    public double[] getOutputPatterns(int pIndex){
        return outputPatterns[pIndex];
    }
    
    public int getIntputPLength(){
        return inputPLength;
    }
    
    public int getOutputPLength(){
        return outputPLength;
    }
    
    public int getNumOfPatterns(){
        return numOfPatterns;
    }
}
