import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.Test;

class NeuralNetworkTest {
    private static final double[][] XOR_INPUTS = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    private static final double[] XOR_OUTPUTS = {0, 1, 1, 0};

    @Test
    void sigmoidHasExpectedValues() {
        assertEquals(0.5, NeuralNetwork.sigmoid(0), 1e-9);
        assertEquals(0.8808, NeuralNetwork.sigmoid(2), 1e-4);
    }

    @Test
    void forwardPassContainsExpectedValues() {
        ForwardPass pass = new NeuralNetwork(2025).forward(new double[]{1, 0});
        assertEquals(2, pass.inputs().length);
        assertEquals(3, pass.hiddenActivations().length);
        assertTrue(pass.outputActivation() > 0 && pass.outputActivation() < 1);
        assertEquals(9, pass.connections().size());
    }

    @Test
    void sameSeedCreatesSameForwardPass() {
        ForwardPass first = new NeuralNetwork(42).forward(new double[]{1, 0});
        ForwardPass second = new NeuralNetwork(42).forward(new double[]{1, 0});
        assertEquals(first.outputActivation(), second.outputActivation(), 1e-12);
        assertEquals(first.connections(), second.connections());
    }

    @Test
    void trainingChangesWeights() {
        NeuralNetwork network = new NeuralNetwork(2025);
        double before = network.getConnections().get(0).getWeight();
        network.trainStep(new double[]{1, 0}, 1);
        assertNotEquals(before, network.getConnections().get(0).getWeight());
    }

    @Test
    void repeatedXorTrainingLowersErrorAndClassifiesPatterns() {
        NeuralNetwork network = new NeuralNetwork(2025);
        double initialError = network.trainEpoch(XOR_INPUTS, XOR_OUTPUTS);
        double error = initialError;
        for (int epoch = 0; epoch < 10000; epoch++) error = network.trainEpoch(XOR_INPUTS, XOR_OUTPUTS);
        assertTrue(error < initialError);
        for (int index = 0; index < XOR_INPUTS.length; index++) {
            assertEquals((int) XOR_OUTPUTS[index], network.forward(XOR_INPUTS[index]).binaryPrediction());
        }
    }
}