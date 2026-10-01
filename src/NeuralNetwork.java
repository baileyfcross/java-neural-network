import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public final class NeuralNetwork {
    public static final int INPUT_COUNT = 2;
    public static final int HIDDEN_COUNT = 3;

    private final long seed;
    private final double learningRate;
    private final Neuron[] inputs = {new Neuron("X1", 0), new Neuron("X2", 0)};
    private final Neuron[] hidden = {new Neuron("H1", 0), new Neuron("H2", 0), new Neuron("H3", 0)};
    private final Neuron output = new Neuron("OUT", 0);
    private final List<Connection> connections = new ArrayList<>();
    private final Connection[][] inputToHidden = new Connection[INPUT_COUNT][HIDDEN_COUNT];
    private final Connection[] hiddenToOutput = new Connection[HIDDEN_COUNT];

    public NeuralNetwork(long seed) { this(seed, 1.2); }

    public NeuralNetwork(long seed, double learningRate) {
        this.seed = seed;
        this.learningRate = learningRate;
        createConnections();
        reset();
    }

    private void createConnections() {
        for (int inputIndex = 0; inputIndex < INPUT_COUNT; inputIndex++) {
            for (int hiddenIndex = 0; hiddenIndex < HIDDEN_COUNT; hiddenIndex++) {
                inputToHidden[inputIndex][hiddenIndex] = new Connection(inputs[inputIndex], hidden[hiddenIndex], 0);
                connections.add(inputToHidden[inputIndex][hiddenIndex]);
            }
        }
        for (int hiddenIndex = 0; hiddenIndex < HIDDEN_COUNT; hiddenIndex++) {
            hiddenToOutput[hiddenIndex] = new Connection(hidden[hiddenIndex], output, 0);
            connections.add(hiddenToOutput[hiddenIndex]);
        }
    }

    public void reset() {
        Random random = new Random(seed);
        for (Neuron neuron : hidden) {
            neuron.setBias(randomWeight(random));
            neuron.clearCalculation();
        }
        output.setBias(randomWeight(random));
        output.clearCalculation();
        for (Connection connection : connections) connection.setWeight(randomWeight(random));
        for (Neuron input : inputs) input.clearCalculation();
    }

    public ForwardPass forward(double[] inputValues) {
        validateInputs(inputValues);
        for (int index = 0; index < INPUT_COUNT; index++) inputs[index].setCalculation(0, inputValues[index]);

        for (int hiddenIndex = 0; hiddenIndex < HIDDEN_COUNT; hiddenIndex++) {
            double weightedSum = hidden[hiddenIndex].getBias();
            for (int inputIndex = 0; inputIndex < INPUT_COUNT; inputIndex++) {
                Connection connection = inputToHidden[inputIndex][hiddenIndex];
                connection.calculateContribution();
                weightedSum += connection.getContribution();
            }
            hidden[hiddenIndex].setCalculation(weightedSum, sigmoid(weightedSum));
        }

        double outputWeightedSum = output.getBias();
        for (Connection connection : hiddenToOutput) {
            connection.calculateContribution();
            outputWeightedSum += connection.getContribution();
        }
        output.setCalculation(outputWeightedSum, sigmoid(outputWeightedSum));
        return snapshot(inputValues);
    }

    public TrainingResult trainStep(double[] inputValues, double expectedOutput) {
        ForwardPass before = forward(inputValues);
        double outputDelta = (expectedOutput - output.getActivation()) * sigmoidDerivative(output.getActivation());
        double[] hiddenDeltas = new double[HIDDEN_COUNT];
        for (int hiddenIndex = 0; hiddenIndex < HIDDEN_COUNT; hiddenIndex++) {
            double hiddenError = outputDelta * hiddenToOutput[hiddenIndex].getWeight();
            hiddenDeltas[hiddenIndex] = hiddenError * sigmoidDerivative(hidden[hiddenIndex].getActivation());
        }

        for (int hiddenIndex = 0; hiddenIndex < HIDDEN_COUNT; hiddenIndex++) {
            Connection connection = hiddenToOutput[hiddenIndex];
            connection.setWeight(connection.getWeight() + learningRate * outputDelta * hidden[hiddenIndex].getActivation());
        }
        output.setBias(output.getBias() + learningRate * outputDelta);
        for (int inputIndex = 0; inputIndex < INPUT_COUNT; inputIndex++) {
            for (int hiddenIndex = 0; hiddenIndex < HIDDEN_COUNT; hiddenIndex++) {
                Connection connection = inputToHidden[inputIndex][hiddenIndex];
                connection.setWeight(connection.getWeight()
                        + learningRate * hiddenDeltas[hiddenIndex] * inputs[inputIndex].getActivation());
            }
        }
        for (int hiddenIndex = 0; hiddenIndex < HIDDEN_COUNT; hiddenIndex++) {
            hidden[hiddenIndex].setBias(hidden[hiddenIndex].getBias() + learningRate * hiddenDeltas[hiddenIndex]);
        }
        return new TrainingResult(before, expectedOutput, squaredError(before.outputActivation(), expectedOutput));
    }

    public double trainEpoch(double[][] trainingInputs, double[] expectedOutputs) {
        if (trainingInputs.length != expectedOutputs.length) {
            throw new IllegalArgumentException("Each training input needs one expected output.");
        }
        double totalError = 0;
        for (int index = 0; index < trainingInputs.length; index++) {
            totalError += trainStep(trainingInputs[index], expectedOutputs[index]).error();
        }
        return totalError / trainingInputs.length;
    }

    public static double sigmoid(double value) { return 1.0 / (1.0 + Math.exp(-value)); }
    public Neuron[] getInputNeurons() { return inputs.clone(); }
    public Neuron[] getHiddenNeurons() { return hidden.clone(); }
    public Neuron getOutputNeuron() { return output; }
    public List<Connection> getConnections() { return List.copyOf(connections); }

    private ForwardPass snapshot(double[] inputValues) {
        List<ForwardPass.ConnectionState> states = connections.stream()
                .map(connection -> new ForwardPass.ConnectionState(
                        connection.getSource().getName(), connection.getDestination().getName(),
                        connection.getWeight(), connection.getContribution()))
                .toList();
        return new ForwardPass(inputValues,
                Arrays.stream(hidden).mapToDouble(Neuron::getWeightedSum).toArray(),
                Arrays.stream(hidden).mapToDouble(Neuron::getActivation).toArray(),
                output.getWeightedSum(), output.getActivation(), states);
    }

    private static double randomWeight(Random random) { return (random.nextDouble() * 2 - 1) * 1.5; }
    private static double sigmoidDerivative(double activation) { return activation * (1 - activation); }
    private static double squaredError(double actual, double expected) { return 0.5 * Math.pow(expected - actual, 2); }

    private static void validateInputs(double[] inputValues) {
        if (inputValues == null || inputValues.length != INPUT_COUNT) {
            throw new IllegalArgumentException("The network needs exactly two input values.");
        }
    }

    public record TrainingResult(ForwardPass pass, double expectedOutput, double error) { }
}