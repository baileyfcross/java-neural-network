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
        TrainingStep step = prepareTrainingStep(inputValues, expectedOutput);
        applyTrainingStep(step);
        return new TrainingResult(step.beforeTraining(), expectedOutput, step.error());
    }

    public TrainingStep prepareTrainingStep(double[] inputValues, double expectedOutput) {
        // Capture the complete gradient before changing any live network values.
        ForwardPass before = forward(inputValues);
        double outputActivation = before.outputActivation();
        double outputDelta = (expectedOutput - outputActivation) * sigmoidDerivative(outputActivation);
        double[] hiddenErrors = new double[HIDDEN_COUNT];
        double[] hiddenDeltas = new double[HIDDEN_COUNT];
        for (int hiddenIndex = 0; hiddenIndex < HIDDEN_COUNT; hiddenIndex++) {
            double hiddenError = outputDelta * hiddenToOutput[hiddenIndex].getWeight();
            hiddenErrors[hiddenIndex] = hiddenError;
            hiddenDeltas[hiddenIndex] = hiddenError
                    * sigmoidDerivative(before.hiddenActivations()[hiddenIndex]);
        }

        double[] oldWeights = connections.stream().mapToDouble(Connection::getWeight).toArray();
        double[] weightChanges = new double[connections.size()];
        for (int inputIndex = 0; inputIndex < INPUT_COUNT; inputIndex++) {
            for (int hiddenIndex = 0; hiddenIndex < HIDDEN_COUNT; hiddenIndex++) {
                int connectionIndex = inputIndex * HIDDEN_COUNT + hiddenIndex;
                weightChanges[connectionIndex] = learningRate * hiddenDeltas[hiddenIndex]
                        * before.inputs()[inputIndex];
            }
        }
        for (int hiddenIndex = 0; hiddenIndex < HIDDEN_COUNT; hiddenIndex++) {
            int connectionIndex = INPUT_COUNT * HIDDEN_COUNT + hiddenIndex;
            weightChanges[connectionIndex] = learningRate * outputDelta
                    * before.hiddenActivations()[hiddenIndex];
        }
        double[] newWeights = addArrays(oldWeights, weightChanges);

        double oldOutputBias = output.getBias();
        double outputBiasChange = learningRate * outputDelta;
        double newOutputBias = oldOutputBias + outputBiasChange;
        double[] oldHiddenBiases = Arrays.stream(hidden).mapToDouble(Neuron::getBias).toArray();
        double[] hiddenBiasChanges = multiply(hiddenDeltas, learningRate);
        double[] newHiddenBiases = addArrays(oldHiddenBiases, hiddenBiasChanges);

        ForwardPass after = forwardWithValues(inputValues, newWeights, newOutputBias, newHiddenBiases);
        return new TrainingStep(before, after, expectedOutput, squaredError(outputActivation, expectedOutput), outputDelta, hiddenErrors,
                hiddenDeltas, learningRate, oldWeights, weightChanges, newWeights, oldOutputBias,
                outputBiasChange, newOutputBias, oldHiddenBiases, hiddenBiasChanges, newHiddenBiases);
    }

    public ForwardPass applyTrainingStep(TrainingStep step) {
        applySnapshotValues(step.newWeights(), step.newOutputBias(), step.newHiddenBiases());
        return forward(step.beforeTraining().inputs());
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
    public double getLearningRate() { return learningRate; }
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

    private void applySnapshotValues(double[] weights, double outputBias, double[] hiddenBiases) {
        for (int index = 0; index < connections.size(); index++) connections.get(index).setWeight(weights[index]);
        output.setBias(outputBias);
        for (int index = 0; index < hidden.length; index++) hidden[index].setBias(hiddenBiases[index]);
    }

    private ForwardPass forwardWithValues(double[] inputValues, double[] weights,
            double outputBias, double[] hiddenBiases) {
        double[] hiddenWeightedSums = new double[HIDDEN_COUNT];
        double[] hiddenActivations = new double[HIDDEN_COUNT];
        List<ForwardPass.ConnectionState> states = new ArrayList<>();
        for (int hiddenIndex = 0; hiddenIndex < HIDDEN_COUNT; hiddenIndex++) {
            hiddenWeightedSums[hiddenIndex] = hiddenBiases[hiddenIndex];
            for (int inputIndex = 0; inputIndex < INPUT_COUNT; inputIndex++) {
                int connectionIndex = inputIndex * HIDDEN_COUNT + hiddenIndex;
                double contribution = inputValues[inputIndex] * weights[connectionIndex];
                hiddenWeightedSums[hiddenIndex] += contribution;
                states.add(new ForwardPass.ConnectionState(
                        inputs[inputIndex].getName(), hidden[hiddenIndex].getName(),
                        weights[connectionIndex], contribution));
            }
            hiddenActivations[hiddenIndex] = sigmoid(hiddenWeightedSums[hiddenIndex]);
        }

        double outputWeightedSum = outputBias;
        for (int hiddenIndex = 0; hiddenIndex < HIDDEN_COUNT; hiddenIndex++) {
            int connectionIndex = INPUT_COUNT * HIDDEN_COUNT + hiddenIndex;
            double contribution = hiddenActivations[hiddenIndex] * weights[connectionIndex];
            outputWeightedSum += contribution;
            states.add(new ForwardPass.ConnectionState(
                    hidden[hiddenIndex].getName(), output.getName(), weights[connectionIndex], contribution));
        }
        return new ForwardPass(inputValues, hiddenWeightedSums, hiddenActivations,
                outputWeightedSum, sigmoid(outputWeightedSum), states);
    }

    private static double[] addArrays(double[] first, double[] second) {
        double[] result = first.clone();
        for (int index = 0; index < result.length; index++) result[index] += second[index];
        return result;
    }

    private static double[] multiply(double[] values, double factor) {
        double[] result = values.clone();
        for (int index = 0; index < result.length; index++) result[index] *= factor;
        return result;
    }

    private static void validateInputs(double[] inputValues) {
        if (inputValues == null || inputValues.length != INPUT_COUNT) {
            throw new IllegalArgumentException("The network needs exactly two input values.");
        }
    }

    public record TrainingResult(ForwardPass pass, double expectedOutput, double error) { }
}