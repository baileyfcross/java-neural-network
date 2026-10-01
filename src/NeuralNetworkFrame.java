import java.awt.BorderLayout;
import java.util.List;
import javax.swing.JFrame;
import javax.swing.Timer;

public final class NeuralNetworkFrame extends JFrame {
    private static final double[][] XOR_INPUTS = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    private static final double[] XOR_OUTPUTS = {0, 1, 1, 0};
    private final NeuralNetwork network = new NeuralNetwork(2025);
    private final NetworkPanel networkPanel = new NetworkPanel();
    private final ControlPanel controls = new ControlPanel();
    private boolean automaticTraining;
    private boolean manualTraining;
    private int automaticSampleIndex;
    private double epochErrorTotal;
    private Timer automaticContinueTimer;
    private Timer manualPauseTimer;
    private int epoch;
    private double error;

    public NeuralNetworkFrame() {
        super("Java Neural Network | XOR Visualization");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(1200, 750);
        setLocationByPlatform(true);
        setLayout(new BorderLayout());
        add(networkPanel, BorderLayout.CENTER);
        add(controls, BorderLayout.EAST);
        controls.getRunForward().addActionListener(event -> runForward());
        controls.getTrainStep().addActionListener(event -> trainOneStep());
        controls.getTrainAutomatically().addActionListener(event -> trainAutomatically());
        controls.getStopTraining().addActionListener(event -> stopTraining());
        controls.getReset().addActionListener(event -> resetNetwork());
    }

    private void runForward() {
        ForwardPass pass = network.forward(selectedInputs());
        networkPanel.animate(pass, controls.getSpeed());
        showInformation("Current layer: input -> hidden -> output\n" + formatInputs()
                + "\nExpected: " + controls.getExpected() + "\n\nRaw Output: " + format(pass.outputActivation())
                + "\nPrediction: " + pass.binaryPrediction()
                + "\n\nWeighted sums and contributions appear during each stage.");
    }

    private void trainOneStep() {
        if (automaticTraining || manualTraining) return;
        double[] inputs = selectedInputs();
        double expected = controls.getExpected();
        TrainingStep step = network.prepareTrainingStep(inputs, expected);
        manualTraining = true;
        controls.setAutomaticTraining(true);
        showInformation("Forward Propagation Complete\n\n" + formatInputs()
            + "\nExpected: " + format(expected)
            + "\nNetwork Output: " + format(step.beforeTraining().outputActivation())
            + "\nPrediction: " + step.beforeTraining().binaryPrediction()
            + "\n\nError: " + format(expected) + " - " + format(step.beforeTraining().outputActivation())
            + " = " + format(expected - step.beforeTraining().outputActivation())
            + "\nLoss: 0.5 x (" + format(expected) + " - "
            + format(step.beforeTraining().outputActivation()) + ")^2 = " + format(step.error())
            + "\n\nPreparing output gradient...");
        networkPanel.animate(step.beforeTraining(), controls.getSpeed(), () -> beginManualBackpropagation(step));
    }

    private void trainAutomatically() {
        if (automaticTraining) return;
        automaticTraining = true;
        automaticSampleIndex = 0;
        epochErrorTotal = 0;
        controls.setAutomaticTraining(true);
        runNextAutomaticExample();
    }

    private void stopTraining() {
        if (manualTraining) {
            manualTraining = false;
            if (manualPauseTimer != null) {
                manualPauseTimer.stop();
                manualPauseTimer = null;
            }
            networkPanel.stopAnimation();
            controls.setAutomaticTraining(false);
            showInformation("Train One Step stopped.\n\nNo weight update was applied.");
            return;
        }
        if (!automaticTraining) return;
        automaticTraining = false;
        if (automaticContinueTimer != null) {
            automaticContinueTimer.stop();
            automaticContinueTimer = null;
        }
        networkPanel.stopAnimation();
        controls.setAutomaticTraining(false);
        showInformation("Automatic training stopped.\n\nEpoch: " + epoch
                + "\nCurrent Error: " + format(error) + "\n\nWeights were kept.");
    }

    private void resetNetwork() {
        stopTraining();
        network.reset();
        epoch = 0;
        error = 0;
        automaticSampleIndex = 0;
        epochErrorTotal = 0;
        controls.setAutomaticTraining(false);
        networkPanel.stopAnimation();
        showInformation("Network reset with deterministic weights.\n\nChoose XOR inputs and run a forward pass.");
    }

    private void beginManualBackpropagation(TrainingStep step) {
        if (!manualTraining) return;
        showInformation("Forward Propagation Complete\n\nExpected: " + format(step.expectedOutput())
                + "\nNetwork Output: " + format(step.beforeTraining().outputActivation())
                + "\nPrediction: " + step.beforeTraining().binaryPrediction()
                + "\n\nError: " + format(step.expectedOutput()) + " - "
                + format(step.beforeTraining().outputActivation()) + " = "
                + format(step.expectedOutput() - step.beforeTraining().outputActivation())
                + "\nLoss: 0.5 x error^2 = " + format(step.error())
                + "\n\nOutput delta begins in a moment...");
        manualPauseTimer = new Timer(650, event -> {
            manualPauseTimer = null;
            if (!manualTraining) return;
            networkPanel.animateBackpropagation(step, controls.getSpeed(),
                    stage -> showBackpropagationStage(step, stage),
                    () -> applyManualTrainingStep(step));
        });
        manualPauseTimer.setRepeats(false);
        manualPauseTimer.start();
    }

    private void showBackpropagationStage(TrainingStep step, int stage) {
        if (!manualTraining) return;
        if (stage == 1) {
            showInformation("Output Delta\n\n(expected - output) x output x (1 - output)\n\n("
                    + format(step.expectedOutput()) + " - " + format(step.beforeTraining().outputActivation())
                    + ") x " + format(step.beforeTraining().outputActivation()) + " x (1 - "
                    + format(step.beforeTraining().outputActivation()) + ")\n\n= "
                    + format(step.outputDelta())
                    + "\n\nThe output gradient is ready to travel backward.");
        } else if (stage == 2) {
            showInformation("Backpropagating Gradient\n\nOUT -> H1\nOUT -> H2\nOUT -> H3\n\nPurple pulses show error gradient, not activation data.\n\nOutput Delta: "
                    + format(step.outputDelta()));
        } else if (stage == 3) {
            showHiddenGradientInformation(step);
        } else if (stage == 4) {
            showInformation("Backpropagating Gradient\n\nH1 -> X1, X2\nH2 -> X1, X2\nH3 -> X1, X2\n\nThe hidden deltas determine the input-to-hidden weight changes.");
        } else if (stage >= 5) {
            showWeightUpdateInformation(step);
        }
    }

    private void showHiddenGradientInformation(TrainingStep step) {
        StringBuilder text = new StringBuilder("Hidden Layer Gradients\n\n");
        double[] activations = step.beforeTraining().hiddenActivations();
        for (int index = 0; index < NeuralNetwork.HIDDEN_COUNT; index++) {
            ForwardPass.ConnectionState connection = step.beforeTraining().connections()
                    .get(NeuralNetwork.INPUT_COUNT * NeuralNetwork.HIDDEN_COUNT + index);
            double activation = activations[index];
            text.append("H").append(index + 1).append("\n")
                    .append("Output Delta: ").append(format(step.outputDelta())).append("\n")
                    .append("Weight H").append(index + 1).append(" -> OUT: ")
                    .append(format(connection.weight())).append("\n")
                    .append("Hidden Error: ").append(format(step.outputDelta())).append(" x ")
                    .append(format(connection.weight())).append(" = ")
                    .append(format(step.hiddenErrors()[index])).append("\n")
                    .append("Activation: ").append(format(activation)).append("\n")
                    .append("Sigmoid Derivative: ").append(format(activation)).append(" x (1 - ")
                    .append(format(activation)).append(")\n")
                    .append("Hidden Delta: ").append(format(step.hiddenDeltas()[index])).append("\n\n");
        }
        showInformation(text.toString());
    }

    private void showWeightUpdateInformation(TrainingStep step) {
        StringBuilder text = new StringBuilder("Weight Updates\n\n");
        List<ForwardPass.ConnectionState> connections = step.beforeTraining().connections();
        double[] oldWeights = step.oldWeights();
        double[] changes = step.weightChanges();
        double[] newWeights = step.newWeights();
        for (int index = 0; index < connections.size(); index++) {
            ForwardPass.ConnectionState connection = connections.get(index);
            text.append(connection.sourceName()).append(" -> ").append(connection.destinationName()).append("\n")
                    .append("Old Weight: ").append(format(oldWeights[index])).append("\n");
            if (index < NeuralNetwork.INPUT_COUNT * NeuralNetwork.HIDDEN_COUNT) {
                int inputIndex = index / NeuralNetwork.HIDDEN_COUNT;
                int hiddenIndex = index % NeuralNetwork.HIDDEN_COUNT;
                text.append("Change: ").append(format(step.learningRate())).append(" x ")
                        .append(format(step.hiddenDeltas()[hiddenIndex])).append(" x ")
                        .append(format(step.beforeTraining().inputs()[inputIndex])).append(" = ")
                        .append(format(changes[index])).append("\n");
            } else {
                int hiddenIndex = index - NeuralNetwork.INPUT_COUNT * NeuralNetwork.HIDDEN_COUNT;
                text.append("Change: ").append(format(step.learningRate())).append(" x ")
                        .append(format(step.outputDelta())).append(" x ")
                        .append(format(step.beforeTraining().hiddenActivations()[hiddenIndex])).append(" = ")
                        .append(format(changes[index])).append("\n");
            }
            text.append("New Weight: ").append(format(newWeights[index])).append("\n\n");
        }
        text.append("\nBias Updates\nOUT: ").append(format(step.oldOutputBias())).append(" -> ")
                .append(format(step.newOutputBias())).append("\n");
        for (int index = 0; index < NeuralNetwork.HIDDEN_COUNT; index++) {
            text.append("H").append(index + 1).append(": ")
                    .append(format(step.oldHiddenBiases()[index])).append(" -> ")
                    .append(format(step.newHiddenBiases()[index])).append("\n");
        }
        text.append("\nPurple connections will reveal the new values after this stage.");
        showInformation(text.toString());
    }

    private void applyManualTrainingStep(TrainingStep step) {
        if (!manualTraining) return;
        ForwardPass after = network.applyTrainingStep(step);
        error = step.error();
        networkPanel.showTrainingChange(step.beforeTraining(), after);
        manualTraining = false;
        controls.setAutomaticTraining(false);
        showInformation("Training Step Complete\n\nBefore Training: "
                + format(step.beforeTraining().outputActivation())
                + "\nAfter Training: " + format(after.outputActivation())
                + "\nExpected: " + format(step.expectedOutput())
                + "\n\nOutput Delta: " + format(step.outputDelta())
                + "\nWeights and biases were updated once.\nPurple connections show the changes.");
    }

    private void runNextAutomaticExample() {
        if (!automaticTraining) return;

        int sampleIndex = automaticSampleIndex;
        double[] inputs = XOR_INPUTS[sampleIndex].clone();
        double expected = XOR_OUTPUTS[sampleIndex];
        controls.setExample((int) inputs[0], (int) inputs[1], (int) expected);

        ForwardPass pass = network.forward(inputs);
        showInformation("Automatic Training\n\nEpoch: " + (epoch + 1)
                + "\nExample: " + (sampleIndex + 1) + " / " + XOR_INPUTS.length
                + "\n\nInputs: [" + (int) inputs[0] + ", " + (int) inputs[1] + "]"
                + "\nExpected: " + (int) expected
                + "\n\nRaw Output: " + format(pass.outputActivation())
                + "\nPrediction: " + pass.binaryPrediction()
                + "\nSample Error: waiting for training step"
                + "\n\nForward pass is animating...");
        networkPanel.animate(pass, controls.getSpeed(),
                () -> trainAutomaticExample(inputs, expected, sampleIndex));
    }

    private void trainAutomaticExample(double[] inputs, double expected, int sampleIndex) {
        if (!automaticTraining) return;

        NeuralNetwork.TrainingResult result = network.trainStep(inputs, expected);
        ForwardPass after = network.forward(inputs);
        error = result.error();
        epochErrorTotal += error;
        networkPanel.showTrainingChange(result.pass(), after);
        showInformation("Automatic Training\n\nEpoch: " + (epoch + 1)
                + "\nExample: " + (sampleIndex + 1) + " / " + XOR_INPUTS.length
                + "\n\nInputs: [" + (int) inputs[0] + ", " + (int) inputs[1] + "]"
                + "\nExpected: " + (int) expected
                + "\n\nRaw Output: " + format(result.pass().outputActivation())
                + "\nPrediction: " + result.pass().binaryPrediction()
                + "\nSample Error: " + format(error)
                + "\nEpoch Error (running): " + format(epochErrorTotal / (sampleIndex + 1))
                + "\n\nPurple connections changed weight.");

        if (sampleIndex == XOR_INPUTS.length - 1) {
            epoch++;
            error = epochErrorTotal / XOR_INPUTS.length;
            epochErrorTotal = 0;
            automaticSampleIndex = 0;
            if (error < 0.005 || epoch >= 10000) {
                finishAutomaticTraining(error < 0.005
                        ? "Target error reached."
                        : "Maximum epoch count reached.");
                return;
            }
            showInformation("Automatic Training\n\nEpoch: " + epoch + " complete"
                    + "\nEpoch Error: " + format(error)
                    + "\n\nStarting next epoch...");
        } else {
            automaticSampleIndex = sampleIndex + 1;
        }
        scheduleNextAutomaticExample();
    }

    private void scheduleNextAutomaticExample() {
        if (!automaticTraining) return;
        automaticContinueTimer = new Timer(700, event -> {
            automaticContinueTimer = null;
            runNextAutomaticExample();
        });
        automaticContinueTimer.setRepeats(false);
        automaticContinueTimer.start();
    }

    private void finishAutomaticTraining(String reason) {
        automaticTraining = false;
        controls.setAutomaticTraining(false);
        showInformation("Automatic Training Complete\n\n" + reason
                + "\nEpoch: " + epoch + "\nEpoch Error: " + format(error)
                + "\n\nWeights are ready to inspect.");
    }

    private double[] selectedInputs() { return new double[]{controls.getInputOne(), controls.getInputTwo()}; }
    private String formatInputs() { return "Inputs: [" + controls.getInputOne() + ", " + controls.getInputTwo() + "]"; }
    private void showInformation(String text) { controls.showInformation(text); }
    private static String format(double value) { return String.format("%.4f", value); }
}