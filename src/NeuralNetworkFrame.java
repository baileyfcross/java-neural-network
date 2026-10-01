import java.awt.BorderLayout;
import javax.swing.JFrame;
import javax.swing.Timer;

public final class NeuralNetworkFrame extends JFrame {
    private static final double[][] XOR_INPUTS = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    private static final double[] XOR_OUTPUTS = {0, 1, 1, 0};
    private final NeuralNetwork network = new NeuralNetwork(2025);
    private final NetworkPanel networkPanel = new NetworkPanel();
    private final ControlPanel controls = new ControlPanel();
    private boolean automaticTraining;
    private int automaticSampleIndex;
    private double epochErrorTotal;
    private Timer automaticContinueTimer;
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
        double[] inputs = selectedInputs();
        NeuralNetwork.TrainingResult result = network.trainStep(inputs, controls.getExpected());
        ForwardPass after = network.forward(inputs);
        error = result.error();
        networkPanel.showTrainingChange(result.pass(), after);
        showInformation("One backpropagation step complete\n\n" + formatInputs()
                + "\nExpected: " + controls.getExpected()
                + "\nBefore output: " + format(result.pass().outputActivation())
                + "\nAfter output:  " + format(after.outputActivation())
                + "\nError: " + format(error) + "\n\nPurple connections changed weight.");
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