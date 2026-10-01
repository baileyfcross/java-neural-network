import java.awt.BorderLayout;
import java.util.concurrent.ExecutionException;
import javax.swing.JFrame;
import javax.swing.SwingWorker;

public final class NeuralNetworkFrame extends JFrame {
    private static final double[][] XOR_INPUTS = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    private static final double[] XOR_OUTPUTS = {0, 1, 1, 0};
    private final NeuralNetwork network = new NeuralNetwork(2025);
    private final NetworkPanel networkPanel = new NetworkPanel();
    private final ControlPanel controls = new ControlPanel();
    private SwingWorker<Void, TrainingUpdate> trainingWorker;
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
        if (trainingWorker != null && !trainingWorker.isDone()) return;
        controls.getTrainAutomatically().setEnabled(false);
        trainingWorker = new SwingWorker<>() {
            @Override protected Void doInBackground() {
                for (int currentEpoch = 1; currentEpoch <= 10000 && !isCancelled(); currentEpoch++) {
                    double currentError = network.trainEpoch(XOR_INPUTS, XOR_OUTPUTS);
                    if (currentEpoch % 25 == 0 || currentError < 0.005) publish(new TrainingUpdate(currentEpoch, currentError));
                    if (currentError < 0.005) break;
                }
                return null;
            }

            @Override protected void process(java.util.List<TrainingUpdate> updates) {
                TrainingUpdate update = updates.get(updates.size() - 1);
                epoch = update.epoch(); error = update.error();
                ForwardPass pass = network.forward(selectedInputs());
                networkPanel.showPass(pass);
                showInformation("Training XOR across all four examples\n\nEpoch: " + epoch
                        + "\nCurrent Error: " + format(error) + "\n\n" + formatInputs()
                        + "\nPrediction: " + pass.binaryPrediction());
            }

            @Override protected void done() {
                controls.getTrainAutomatically().setEnabled(true);
                try { get(); }
                catch (InterruptedException interrupted) { Thread.currentThread().interrupt(); }
                catch (ExecutionException ignored) { showInformation("Training stopped because an error occurred."); }
            }
        };
        trainingWorker.execute();
    }

    private void stopTraining() {
        if (trainingWorker != null) trainingWorker.cancel(true);
        controls.getTrainAutomatically().setEnabled(true);
    }

    private void resetNetwork() {
        stopTraining(); network.reset(); epoch = 0; error = 0; networkPanel.stopAnimation();
        showInformation("Network reset with deterministic weights.\n\nChoose XOR inputs and run a forward pass.");
    }

    private double[] selectedInputs() { return new double[]{controls.getInputOne(), controls.getInputTwo()}; }
    private String formatInputs() { return "Inputs: [" + controls.getInputOne() + ", " + controls.getInputTwo() + "]"; }
    private void showInformation(String text) { controls.showInformation(text); }
    private static String format(double value) { return String.format("%.4f", value); }
    private record TrainingUpdate(int epoch, double error) { }
}