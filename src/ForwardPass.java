import java.util.List;

public record ForwardPass(
        double[] inputs,
        double[] hiddenWeightedSums,
        double[] hiddenActivations,
        double outputWeightedSum,
        double outputActivation,
        List<ConnectionState> connections) {

    public ForwardPass {
        inputs = inputs.clone();
        hiddenWeightedSums = hiddenWeightedSums.clone();
        hiddenActivations = hiddenActivations.clone();
        connections = List.copyOf(connections);
    }

    @Override public double[] inputs() { return inputs.clone(); }
    @Override public double[] hiddenWeightedSums() { return hiddenWeightedSums.clone(); }
    @Override public double[] hiddenActivations() { return hiddenActivations.clone(); }
    public int binaryPrediction() { return outputActivation >= 0.5 ? 1 : 0; }

    public record ConnectionState(
            String sourceName,
            String destinationName,
            double weight,
            double contribution) {
    }
}