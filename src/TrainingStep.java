public record TrainingStep(
        ForwardPass beforeTraining,
        ForwardPass afterTraining,
        double expectedOutput,
        double error,
        double outputDelta,
        double[] hiddenErrors,
        double[] hiddenDeltas,
        double learningRate,
        double[] oldWeights,
        double[] weightChanges,
        double[] newWeights,
        double oldOutputBias,
        double outputBiasChange,
        double newOutputBias,
        double[] oldHiddenBiases,
        double[] hiddenBiasChanges,
        double[] newHiddenBiases) {

    public TrainingStep {
        hiddenErrors = hiddenErrors.clone();
        hiddenDeltas = hiddenDeltas.clone();
        oldWeights = oldWeights.clone();
        weightChanges = weightChanges.clone();
        newWeights = newWeights.clone();
        oldHiddenBiases = oldHiddenBiases.clone();
        hiddenBiasChanges = hiddenBiasChanges.clone();
        newHiddenBiases = newHiddenBiases.clone();
    }

    @Override public double[] hiddenErrors() { return hiddenErrors.clone(); }
    @Override public double[] hiddenDeltas() { return hiddenDeltas.clone(); }
    @Override public double[] oldWeights() { return oldWeights.clone(); }
    @Override public double[] weightChanges() { return weightChanges.clone(); }
    @Override public double[] newWeights() { return newWeights.clone(); }
    @Override public double[] oldHiddenBiases() { return oldHiddenBiases.clone(); }
    @Override public double[] hiddenBiasChanges() { return hiddenBiasChanges.clone(); }
    @Override public double[] newHiddenBiases() { return newHiddenBiases.clone(); }
}