public final class Connection {
    private final Neuron source;
    private final Neuron destination;
    private double weight;
    private double contribution;

    public Connection(Neuron source, Neuron destination, double weight) {
        this.source = source;
        this.destination = destination;
        this.weight = weight;
        destination.addIncomingConnection(this);
    }

    public Neuron getSource() { return source; }
    public Neuron getDestination() { return destination; }
    public double getWeight() { return weight; }
    public void setWeight(double weight) { this.weight = weight; }
    public double getContribution() { return contribution; }

    void calculateContribution() {
        contribution = source.getActivation() * weight;
    }
}