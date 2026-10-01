import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class Neuron {
    private final String name;
    private final List<Connection> incomingConnections = new ArrayList<>();
    private double bias;
    private double weightedSum;
    private double activation;

    public Neuron(String name, double bias) {
        this.name = name;
        this.bias = bias;
    }

    public String getName() { return name; }
    public double getBias() { return bias; }
    public void setBias(double bias) { this.bias = bias; }
    public double getWeightedSum() { return weightedSum; }
    public double getActivation() { return activation; }

    public List<Connection> getIncomingConnections() {
        return Collections.unmodifiableList(incomingConnections);
    }

    void addIncomingConnection(Connection connection) { incomingConnections.add(connection); }

    void setCalculation(double weightedSum, double activation) {
        this.weightedSum = weightedSum;
        this.activation = activation;
    }

    void clearCalculation() {
        weightedSum = 0;
        activation = 0;
    }
}