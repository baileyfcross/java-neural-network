# Java Neural Network Visualization

A small educational Java 25 program that shows a neural network learning XOR. The network and visualization are implemented from scratch with standard Java, Swing, Java2D, and `javax.swing.Timer`. There is no Maven, Gradle, or machine-learning framework.

## What It Demonstrates

The network has a `2 -> 3 -> 1` architecture:

- `X1`, `X2`: input neurons
- `H1`, `H2`, `H3`: hidden neurons
- `OUT`: output neuron

It learns XOR:

| X1 | X2 | Expected |
|---:|---:|---------:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Each neuron calculates a weighted sum, adds a bias, and applies:

`sigmoid(x) = 1 / (1 + e^(-x))`

The `NeuralNetwork` class contains the forward propagation, error calculation, backpropagation, and gradient-descent weight changes. A `Connection` is a Java object for one line in the drawing. Its contribution is `source activation * weight`.

## Run in Visual Studio Code

Install a Java 25 JDK and open this folder in VS Code. The workspace file `.vscode/settings.json` selects the installed Eclipse Adoptium JDK 25 for the Java language server, Run/Debug, and new integrated terminals. Open a new terminal after opening the project so its `java` and `javac` commands use that JDK. Run `src/Main.java` using the Java extension's Run button.

The same program can be compiled directly from a terminal:

```text
mkdir out
javac -d out src\*.java
java -cp out Main
```

On macOS or Linux, replace the backslash in `src\*.java` with `/`.

## Project Structure

```text
src/
	Main.java
	NeuralNetwork.java
	Neuron.java
	Connection.java
	ForwardPass.java
	NeuralNetworkFrame.java
	NetworkPanel.java
	ControlPanel.java
test/
	NeuralNetworkTest.java
```

`ForwardPass` is a completed, immutable snapshot. The UI animates that snapshot instead of calculating inside `paintComponent`. This keeps the relationship clear: neurons are circles, connections are lines, weights are labels, and activations/contributions are the values students see.

## Optional JUnit Tests

The application does not need JUnit. To run the optional tests, place a JUnit Platform Console Standalone JAR in `lib/`, then compile the source and tests:

```text
mkdir out
javac -cp "lib\junit-platform-console-standalone.jar" -d out src\*.java test\*.java
java -jar lib\junit-platform-console-standalone.jar --class-path out --scan-class-path
```

The tests cover sigmoid calculations, deterministic initialization, forward propagation, changing weights, decreasing XOR error, and final XOR classification.

## Classroom Demonstration

1. Start the program.
2. Reset the neural network.
3. Select `X1 = 1` and `X2 = 0`.
4. Click **Run Forward Pass**.
5. Watch signals move from the input layer.
6. Observe weighted calculations entering the hidden layer.
7. Watch the hidden neurons activate.
8. Watch their values travel to the output neuron.
9. Observe the initial prediction.
10. Click **Train One Step** or **Train Automatically**.
11. Watch the error decrease.
12. Run the same input again.
13. Compare the new output.
14. Show students which connection weights changed.
