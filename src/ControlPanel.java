import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.GridLayout;
import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.JTextArea;

public final class ControlPanel extends JPanel {
    private final JComboBox<Integer> inputOne = new JComboBox<>(new Integer[]{0, 1});
    private final JComboBox<Integer> inputTwo = new JComboBox<>(new Integer[]{0, 1});
    private final JComboBox<Integer> expected = new JComboBox<>(new Integer[]{0, 1});
    private final JSlider speed = new JSlider(1, 20, 5);
    private final JTextArea information = new JTextArea();
    private final JButton runForward = new JButton("Run Forward Pass");
    private final JButton trainStep = new JButton("Train One Step");
    private final JButton trainAutomatically = new JButton("Train Automatically");
    private final JButton stopTraining = new JButton("Stop Training");
    private final JButton reset = new JButton("Reset Network");

    public ControlPanel() {
        setPreferredSize(new Dimension(300, 750));
        setBorder(BorderFactory.createEmptyBorder(18, 18, 18, 18));
        setLayout(new BorderLayout(0, 14));
        add(createControls(), BorderLayout.NORTH);
        information.setEditable(false);
        information.setLineWrap(true);
        information.setWrapStyleWord(true);
        information.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 13));
        information.setBackground(new Color(244, 247, 249));
        information.setBorder(BorderFactory.createEmptyBorder(12, 12, 12, 12));
        information.setText("Run a forward pass to inspect the calculation.");
        add(information, BorderLayout.CENTER);
        stopTraining.setEnabled(false);
    }

    private JPanel createControls() {
        JPanel wrapper = new JPanel(new BorderLayout(0, 12));
        JPanel values = new JPanel(new GridLayout(3, 2, 8, 8));
        values.add(new JLabel("X1")); values.add(inputOne);
        values.add(new JLabel("X2")); values.add(inputTwo);
        values.add(new JLabel("Expected")); values.add(expected);
        wrapper.add(values, BorderLayout.NORTH);

        JPanel buttons = new JPanel(new GridLayout(0, 1, 0, 7));
        buttons.add(runForward); buttons.add(trainStep); buttons.add(trainAutomatically);
        buttons.add(stopTraining); buttons.add(reset);
        wrapper.add(buttons, BorderLayout.CENTER);

        JPanel speedRow = new JPanel(new BorderLayout(8, 0));
        speedRow.add(new JLabel("Animation speed"), BorderLayout.NORTH);
        speedRow.add(speed, BorderLayout.CENTER);
        wrapper.add(speedRow, BorderLayout.SOUTH);
        return wrapper;
    }

    public int getInputOne() { return (Integer) inputOne.getSelectedItem(); }
    public int getInputTwo() { return (Integer) inputTwo.getSelectedItem(); }
    public int getExpected() { return (Integer) expected.getSelectedItem(); }
    public int getSpeed() { return speed.getValue(); }
    public JButton getRunForward() { return runForward; }
    public JButton getTrainStep() { return trainStep; }
    public JButton getTrainAutomatically() { return trainAutomatically; }
    public JButton getStopTraining() { return stopTraining; }
    public JButton getReset() { return reset; }
    public void showInformation(String text) { information.setText(text); }

    public void setExample(int firstInput, int secondInput, int expectedOutput) {
        inputOne.setSelectedItem(firstInput);
        inputTwo.setSelectedItem(secondInput);
        expected.setSelectedItem(expectedOutput);
    }

    public void setAutomaticTraining(boolean running) {
        runForward.setEnabled(!running);
        trainStep.setEnabled(!running);
        trainAutomatically.setEnabled(!running);
        stopTraining.setEnabled(running);
    }
}