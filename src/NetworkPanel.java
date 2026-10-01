import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.GradientPaint;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.geom.Ellipse2D;
import java.util.HashMap;
import java.util.Map;
import java.util.function.IntConsumer;
import javax.swing.JPanel;
import javax.swing.Timer;

public final class NetworkPanel extends JPanel {
    private static final Color INK = new Color(31, 43, 51);
    private static final Color TEAL = new Color(24, 135, 145);
    private static final Color CORAL = new Color(224, 105, 76);
    private static final Map<String, Point> POSITIONS = Map.of(
            "X1", new Point(160, 280), "X2", new Point(160, 480),
            "H1", new Point(500, 180), "H2", new Point(500, 380), "H3", new Point(500, 580),
            "OUT", new Point(850, 380));

    private ForwardPass pass;
    private int animationStage;
    private double animationProgress;
    private Timer animationTimer;
    private Timer weightHighlightTimer;
    private Runnable animationCompletion;
    private IntConsumer animationStageChanged;
    private AnimationMode animationMode = AnimationMode.FORWARD;
    private final Map<String, Double> changedWeights = new HashMap<>();

    public NetworkPanel() {
        setBackground(new Color(249, 250, 247));
        setPreferredSize(new Dimension(900, 750));
    }

    public void animate(ForwardPass pass, int speed) {
        animate(pass, speed, null);
    }

    public void animate(ForwardPass pass, int speed, Runnable onComplete) {
        stopTimer();
        clearWeightHighlight();
        this.pass = pass;
        this.animationMode = AnimationMode.FORWARD;
        animationStage = 1;
        animationProgress = 0;
        animationCompletion = onComplete;
        animationStageChanged = null;
        animationTimer = new Timer(30, event -> {
            animationProgress += 0.018 * speed;
            if (animationProgress >= 1) {
                animationProgress = 0;
                animationStage++;
                if (animationStageChanged != null) animationStageChanged.accept(animationStage);
                if (animationStage > 5) {
                    animationStage = 5;
                    stopTimer();
                    Runnable completion = animationCompletion;
                    animationCompletion = null;
                    if (completion != null) completion.run();
                }
            }
            repaint();
        });
        animationTimer.start();
        repaint();
    }

    public void animateBackpropagation(TrainingStep step, int speed, IntConsumer onStageChanged,
            Runnable onComplete) {
        // These purple pulses represent error gradients moving right to left, not activations.
        stopTimer();
        clearWeightHighlight();
        this.pass = step.beforeTraining();
        this.animationMode = AnimationMode.BACKPROPAGATION;
        this.animationStage = 1;
        this.animationProgress = 0;
        this.animationCompletion = onComplete;
        this.animationStageChanged = onStageChanged;
        if (onStageChanged != null) onStageChanged.accept(animationStage);
        animationTimer = new Timer(30, event -> {
            animationProgress += 0.018 * speed;
            if (animationProgress >= 1) {
                animationProgress = 0;
                animationStage++;
                if (animationStageChanged != null) animationStageChanged.accept(animationStage);
                if (animationStage > 5) {
                    animationStage = 5;
                    stopTimer();
                    Runnable completion = animationCompletion;
                    animationCompletion = null;
                    animationStageChanged = null;
                    if (completion != null) completion.run();
                }
            }
            repaint();
        });
        animationTimer.start();
        repaint();
    }

    public void showPass(ForwardPass pass) {
        stopAnimation();
        this.pass = pass;
        this.animationMode = AnimationMode.FORWARD;
        animationStage = 5;
        repaint();
    }

    public void showTrainingChange(ForwardPass before, ForwardPass after) {
        showPass(after);
        clearWeightHighlight();
        changedWeights.clear();
        for (int index = 0; index < before.connections().size(); index++) {
            ForwardPass.ConnectionState oldState = before.connections().get(index);
            ForwardPass.ConnectionState newState = after.connections().get(index);
            if (Math.abs(oldState.weight() - newState.weight()) > 1e-12) {
                changedWeights.put(newState.sourceName() + newState.destinationName(), oldState.weight());
            }
        }
        weightHighlightTimer = new Timer(2200, event -> {
            changedWeights.clear();
            weightHighlightTimer = null;
            repaint();
        });
        weightHighlightTimer.setRepeats(false);
        weightHighlightTimer.start();
        repaint();
    }

    public void stopAnimation() {
        stopTimer();
        clearWeightHighlight();
        animationCompletion = null;
        animationStageChanged = null;
        animationMode = AnimationMode.FORWARD;
        animationStage = 0;
        animationProgress = 0;
        repaint();
    }

    private void clearWeightHighlight() {
        if (weightHighlightTimer != null) {
            weightHighlightTimer.stop();
            weightHighlightTimer = null;
        }
        changedWeights.clear();
    }

    private void stopTimer() {
        if (animationTimer != null) animationTimer.stop();
    }

    @Override
    protected void paintComponent(Graphics graphics) {
        super.paintComponent(graphics);
        Graphics2D g = (Graphics2D) graphics.create();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        drawHeader(g);
        drawConnections(g);
        drawNeurons(g);
        g.dispose();
    }

    private void drawHeader(Graphics2D g) {
        g.setColor(INK); g.setFont(new Font("Georgia", Font.BOLD, 28));
        g.drawString("A small network learning XOR", 45, 55);
        g.setFont(new Font("SansSerif", Font.PLAIN, 14)); g.setColor(new Color(85, 101, 108));
        g.drawString("Signals, weighted sums, and sigmoid activations", 48, 80);
        drawLayerLabel(g, "INPUT LAYER", 105, 100);
        drawLayerLabel(g, "HIDDEN LAYER", 435, 100);
        drawLayerLabel(g, "OUTPUT LAYER", 785, 100);
    }

    private void drawLayerLabel(Graphics2D g, String label, int x, int y) {
        g.setColor(new Color(85, 101, 108));
        g.setFont(new Font("SansSerif", Font.BOLD, 12));
        g.drawString(label, x, y);
    }

    private void drawConnections(Graphics2D g) {
        if (pass == null) return;
        for (ForwardPass.ConnectionState state : pass.connections()) {
            Point source = POSITIONS.get(state.sourceName());
            Point destination = POSITIONS.get(state.destinationName());
            String key = state.sourceName() + state.destinationName();
            boolean active = isActiveConnection(state);
            boolean changed = changedWeights.containsKey(key);
            g.setStroke(new BasicStroke((float) (1.5 + Math.min(5, Math.abs(state.weight()) * 2)), BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
            Color activeColor = animationMode == AnimationMode.BACKPROPAGATION
                    ? new Color(132, 93, 209) : new Color(245, 176, 58);
            g.setColor(changed ? new Color(194, 83, 178) : active ? activeColor
                    : state.weight() >= 0 ? new Color(73, 161, 159) : new Color(218, 122, 106));
            g.drawLine(source.x, source.y, destination.x, destination.y);
            drawWeight(g, state, source, destination, active, changed);
            if (active && animationStage != 3 && animationStage != 5) {
                if (animationMode == AnimationMode.BACKPROPAGATION) {
                    drawPulse(g, destination, source, new Color(132, 93, 209));
                } else {
                    drawPulse(g, source, destination, new Color(255, 194, 64));
                }
            }
        }
    }

    private void drawWeight(Graphics2D g, ForwardPass.ConnectionState state, Point source, Point destination,
            boolean active, boolean changed) {
        int x = (source.x + destination.x) / 2;
        int y = (source.y + destination.y) / 2;
        g.setColor(active || changed ? INK : new Color(75, 88, 93));
        g.setFont(new Font("Monospaced", Font.PLAIN, 12));
        g.drawString(String.format("w=%+.2f", state.weight()), x - 25, y - 7);
        if (changed) {
            g.setFont(new Font("Monospaced", Font.BOLD, 11));
            g.drawString(String.format("was %+.2f", changedWeights.get(state.sourceName() + state.destinationName())), x - 25, y + 12);
        } else if (animationMode == AnimationMode.FORWARD && active
            && (animationStage == 2 || animationStage == 4)) {
            g.setFont(new Font("Monospaced", Font.BOLD, 11));
            g.drawString(String.format("%.2f x %+.2f = %+.2f", activationFor(state.sourceName()), state.weight(), state.contribution()), x - 50, y + 12);
        }
    }

    private void drawPulse(Graphics2D g, Point source, Point destination, Color color) {
        int x = (int) (source.x + animationProgress * (destination.x - source.x));
        int y = (int) (source.y + animationProgress * (destination.y - source.y));
        g.setColor(color);
        g.fill(new Ellipse2D.Double(x - 9, y - 9, 18, 18));
    }

    private void drawNeurons(Graphics2D g) {
        if (pass == null) return;
        for (String name : POSITIONS.keySet()) {
            double activation = activationFor(name);
            Point point = POSITIONS.get(name);
            boolean highlighted = isActiveNeuron(name);
            int radius = highlighted ? 55 : 48;
            Color base = name.startsWith("H") ? TEAL : CORAL;
            g.setPaint(new GradientPaint(point.x - radius, point.y - radius, blend(base, activation),
                    point.x + radius, point.y + radius, Color.WHITE));
            g.fillOval(point.x - radius, point.y - radius, radius * 2, radius * 2);
            g.setColor(highlighted ? new Color(245, 176, 58) : INK);
            g.setStroke(new BasicStroke(highlighted ? 5 : 2));
            g.drawOval(point.x - radius, point.y - radius, radius * 2, radius * 2);
            g.setColor(INK); g.setFont(new Font("SansSerif", Font.BOLD, 16));
            g.drawString(name, point.x - 14, point.y - radius - 12);
            g.setFont(new Font("Monospaced", Font.BOLD, 16));
            g.drawString(String.format("%.3f", activation), point.x - 25, point.y + 6);
        }
    }

    private boolean isActiveConnection(ForwardPass.ConnectionState state) {
        if (animationMode == AnimationMode.BACKPROPAGATION) {
            return animationStage == 2 && state.sourceName().startsWith("H")
                || animationStage == 4 && state.sourceName().startsWith("X");
        }
        return animationStage == 2 && state.sourceName().startsWith("X")
            || animationStage == 4 && state.sourceName().startsWith("H");
    }

    private boolean isActiveNeuron(String name) {
        if (animationMode == AnimationMode.BACKPROPAGATION) {
            return animationStage == 1 && name.equals("OUT")
                || animationStage == 3 && name.startsWith("H");
        }
        return animationStage == 1 && name.startsWith("X")
                || animationStage == 3 && name.startsWith("H")
                || animationStage == 5 && name.equals("OUT");
    }

    private double activationFor(String name) {
        if (name.equals("X1")) return pass.inputs()[0];
        if (name.equals("X2")) return pass.inputs()[1];
        if (name.startsWith("H")) return pass.hiddenActivations()[Integer.parseInt(name.substring(1)) - 1];
        return pass.outputActivation();
    }

    private static Color blend(Color color, double amount) {
        int alpha = (int) (Math.max(0.15, Math.min(1, amount)) * 210);
        return new Color(color.getRed(), color.getGreen(), color.getBlue(), alpha);
    }

    private enum AnimationMode {
        FORWARD,
        BACKPROPAGATION
    }

    private record Point(int x, int y) { }
}