Remove-Item build -Recurse -Force -ErrorAction SilentlyContinue

New-Item -ItemType Directory -Path build\classes
New-Item -ItemType Directory -Path build\package

javac -d build\classes src\*.java

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

jar --create `
    --file build\package\NeuralNetworkVisualizer.jar `
    --main-class Main `
    -C build\classes .

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

jpackage `
    --type app-image `
    --name "Neural Network Visualizer" `
    --input build\package `
    --main-jar NeuralNetworkVisualizer.jar `
    --main-class Main `
    --dest build\distribution