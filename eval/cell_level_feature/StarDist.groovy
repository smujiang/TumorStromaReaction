import qupath.tensorflow.stardist.StarDist2D

// Specify the model directory (you will need to change this!)
def pathModel = 'H:\\Jun_anonymized_dir\\OvaryCancer\\CellDetection\\he_heavy_augment'

def stardist = StarDist2D.builder(pathModel)
      .threshold(0.5)              // Prediction threshold
      .normalizePercentiles(1, 99) // Percentile normalization
      .pixelSize(0.25)              // Resolution for detection
      .cellExpansion(5.0)
      .cellConstrainScale(1.5)
      .measureShape()
      .measureIntensity()
      .build()

// Run detection for the selected objects
def imageData = getCurrentImageData()
def pathObjects = getSelectedObjects()
if (pathObjects.isEmpty()) {
    Dialogs.showErrorMessage("StarDist", "Please select a parent object!")
    return
}
stardist.detectObjects(imageData, pathObjects)
println 'Done!'