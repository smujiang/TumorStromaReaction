import qupath.lib.objects.PathObject
import static qupath.lib.gui.scripting.QPEx.*
import qupath.lib.roi.RectangleROI

def pathObjects = getDetectionObjects()

def patch_size = 200
// load x, y, probabilities from txt file. In each line,
// x, y: location where image patch was extracted
// probabilities: current have two values, denote the probability of this patch classified to TIL(lymphocyte) and necrosis.
def filePath = "H:\\Jun_anonymized_dir\\TIL\\example\\_HE_lym_nec.txt"
def cnt = 0
def hierarchy = getCurrentHierarchy()
def tile_obj_list = []
new File(filePath).eachLine() { line ->
    def ele = line.trim().split(",")
    double prob1 = Double.parseDouble(ele[2])
    double prob2 = Double.parseDouble(ele[3])
    if(prob1 >= 0.1 || prob2 >= 0.1) {
        def x_str = ele[0]
        def y_str = ele[1]
        def cx = x_str.toInteger()
        def cy = y_str.toInteger()
        def roi = new RectangleROI(cx, cy, patch_size, patch_size)
        // Add tile detection objects to the image
        def tile_obj = PathObjects.createTileObject(roi)
        // Add the measurements
        tile_obj.getMeasurementList().putMeasurement("TIL probability", prob1)
        tile_obj.getMeasurementList().putMeasurement("NEC probability", prob2)
        tile_obj_list.add(tile_obj)
    }
}
hierarchy.addPathObjects(tile_obj_list)
fireHierarchyUpdate()
print("Done!")

// show measurements map, select the















