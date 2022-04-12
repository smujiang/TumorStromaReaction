import qupath.lib.objects.PathObject
import static qupath.lib.gui.scripting.QPEx.*
import qupath.lib.roi.RectangleROI

def pathObjects = getDetectionObjects()

def patch_size = 200
// load x, y, probability from txt file

def filePath = "H:\\Jun_anonymized_dir\\TIL\\example\\xxx_HE_lym.txt"

def cnt = 0
def hierarchy = getCurrentHierarchy()
new File(filePath).eachLine() { line ->
    def ele = line.trim().split(" ")
    double prob = Double.parseDouble(ele[2])

    if(prob >= 0.5) {
        def x_str = ele[0]
        def y_str = ele[1]
        def cx = x_str.toInteger()
        def cy = y_str.toInteger()
        def roi = new RectangleROI(cx, cy, patch_size, patch_size)
        // Add tile detection objects to the image
        def tile_obj = PathObjects.createTileObject(roi)
        // Add the measurements
        tile_obj.getMeasurementList().putMeasurement("TIL probability", prob)
        tile_obj.getMeasurementList().putMeasurement("NEC probability", prob)
        hierarchy.addPathObject(tile_obj)
    }
//     cnt += 1
//     if(cnt > 1000){
//         break
//     }
}
fireHierarchyUpdate()
print("Done!")



// show measurements map, select the















