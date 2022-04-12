System.setProperty("java.awt.headless", "true")  // important if in linux


import qupath.lib.geom.Point2
// import qupath.lib.gui.scripting.QPEx
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.objects.PathDetectionObject
import qupath.lib.roi.PolygonROI
import qupath.lib.objects.classes.PathClassFactory
import qupath.lib.roi.RectangleROI
import qupath.lib.scripting.QP

// import static com.xlson.groovycsv.CsvParser.parseCsv
// import javax.imageio.ImageIO
import java.io.BufferedReader;
import java.io.FileReader;



// def testing_case_output_dir = "H:\\Jun_anonymized_dir\\OvaryCancer\\StromaReaction\\pipeline\\result_analysis\\cell_rois"
def testing_case_output_dir = "\\\\anonymized_dir\\result_analysis\\cell_rois"
print(testing_case_output_dir)
def test_dir = new File(testing_case_output_dir)

def hierarchy = getCurrentHierarchy()
def server = getCurrentImageData().getServer()
// def server = new qupath.lib.images.servers.openslide.OpenslideServerBuilder().buildServer(uri)

p_sz_h = server.getPixelCalibration().pixelHeightMicrons
p_sz_w = server.getPixelCalibration().pixelWidthMicrons
def case_id = server.getMetadata().getName().take(server.getMetadata().getName().lastIndexOf('.'))

def csv_fn = testing_case_output_dir + File.separator + case_id + "_roi_box.csv"
print(csv_fn +"\n")
try {
    def annotations = []
    // parse txt file created by python (locations and scores)
//     def fh = new File(csv_fn)
//     def csv_content = fh.getText('utf-8')
//     def data_iterator = parseCsv(csv_content, separator: ',', readFirstLine: false)

    def csvReader = new BufferedReader(new FileReader(csv_fn));
    def row = csvReader.readLine()
    // Loop through all the rows of the CSV file.
    def pathClass = "ROI"
    while ((row = csvReader.readLine()) != null) {
        print(row + "\n")
        def rowContent = row.split(",")
//         double cx = rowContent[1] as double;
//         double cy = rowContent[2] as double;

        def tl_x = rowContent[0] as int
        def tl_y = rowContent[1] as int
        def br_x = rowContent[2] as int
        def br_y = rowContent[3] as int

        def w = br_x - tl_x
        def h = br_y - tl_y
        print(w.toString() + "--" + h.toString() +"\n")

        def roi = new RectangleROI(tl_x, tl_y, w, h)
        def annotation = new PathAnnotationObject(roi)
        annotation.setPathClass(QPEx.getPathClass(pathClass))

//         hierarchy.addPathObject(annotation, false)
        annotations.add(annotation)
        print(annotation)

    }
//     def pathClass = "ROI"
//     for (line in data_iterator) {
//         def tl_x = line[0] as int
//         def tl_y = line[1] as int
//         def br_x = line[2] as int
//         def br_y = line[3] as int
//
//         def w = br_x - tl_x
//         def h = br_y - tl_y
//         print(w)
//         print(h)
//         def roi = new RectangleROI(tl_x, tl_y, w, h)
//         def annotation = new PathAnnotationObject(roi)
//         annotation.setPathClass(QPEx.getPathClass(pathClass))
//         // add scores to those annotations
//         annotations.add(annotation)
//     }
    //TODO: can't add annotations ???????????????????????
//     hierarchy.addPathObjects(annotations) //TODO: can't add objects?
//     QPEx.fireHierarchyUpdate()
} catch (Exception e) {
    print('Unable to parse evaluation results from ' + csv_fn + ': ' + e.getLocalizedMessage())
}

print("\n")
print("\n")
