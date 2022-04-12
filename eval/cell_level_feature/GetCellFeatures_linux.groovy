/**
 * * Create a region annotation by enlarging the original area a little bit to fit the patch extraction size
 *
 * @author Jun Jiang  Jiang.Jun@mayo.edu
 * tested on QuPath 0.2.3
 */
//https://forum.image.sc/t/creating-project-from-command-line/45608/2

import java.awt.image.BufferedImage
import groovy.io.FileType
import qupath.lib.images.servers.ImageServerProvider
import qupath.lib.gui.commands.ProjectCommands
import qupath.lib.roi.RectangleROI
import qupath.lib.objects.PathAnnotationObject
import java.io.BufferedReader
// import qupath.ext.stardist.StarDist2D
// import qupath.ext.tensorflow

// Paths
// def proj_dir = "\\\\anonymized_dir\\QuPathProj"
// def wsi_dir = "\\\\anonymized_dir\\POLE_WSI\\req30990"

// def proj_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/QuPathProj"
// def wsi_dir = "/Jun_anonymized_dir/OvaryCancer/WSIs"


// def proj_dir = "H:\\Jun_anonymized_dir\\OvaryCancer\\ImageData\\QuPathProj"
// def wsi_dir = "H:\\Jun_anonymized_dir\\OvaryCancer\\ImageData\\WSIs"

def proj_dir = "\\\\anonymized_dir\\QuPathProj"
def wsi_dir = "\\\\anonymized_dir\\WSIs\\AI_analysis_on_stromal_reactions"
def wsi_thumbnail_dir = "\\\\anonymized_dir\\thumbnails"

def testing_case_output_dir = "\\\\anonymized_dir\\result_analysis\\cell_rois"
// def pathModel = "/Jun_anonymized_dir/model/StarDist/he_heavy_augment.pb"
def output_dir = "\\\\anonymized_dir\\result_analysis\\roi_cell_features"

System.setProperty("java.awt.headless", "true")  // important if in linux

def wsi_list = []
def dir = new File(wsi_thumbnail_dir)
dir.eachFileRecurse (FileType.DIRECTORIES) {
    wsi_list << it
}

////////////////////////////////////////////
// Process ROIs
// enlarge the detected area a little bit to fit the patch extraction size
////////////////////////////////////////////
def patch_size = [512, 512]
def adjustROIforProperExtraction(List rect, List patch_size, List img_size){
    w_m = rect[2] % patch_size[0]
    h_m = rect[3] % patch_size[1]
    def new_w = 0
    def new_h = 0
    for (i in 0..patch_size[0]){
        new_w = rect[2]+i
        if ((new_w) % patch_size[0] == 0){
            break
        }
    }
    for (i in 0..patch_size[1]){
        new_h = rect[3]+i
        if ((new_h) % patch_size[1] == 0){
            break
        }
    }
    return [rect[0], rect[1], new_w, new_h]
}

//
// def pathModel = '/Jun_anonymized_dir/model/StarDist/he_heavy_augment.pb'
// def pathModel = '/Jun_anonymized_dir/model/StarDist'

// def stardist = StarDist2D.builder(pathModel)
//       .threshold(0.5)              // Prediction threshold
//       .normalizePercentiles(1, 99) // Percentile normalization
//       .pixelSize(0.25)              // Resolution for detection
//       .cellExpansion(5.0)
//       .cellConstrainScale(1.5)
//       .measureShape()
//       .measureIntensity()
//       .build()

// def dnn = tensorflow.TensorFlowTools.createDnnModel(pathModel)
//
// def stardist = StarDist2D.builder(dnn)
//       .threshold(0.5)              // Prediction threshold
//       .normalizePercentiles(1, 99) // Percentile normalization
//       .pixelSize(0.25)              // Resolution for detection
//       .cellExpansion(5.0)
//       .cellConstrainScale(1.5)
//       .measureShape()
//       .measureIntensity()
//       .build()



println(sprintf('%d WSIs in %s', wsi_list.size, wsi_dir))
def group_per_imgs = 30
def wsi_groups = (0..wsi_list.size).step(group_per_imgs)
if (!wsi_groups.contains(wsi_list.size)){
    wsi_groups << wsi_list.size
}
def cnt = 0
for (g_ids in wsi_groups){
    // Create project
//     def wsi_par_folder = new File(wsi_dir).getName()
    def wsi_par_folder = "TSR"
    def projectName = wsi_par_folder + "_" + cnt.toString()  + ".qpproj"
    def sub_dir = wsi_par_folder + "_" + cnt.toString()
    File directory = new File(proj_dir + File.separator + sub_dir + File.separator + projectName)
    def project = Projects.createProject(directory , BufferedImage.class)
//     print(wsi_groups)
    if (cnt+1 >= wsi_groups.size){
        break
    }
    def grp = (g_ids..(wsi_groups[cnt+1]-1))
//     print(grp)
    println("*******************************************")
    for (g in grp){
        //setImageType('BRIGHTFIELD_H_E')
        def imagePath = wsi_dir + File.separator + wsi_list[g].getName() + ".svs"
        def fn = new File(imagePath)
        def uri = fn.toURI()
        // get image server
        def server = new qupath.lib.images.servers.openslide.OpenslideServerBuilder().buildServer(uri)
        def builder = server.getBuilder()  // Get serverBuilder
        if (builder == null) {
            print "Image not supported"
            return
        }

        // Add the image as entry to the project
        print "Adding: " + imagePath
        entry = project.addImage(builder)
        // Set a particular image type
        def imageData = entry.readImageData()
        def hierarchy = imageData.getHierarchy()
        imageData.setImageType(ImageData.ImageType.BRIGHTFIELD_H_E)
        entry.saveImageData(imageData)

        // Write a thumbnail if we can
        var img = ProjectCommands.getThumbnailRGB(imageData.getServer());
        entry.setThumbnail(img)

        // Add ROI to the image
        p_sz_h = server.getPixelCalibration().pixelHeightMicrons
        p_sz_w = server.getPixelCalibration().pixelWidthMicrons
        def case_id = server.getMetadata().getName().take(server.getMetadata().getName().lastIndexOf('.'))
        def csv_fn = testing_case_output_dir + File.separator + case_id + "_roi_box.csv"
        println(csv_fn +"\n")
        try {
            def annotations = []

            def csvReader = new BufferedReader(new FileReader(csv_fn));
            def row = csvReader.readLine()
            // Loop through all the rows of the CSV file.
            def pathClass = "ROI"
            while ((row = csvReader.readLine()) != null) {
                print(row + "\n")
                def rowContent = row.split(",")

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

                annotations.add(annotation)
            }
            println(annotations)
            hierarchy.addPathObjects(annotations) //TODO: can't add objects?
            fireHierarchyUpdate()

            // Add an entry name (the filename)
            entry.setImageName(wsi_list[g].getName())
            // Changes should now be reflected in the project directory
            project.syncChanges()

        } catch (Exception e) {
            print('Unable to parse evaluation results from ' + csv_fn + ': ' + e.getLocalizedMessage())
        }

        // detect cells and get cell features
        def Anno_list = []
        def annotations = hierarchy.getFlattenedObjectList(null).findAll {it.isAnnotation()}
        print(sprintf("Annotation counts: %d \n", annotations.size))
        int idx = 0
        for (anno in annotations){
            if(anno.getPathClass().toString().equals("ROI")){
                Anno_list << anno
                // remove the old annotation
        //            imageData.getHierarchy().removeObjects(anno, false)
                annotations.getAt(idx).setLocked(false)
                imageData.getHierarchy().removeObject(annotations.getAt(idx), false)
            }
            idx += 1
        }

        if (server.getPixelCalibration().hasPixelSizeMicrons()){
        //    if (server.hasPixelSizeMicrons()){
            for (anno in Anno_list){
                // TODO: optimize the size of ROI
                int cx = (int)anno.getROI().x
                int cy = (int)anno.getROI().y
                int w = (int)(anno.getROI().x2 - cx)
                int h = (int)(anno.getROI().y2 - cy)
                def xy = [cx, cy, w, h]

                def WSI_Width = server.width
                def WSI_Height = server.height
                def rect = adjustROIforProperExtraction(xy, patch_size, [WSI_Width, WSI_Height])
                print(sprintf("\t Original size:[x=%d, y=%d, w=%d, h=%d]", (int)xy[0], (int)xy[1], (int)xy[2], (int)xy[3]))
                print(sprintf("\t Optimized size:[x=%d, y=%d, w=%d, h=%d]",(int)rect[0], (int)rect[1], (int)rect[2], (int)rect[3]))
                def roi = new RectangleROI(cx, cy, rect[2], rect[3])

                print(sprintf("\t Can get %d patches from this ROI", (int)(rect[2]/patch_size[0]+ rect[3]/patch_size[1])))
                // Create & new annotation & add it to the object hierarchy
                def annotation = new PathAnnotationObject(roi, PathClassFactory.getPathClass("ROI"))
        //        annotation.setLocked(true) //Lock this annotation
                imageData.getHierarchy().addPathObject(annotation, false)
                fireHierarchyUpdate()
            }
        }
        else{
            return
        }
        println("Detect cells!\n")
        selectAnnotations()
        runPlugin('qupath.imagej.detect.cells.WatershedCellDetection',
                '{"requestedPixelSizeMicrons": 0.25, "backgroundRadiusMicrons": 8, "medianRadiusMicrons": 0,  "sigmaMicrons": 1.5,  "minAreaMicrons": 10.0,  "maxAreaMicrons": 400.0,  "threshold": 0.1,  "watershedPostProcess": true,  "cellExpansionMicrons": 1.0,  "includeNuclei": true,  "smoothBoundaries": true,  "makeMeasurements": true}');

        def det = hierarchy.getFlattenedObjectList(null).findAll {it.isDetection()}
        println(det)

        fireHierarchyUpdate()
//         print("Process ROIs Done!")
//         annotations = hierarchy.getFlattenedObjectList(null).findAll {it.isAnnotation()}
// //         def detections = hierarchy.getFlattenedObjectList(null).findAll {it.isDetection()}
//         println(detections)
// //         if (!detections) {  // have no cell detections
// //         hierarchy.getSelectionModel().clearSelection()
//         println(annotations)
//         for (anno in annotations){
//
//
//             hierarchy.getSelectionModel().setSelectedObject(anno)
// // //             selectAnnotations()
// //             println("\n \n detect cells \n \n ")
//             runPlugin('qupath.imagej.detect.cells.WatershedCellDetection',
//                     '{"requestedPixelSizeMicrons": 0.25, "backgroundRadiusMicrons": 8, "medianRadiusMicrons": 0,  "sigmaMicrons": 1.5,  "minAreaMicrons": 10.0,  "maxAreaMicrons": 400.0,  "threshold": 0.1,  "watershedPostProcess": true,  "cellExpansionMicrons": 1.0,  "includeNuclei": true,  "smoothBoundaries": true,  "makeMeasurements": true}');
//             def det = hierarchy.getFlattenedObjectList(null).findAll {it.isDetection()}
//             println(det)
//
// //             stardist.detectObjects(imageData, anno)
//
//
//             println(det)
//
//
//
//
//             // TODO: why there is no cell detected
//
//
//
//
//
//
//
//
//
//
//
//
//             resetSelection()
//             fireHierarchyUpdate()
//          }
//         }
//         def detections = hierarchy.getFlattenedObjectList(null).findAll {it.isDetection()}
//         println(detections)


        // output_dir = "H:\\Jun_anonymized_dir\\OvaryCancer\\StromaReaction\\pipeline\\result_analysis\\roi_cell_features"
//         output_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/result_analysis/roi_cell_features"

        pathOutput = output_dir + File.separator
        File newDir = new File(pathOutput)
        if (!newDir.exists()) {
            newDir.mkdirs()
        }
        save_fn = pathOutput + server.getMetadata().getName().take(server.getMetadata().getName().lastIndexOf('.')) + "_cell_features.txt"
        println(save_fn)
        saveDetectionMeasurements(save_fn)

        //
    }
    cnt += 1
//     break
}

print("Done")


























