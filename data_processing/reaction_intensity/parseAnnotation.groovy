/*
 * Author: Jun Jiang (Jiang.Jun@mayo.edu), tested on QuPath 0.2.0-m8
 * This is the script for parsing polygon annotations.
 * If any package is missing, it will throw errors. You need to drag the missing jar onto the QuPath window, and you just need to do this only once. This step copies the jar into QuPath's jar directory
 *  --------------------------------------------------
 * 1). get the annotations before any processing, in case annotations were messed up.
 *      a.

 * */


import qupath.lib.images.servers.ImageServer
import qupath.lib.objects.PathObject
import qupath.lib.regions.RegionRequest
import qupath.lib.roi.PointsROI
import qupath.lib.roi.PolygonROI
import qupath.lib.roi.GeometryROI
import qupath.lib.gui.scripting.QPEx
import qupath.lib.roi.RoiTools
import qupath.lib.scripting.QP
import qupath.lib.objects.classes.PathClassFactory

import javax.imageio.ImageIO
import java.awt.Color
import java.awt.image.BufferedImage

// Get the main QuPath data structures
def imageData = QP.getCurrentImageData()
def hierarchy = imageData.getHierarchy()
def server = imageData.getServer()

// Request all objects from the hierarchy & filter only the annotations
def annotations = hierarchy.getFlattenedObjectList(null).findAll {it.isAnnotation()}

// Define downsample value for export resolution & output directory, creating directory if necessary
def downsample = 1.0
// modify this root_dir to specify where you would like to save your export
// def pathOutput = "/anonymized_dir/Dataset/Annotations/QP1.2_annotation/output"
def pathOutput = "H:\\Jun_anonymized_dir\\OvaryCancer\\StromaReaction\\Annotation"

// Define image export type; valid values are JPG, PNG or null (if no image region should be exported with the mask)
// Note: this just define the image extension, not mask extension. masks will always be exported as PNG
def imageExportType = 'JPG'

//def detections = hierarchy.getFlattenedObjectList(null).findAll {it.isDetection()}
def cell_detections = QP.getCellObjects()
// Get all ROI annotations
def ROI_list = []
def Polygon_annotation_list = []
for (anno in annotations){
    if(anno.getPathClass().toString().equals("ROI")){
        ROI_list << anno
    }
    else if(anno.getROI() instanceof PolygonROI){ //TODO: important!!! could also be geometry
        //https://gist.github.com/Svidro/5829ba53f927e79bb6e370a6a6747cfd
        Polygon_annotation_list << anno
    }
    else if(anno.getROI() instanceof GeometryROI){ //TODO: important!!! could also be geometry
        //https://gist.github.com/Svidro/5829ba53f927e79bb6e370a6a6747cfd
        // seems no need to fill holes
        Polygon_annotation_list << anno
    }
}

// "Fibrosis - 1", "Fibrosis - 2"
// List<String> class_txt_list = ["Fibrosis", "Cellularity", "Orientation / parallelity"]
List<String> class_txt_list = ["Fibrosis", "Cellularity", "Orientation / parallelity", "Tumor"]
List<Integer> class_scores = [1, 2, 3]
List<List<Integer>> class_color_list1 = [[100, 0, 0], [180, 0, 0], [255, 0, 0]]
List<List<Integer>> class_color_list2 = [[0, 100, 0], [0, 180, 0], [0, 255, 0]]
List<List<Integer>> class_color_list3 = [[0, 0, 100], [0, 0, 180], [0, 0, 255]]
// List<List<Integer>> class_color_list4 = [[0, 255, 255], [0, 255, 255], [0, 255, 255]]
def cmap = new ArrayList([class_color_list1, class_color_list2, class_color_list3])
// def cmap = new ArrayList([class_color_list1, class_color_list2, class_color_list3, class_color_list4])
def get_colors_by_label(class_label, class_label_list, color_mp){
    if (class_label.equals("Tumor")){ // Tumor region
        return [0, 255, 255]
    }
    else{  // Stroma region
        String[] str = class_label.split(" - ")
        def selected = [0, 0, 0]
        class_label_list.eachWithIndex { item, index ->
            if (item.equals(str[0])){
                selected = color_mp[index][str[1].toInteger()-1]
            }
        }
        return selected
    }
}

//lm1 = get_colors_by_label("Fibrosis - 1", class_txt_list, cmap)
//print(lm1)
//lm2 = get_colors_by_label("Cellularity - 2", class_txt_list, cmap)
//print(lm2)
//lm3 = get_colors_by_label("Tumor", class_txt_list, cmap)
//print(lm3)

String case_id = server.getMetadata().getName().take(server.getMetadata().getName().lastIndexOf('.'))
roi_cnt = 0
for (roi in ROI_list){
    def patch_roi = roi.getROI()
    String name = String.format('%s_%d_%d_%d_%d',
            case_id,
            patch_roi.getBoundsX().toInteger(),
            patch_roi.getBoundsY().toInteger(),
            patch_roi.getBoundsWidth().toInteger(),
            patch_roi.getBoundsHeight().toInteger()
    )

    def patch_region = RegionRequest.createInstance(server.getPath(), downsample, patch_roi)
    // Request the BufferedImage
    def org_img = server.readBufferedImage(patch_region)
    def fileImage = new File(pathOutput + File.separator + case_id + File.separator + name + '.' + imageExportType.toLowerCase())
    print(pathOutput + File.separator + case_id + File.separator + name + '.' + imageExportType.toLowerCase())
    def directory = new File(pathOutput + File.separator + case_id)
    if (!directory.exists()){
        directory.mkdir()
    }
    ImageIO.write(org_img, imageExportType, fileImage)



    print(roi)
    roi_cnt += 1
    polygon_cnt = 0
    roi_x = roi.getROI().centroidX
    roi_y = roi.getROI().centroidY
    print(sprintf("\tROI location: %f %f", roi_x, roi_y))
    for (c in class_txt_list) {
        // TODO: create a empty mask with the same size of ROI
        def mask_img = new BufferedImage(org_img.getWidth(), org_img.getHeight(), BufferedImage.TYPE_INT_RGB)
        def g2d = mask_img.createGraphics()
        g2d.setColor(Color.WHITE)
        g2d.fillRect(0, 0, mask_img.width, mask_img.height) //create white background
        g2d.scale(1.0/downsample, 1.0/downsample)
        g2d.translate(-patch_region.getX(), -patch_region.getY())  //should not be in for loop, otherwise the drawing will be incomplete

        for (p in Polygon_annotation_list){
            String class_label = p.getPathClass().toString()
            if (class_label.contains(c)){
                polygon_ROI = p.getROI()
                px = polygon_ROI.centroidX
                py = polygon_ROI.centroidY
                if (roi.getROI().contains(px, py)){
                    def shape = RoiTools.getShape(polygon_ROI)
                    brush_color =  get_colors_by_label(class_label, class_txt_list, cmap)
                    print(class_label)
                    print(brush_color)
                    g2d.setColor(new Color(brush_color[0], brush_color[1], brush_color[2]))
                    g2d.fill(shape)
                }
            }
        }
        g2d.dispose()
        String ct = c.split("/")[0]
        def fileMask = new File(pathOutput + File.separator + case_id + File.separator + name + "_" + ct + '-mask.png')
        ImageIO.write(mask_img, 'PNG', fileMask)
    }
}



print "finished"










