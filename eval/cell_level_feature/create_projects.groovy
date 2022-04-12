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

// Paths
def proj_dir = "\\\\anonymized_dir\\QuPathProj"
def wsi_dir = "\\\\anonymized_dir\\WSIs\\AI_analysis_on_stromal_reactions"
// def wsi_dir = "\\\\anonymized_dir\\POLE_WSI\\req30990"

// def proj_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/QuPathProj"
// def wsi_dir = "/Jun_anonymized_dir/OvaryCancer/WSIs"

// def proj_dir = "H:\\Jun_anonymized_dir\\OvaryCancer\\ImageData\\QuPathProj"
// def wsi_dir = "H:\\Jun_anonymized_dir\\OvaryCancer\\ImageData\\WSIs"
System.setProperty("java.awt.headless", "true")  // important if in linux

def wsi_list = []
def dir = new File(wsi_dir)
dir.eachFileRecurse (FileType.FILES) {
    if (it.name.endsWith('.svs')){
        wsi_list << it
    }
}

println(sprintf('%d WSIs in %s', wsi_list.size, wsi_dir))
def group_per_imgs = 300
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
    print(wsi_groups)
    if (cnt+1 >= wsi_groups.size){
        break
    }
    def grp = (g_ids..(wsi_groups[cnt+1]-1))
    print(grp)
    for (g in grp){
        //setImageType('BRIGHTFIELD_H_E')
        def imagePath = wsi_dir + File.separator + wsi_list[g].getName()
        def fn = new File(imagePath)
        def uri = fn.toURI()
        // get image server
        def server = new qupath.lib.images.servers.openslide.OpenslideServerBuilder().buildServer(uri)
        def builder = server.getBuilder()  // Get serverBuilder
        // Make sure we don't have null
        if (builder == null) {
            print "Image not supported"
            return
        }

        // Add the image as entry to the project
        print "Adding: " + imagePath
        entry = project.addImage(builder)
        // Set a particular image type
        def imageData = entry.readImageData()
        imageData.setImageType(ImageData.ImageType.BRIGHTFIELD_H_E)
        entry.saveImageData(imageData)

        // Write a thumbnail if we can
        var img = ProjectCommands.getThumbnailRGB(imageData.getServer());
        entry.setThumbnail(img)

        // Add an entry name (the filename)
        entry.setImageName(wsi_list[g].getName())
        // Changes should now be reflected in the project directory
        project.syncChanges()

    }




    cnt += 1
//     break
}

print("Done")







