rm(list = ls())

options(stringsAsFactors = FALSE)

setwd("H:\\Jun_anonymized_dir\\OvaryCancer\\StromaReaction\\pipeline\\result_analysis\\")


r1 <- read.csv("all_case_metadata.csv")

f.mtx0 <- as.matrix(r1[,15:193])
f.mtx <- f.mtx0
for(k in 1:ncol(f.mtx0)){
  f.mtx[,k] <- (f.mtx0[,k] - mean(f.mtx0[,k]))/sd(f.mtx0[,k])
}

require(gplots)

row_color <- as.vector(rep("black", ncol(f.mtx)))
row_color[0:170]<- "grey"

png("feature_clust2_new.png", width = 2e3, height = 3e3, res = 250)
heatmap.2(t(f.mtx), trace = "none", col = bluered(25),
          symbreaks=TRUE,breaks=seq(-3,3,length.out=26),
          margins = c(10, 12),
          ColSideColors = ifelse(r1$histology_type=="HGSOC",yes = "black", no = "grey"),
          RowSideColors = row_color)
dev.off()


png("feature_clust2.png", width = 2e3, height = 3e3, res = 250)
heatmap.2(t(f.mtx), trace = "none", col = bluered(25),
          symbreaks=TRUE,breaks=seq(-3,3,length.out=26),
          margins = c(10, 12),
          ColSideColors = ifelse(r1$histology_type=="HGSOC",yes = "black", no = "grey"))
dev.off()

png("feature_clust2_TSR.png", width = 2e3, height = 2e3, res = 350)
heatmap.2(t(f.mtx[,171:179]), trace = "none", col = bluered(25),
          symbreaks=TRUE,breaks=seq(-3,3,length.out=26),
          margins = c(10, 12),
          ColSideColors = ifelse(r1$histology_type=="HGSOC",yes = "black", no = "grey"))
dev.off()


png("feature_clust3_HGSC_only_TSR.png", width = 2e3, height = 2e3, res = 350)
heatmap.2(t(f.mtx[which(r1$histology_type=="HGSOC"),171:179]), trace = "none", col = bluered(25),
          symbreaks=TRUE,breaks=seq(-3,3,length.out=26),
          margins = c(10, 12))
dev.off()
