library(scales)
library(corrplot)
library(psych)
library(dplyr)
require(survival)
library(ggplot2)

rm(list = ls())

options(stringsAsFactors = FALSE)

setwd("/anonymized_dir/Dataset/OvaryData/StromaReaction/survival")
output_dir <- "/anonymized_dir/Dataset/OvaryData/StromaReaction/survival_output"

################################
#  Summarized features
################################
r1 <- read.delim("batch1_metadata.tsv", sep = "\t")

non_color_feature_index <- c(18, 19, 20, 21, 22, 35, 36, 37, 38, 39, 40, 57, 58, 59, 60, 61, 62, 63, 76, 77, 78, 79, 80, 81, 98, 99, 100, 101, 102, 103, 104, 117, 118, 119, 120, 121, 122, 139, 140, 141, 142, 143, 144, 145, 158, 159, 160, 161, 162, 163, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194)
useful_r1 <- r1[, non_color_feature_index] # use only color unrelated features
#useful_r1 <- r1[,seq(18,196)] # use all features

feature_names <- names(useful_r1[0,])
tsv_header <- names(r1[0,])
N_feature <- length(feature_names)

################################
#  Read and match EHR records
################################
# match two tables
rclin <- read.csv("mayo_clinical_data_Feb12_2018.csv")
mch.clin.idx <- match(r1$clinic_num, rclin$clinic) # match(x,y) != match(y,x)
rclin.mch <- rclin[mch.clin.idx,]
# select RD0 and RD1/2
sel_case_idx_RD0 <- which(rclin.mch$histology=="1.2" & rclin.mch$residual==0)
sel_case_idx_RD12<- which(rclin.mch$histology=="1.2" & rclin.mch$residual!=0)

RD0_clinic_num <- r1$clinic_num[sel_case_idx_RD0]
RD12_clinic_num <- r1$clinic_num[sel_case_idx_RD12]

##################################
# draw correlation map
##################################
# Since the formula for calculating the correlation coefficient standardizes the variables, changes in scale or units of measurement will not affect its value. For this reason, normalizing will NOT affect the correlation.
feature_correlations <- cor(useful_r1) # calculate feature correlations
corrplot(feature_correlations, title="Overall correlation",mar=c(0,0,1,0),  addrect = 6, method = 'color',number.cex= 1/(ncol(feature_correlations)*2), col=colorRampPalette(c("blue","white","red"))(200),order = 'hclust')

# exclude tumor features for RD0
selected_idx <- list()
a <- colnames(useful_r1)
for(idx_fn in seq_along(feature_names)){ 
  if(! grepl("tumor", feature_names[idx_fn])) {
    selected_idx <- append(selected_idx, idx_fn)
  }
}
stroma_features <- useful_r1[sel_case_idx_RD0,unlist(selected_idx)]
feature_correlations <- cor(stroma_features)
corrplot(feature_correlations, title="RD0 stroma feature correlation", mar=c(0,0,1,0),addrect = 6, method = 'color',number.cex= 1/(ncol(feature_correlations)*2), col=colorRampPalette(c("blue","white","red"))(200),order = 'hclust')

# exclude tumor features for RD12
selected_idx <- list()
a <- colnames(useful_r1)
for(idx_fn in seq_along(feature_names)){ 
  if(! grepl("tumor", feature_names[idx_fn])) {
    selected_idx <- append(selected_idx, idx_fn)
  }
}
stroma_features <- useful_r1[sel_case_idx_RD12,unlist(selected_idx)]
feature_correlations <- cor(stroma_features)
corrplot(feature_correlations, title="RD12 stroma feature correlation",mar=c(0,0,1,0), addrect = 6, method = 'color',number.cex= 1/(ncol(feature_correlations)*2), col=colorRampPalette(c("blue","white","red"))(200),order = 'hclust')



# exclude stroma features for RD0
selected_idx <- list()
a <- colnames(useful_r1)
for(idx_fn in seq_along(feature_names)){ 
  if(! grepl("stroma", feature_names[idx_fn])) {
    selected_idx <- append(selected_idx, idx_fn)
  }
}
stroma_features <- useful_r1[sel_case_idx_RD0,unlist(selected_idx)]
feature_correlations <- cor(stroma_features)
corrplot(feature_correlations, title="RD0 tumor feature correlation", mar=c(0,0,1,0), addrect = 6, method = 'color',number.cex= 1/(ncol(feature_correlations)*2), col=colorRampPalette(c("blue","white","red"))(200),order = 'hclust')

# exclude stroma features for RD12
selected_idx <- list()
a <- colnames(useful_r1)
for(idx_fn in seq_along(feature_names)){ 
  if(! grepl("stroma", feature_names[idx_fn])) {
    selected_idx <- append(selected_idx, idx_fn)
  }
}
stroma_features <- useful_r1[sel_case_idx_RD12,unlist(selected_idx)]
feature_correlations <- cor(stroma_features)
corrplot(feature_correlations, title="RD12 tumor feature correlation", mar=c(0,0,1,0), addrect = 6, method = 'color',number.cex= 1/(ncol(feature_correlations)*2), col=colorRampPalette(c("blue","white","red"))(200),order = 'hclust')


########################################
# Calculate concordance of features in each patient(different tissue block)
# use standard deviation inside of a group (same patient but different tissue block). Other better measurements?
########################################
unique_clinic_num <- unique(r1$clinic_num)
sds <- matrix(nrow = length(unique_clinic_num), ncol = N_feature)
for(cn_idx in seq_along(unique_clinic_num)){
  selected_idx <- which(r1$clinic_num==unique_clinic_num[cn_idx])
  selected_sample <- useful_r1[selected_idx,]
  #print(selected_sample)
  if(length(selected_idx) == 2){ # calculate kappa
    sd_all_column <- sapply(selected_sample, sd)
    sds[cn_idx,] <- sd_all_column
    print(sds)
  }
  else if(length(selected_idx) == 3){
    sd_all_column <- sapply(selected_sample, sd)
    sds[cn_idx,] <- sd_all_column
  }
  else{
    sds[cn_idx,] <- rep(-1, N_feature)
  }
}
sds_table <- data.frame(sds)
colnames(sds_table) <- feature_names 
sds_table$clinic_num <- unique_clinic_num # add clinical number to the table



################################
#  histogram RD0 and RD1/2
################################
save_to <- file.path(output_dir, "RGB_distribution.pdf")
pdf(save_to)
for(k in 1:N_feature){
  tmp.feature <- feature_names[k]
  x.vec_rd0 <- r1[sel_case_idx_RD0, tmp.feature]
  x.vec_rd12 <- r1[sel_case_idx_RD12, tmp.feature]
  data_table <- data.frame(
    res=factor(c(rep("RD0", length(sel_case_idx_RD0)), rep("RD12", length(sel_case_idx_RD12)))),
    fea_n=c(x.vec_rd0, x.vec_rd12))
  print(ggplot(data_table, device="pdf" , aes(x=fea_n, fill=res, color=res)) + geom_histogram( position="identity", alpha=0.5) + 
    labs(title="Feature histogram plot",x=tmp.feature, y = "Count") + geom_density(alpha=0.6))
  # save_to <- file.path(output_dir, paste(tmp.feature, "_hist.png"))
  # ggsave(save_to, plot = myPlot)
}
dev.off()
###############################
#  survival analysis or RD0 and RD1/2
###############################
surv.obj <- Surv(rclin.mch$t_fu_mth, rclin.mch$vital)

binary.x.vec <- rep(TRUE, nrow(r1))
binary.x.vec[sel_case_idx_RD12] <- FALSE
fit <- survfit(surv.obj[c(sel_case_idx_RD0, sel_case_idx_RD12)] ~ binary.x.vec[c(sel_case_idx_RD0, sel_case_idx_RD12)])
summary(fit)
save_to <- file.path(output_dir, "RD0_RD12_overall_survival.pdf")
pdf(save_to)
plot(fit, col=1:2, main = "RD0/RD12 Survival", lwd = 3,
     xlab = "months", ylab = "Prob. of overall survival") 
legend("topright", c("RD1/2", "RD0"),col = 1:2, lwd = 3, cex = 1.5)
dev.off()
###############################
pval.vec <- rep(1, N_feature)
save_to <- file.path(output_dir, paste("RD0_survival.pdf"))
pdf(save_to)
for(k in 1:N_feature){
  tmp.feature <- feature_names[k]
  x.vec <- r1[, tmp.feature]
  binary.x.vec <- x.vec>median(x.vec, na.rm = T)
  
  sum.res <- summary(coxph(surv.obj[sel_case_idx_RD0] ~ binary.x.vec[sel_case_idx_RD0]))
  pval.vec[k] <- sum.res$waldtest["pvalue"]
  
  fit <- survfit(surv.obj[sel_case_idx_RD0] ~ binary.x.vec[sel_case_idx_RD0])
  
  
  plot(fit,
       col=1:2, main = paste("RD0", tmp.feature,"\n > median?"), lwd = 3,
       xlab = "months", ylab = "Prob. of overall survival")
  legend("topright", levels(as.factor(binary.x.vec)) ,col = 1:2, lwd = 3, cex = 1.5)
  
}
dev.off()
############################################
save_to <- file.path(output_dir, paste("RD12_survival.pdf"))
pdf(save_to)
pval.vec <- rep(1, N_feature)
for(k in 1:N_feature){
  tmp.feature <- feature_names[k]
  x.vec <- r1[, tmp.feature]
  binary.x.vec <- x.vec>median(x.vec, na.rm = T)
  
  sum.res <- summary(coxph(surv.obj[sel_case_idx_RD12] ~ binary.x.vec[sel_case_idx_RD12]))
  pval.vec[k] <- sum.res$waldtest["pvalue"]
  
  fit <- survfit(surv.obj[sel_case_idx_RD12] ~ binary.x.vec[sel_case_idx_RD12])
  
  
  print(plot(fit,
       col=1:2, main = paste("RD12", tmp.feature,"\n > median?"), lwd = 3,
       xlab = "months", ylab = "Prob. of overall survival"))
  legend("topright", levels(as.factor(binary.x.vec)) ,col = 1:2, lwd = 3, cex = 1.5)
  
}
dev.off()




