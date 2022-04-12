rm(list = ls())

options(stringsAsFactors = FALSE)

setwd("H:\\Jun_anonymized_dir\\OvaryCancer\\StromaReaction\\result_analysis")

#r1 <- read.delim("data/Jun_HE_OC_TSR_cases_metadata_12_08_2021.txt", sep = "\t")
# r1 <- read.delim("data/batch1_metadata.tsv", sep = "\t")
r1 <- read.delim("data/batch3_metadata.tsv", sep = "\t")
#r1 <- r1[which(r1$fresh_cut),]



hist(r1$fibrosis_2)

plot(r1$fibrosis_2, r1$orientation_2)
plot(r1$fibrosis_2, r1$cellularity_2)

colnames(r1)

head(r1[,seq(181,195)])


sub.mtx <- r1[,seq(181,195)]

#sub.mtx <- r1[,seq(187,195)]
sub.mtx <- r1[,c(182,183,seq(187,195))]
sub.mtx <- r1[,c(72,73,seq(187,195))]

sub.mtx <- r1[,seq(17,23)]
sub.mtx <- r1[,seq(24,31)]
sub.mtx <- r1[,seq(32,42)]
sub.mtx <- r1[,seq(43,51)]
sub.mtx <- r1[,seq(52,61)]
sub.mtx <- r1[,seq(62,71)]
sub.mtx <- r1[,seq(72,81)]
sub.mtx <- r1[,seq(82,91)]
sub.mtx <- r1[,seq(92,101)]
sub.mtx <- r1[,seq(102,111)]
sub.mtx <- r1[,seq(112,121)]
sub.mtx <- r1[,seq(122,131)]
sub.mtx <- r1[,seq(132,141)]
sub.mtx <- r1[,seq(142,151)]
sub.mtx <- r1[,seq(152,161)]
sub.mtx <- r1[,seq(162,171)]
sub.mtx <- r1[,seq(172,181)]

library(psych)
pairs.panels(sub.mtx, method = "spearman")

plot(r1[,174], r1[,180])
sub.info <- r1[which(r1[,174]<0.1 & r1[,180]<0.05),]

#=== excluding outlier batch
#r1 <- r1[-which(r1[,174]<0.1 & r1[,180]<0.05),]

plot(r1[,174], r1[,180])
sub.mtx <- r1[,seq(172,181)]
library(psych)
pairs.panels(sub.mtx, method = "spearman")



rclin0 <- read.csv("data/mayo_clinical_data_Feb12_2018.csv")
mch.clin.idx <- match(r1$clinic_num, rclin0$clinic)
rclin.mch <- rclin0[mch.clin.idx,]

#batch1 <- read.csv("H:\\Jun_anonymized_dir\\OvaryCancer\\StromaReaction\\result_analysis\\batch1_R0_R12.csv")
#batch1_idx <- match(batch1$case_id,  rclin0$clinic)
#rclin.mch <- rclin0[batch1_idx,]

require(survival)
surv.obj <- Surv(rclin.mch$t_fu_mth, rclin.mch$vital)

summary(coxph(surv.obj ~ r1$cellularity_2))
summary(coxph(surv.obj ~ r1$tumor_density_mean))
summary(coxph(surv.obj ~ r1$fresh_cut))
#summary(coxph(surv.obj ~ r1$histology_type))

table(r1$fresh_cut, r1$histology_type)
table(r1$histology_type, rclin.mch$histology)

feature.name.vec <- colnames(r1)[17:195]

#sel.case.idx <- which(rclin.mch$histology=="1.2" & rclin.mch$debulkstatus=="optimal")
#sel.case.idx <- which(rclin.mch$histology=="1.2" & rclin.mch$debulkstatus=="suboptimal")


#=== stratifying to RD0 (residual disease = 0cm) patients
sel.case.idx <- which(rclin.mch$histology=="1.2" & rclin.mch$residual==0)
#a <- r1$deidentified_id
#aa <- a[sel.case.idx]
#write.table(aa, file = "RD0_cases.csv", sep = ",", col.names = NA)
#write.table(a, file = "all_cases.csv", sep = ",", col.names = NA)

#=== stratifying to RD1 or RD2 (residual disease <1cm or >1cm, but not 0cm) patients
#sel.case.idx_a <- which(rclin.mch$histology=="1.2" & rclin.mch$residual!=0)
#aaa <- a[sel.case.idx_a]
#write.table(aaa, file = "RD1.2_cases.csv", sep = ",", col.names = NA)



N.feature <- length(feature.name.vec)
pval.vec <- binary.pval.vec <- rep(1, N.feature)
names(pval.vec) <- names(binary.pval.vec) <- feature.name.vec
for(k in 1:N.feature){
  #k <- 80  
  tmp.feature <- feature.name.vec[k]
  x.vec <- r1[, tmp.feature]
  
  cat(k, "\n", tmp.feature,"\n")
  sum.res <- summary(coxph(surv.obj[sel.case.idx] ~ x.vec[sel.case.idx]))
  print(sum.res)
  pval.vec[k] <- sum.res$waldtest["pvalue"]
  
  binary.x.vec <- x.vec>median(x.vec, na.rm = T)
  binary.sum.res <- summary(coxph(surv.obj[sel.case.idx] ~ binary.x.vec[sel.case.idx]))
  print(binary.sum.res)
  binary.pval.vec[k] <- binary.sum.res$waldtest["pvalue"]
  
  plot(survfit(surv.obj[sel.case.idx] ~ binary.x.vec[sel.case.idx]),
       col = 1:2, main = paste(tmp.feature,"\n > median?"), lwd = 3,
       xlab = "months", ylab = "Prob. of overall survival")
  legend("topright", levels(as.factor(binary.x.vec)),
         col = 1:2, lwd = 3, cex = 1.5)
}

#=== sort the most significant digPath features
head(sort(binary.pval.vec))
head(order(binary.pval.vec))



#==== checking whether TSR concordant within same patient

boxplot(r1$orientation_2 ~ r1$clinic_num)

clin.ids.w.replicates <- names(which(table(r1$clinic_num)>=2))
N.clinc <- length(clin.ids.w.replicates)
avg.vec <- mad.vec <- mat.or.vec(N.clinc,1)
for(j in 1:N.clinc){
  #j <- 1
  tmp.clinic.id <- clin.ids.w.replicates[j]
  sel.replicate.idx <- which(r1$clinic_num==tmp.clinic.id)
  
  
  ###==random sampling
  #sel.replicate.idx <- sample(seq(1, N.clinc), length(sel.replicate.idx))
  ###==random sampling
  
  #avg.vec[j] <- median(r1$orientation_2[sel.replicate.idx])
  #mad.vec[j] <- mad(r1$orientation_2[sel.replicate.idx])
  
  avg.vec[j] <- median(r1$cellularity_2[sel.replicate.idx])
  mad.vec[j] <- mad(r1$cellularity_2[sel.replicate.idx], constant = 1)
  
  #avg.vec[j] <- median(r1$fibrosis_2[sel.replicate.idx])
  #mad.vec[j] <- mad(r1$fibrosis_2[sel.replicate.idx])
  
  #avg.vec[j] <- median(r1$tumor.Nucleus..Perimeter_mean[sel.replicate.idx])
  #mad.vec[j] <- mad(r1$tumor.Nucleus..Perimeter_mean[sel.replicate.idx])
}


plot(avg.vec, log2(avg.vec/mad.vec), 
     xlab = "average", ylab = "SNR")
abline(h=0, col = "red")
abline(h=1, col = "blue")
plot(avg.vec, mad.vec)

hist(log2(avg.vec/mad.vec),20, 
     main = 
       paste("SNR>1, N =",length(which(log2(avg.vec/mad.vec)>1)),
             "\n avg.SNR =", round(mean(log2(avg.vec/mad.vec), na.rm = T),1))
     )
abline(v=1, col = "red", lwd =3)





mch.dedup.idx <- match(clin.ids.w.replicates, rclin0$clinic)
rclin.mch.dedup <- rclin0[mch.dedup.idx,]
dedup.surv.obj <- Surv(rclin.mch.dedup$t_fu_mth, rclin.mch.dedup$vital)

coxph(dedup.surv.obj ~ avg.vec)
coxph(dedup.surv.obj ~ avg.vec>median(avg.vec))

avgQ.vec <- paste("Qtl",findInterval(avg.vec, 
             vec = quantile(avg.vec, probs = seq(0, 1, by = 0.25)), rightmost.closed = T))
summary(coxph(dedup.surv.obj ~ avgQ.vec ))


coxph(dedup.surv.obj ~ mad.vec)
summary(coxph(dedup.surv.obj ~ mad.vec>0.4))
summary(coxph(dedup.surv.obj ~ mad.vec<0.1))

madQ.vec <- paste("Qtl",findInterval(mad.vec, 
                             vec = quantile(mad.vec, probs = seq(0, 1, by = 0.25)), rightmost.closed = T))

#madQ.vec <- paste("Qtl",findInterval(mad.vec, 
#                                     vec = c(0.1, 0.36, 0.8), rightmost.closed = T))
summary(coxph(dedup.surv.obj ~ madQ.vec))