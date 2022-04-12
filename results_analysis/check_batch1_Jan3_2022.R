rm(list = ls())

setwd("H:\\Jun_anonymized_dir\\OvaryCancer\\StromaReaction\\result_analysis\\data")


options(stringsAsFactors = FALSE)


r1 <- read.delim("BATCH_1_DATA_Jun_Jan3_2022.tsv")

plot(unlist(r1[1,197:388]), type = "l")
for(k in 1:nrow(r1)){
  lines(unlist(r1[k,197:388]), type = "l")
}

kres <- kmeans(r1[,197:388], centers = 4, nstart = 100)
print(table(kres$cluster))

par(mfrow = c(1,2))
boxplot(r1$tumor_area ~ kres$cluster, col = 1:4, notch = T)
boxplot(r1$stroma_area ~ kres$cluster, col = 1:4, notch = T)

pdf("batch1_check.pdf")
pval.vec <- mat.or.vec(196-17+1,1) + NA
for(i in 17:196){
  # i <- 17
  tmp.id <- colnames(r1)[i]
  ano.res <-  anova(lm(r1[,i] ~ kres$cluster))
  pval.vec[i]  <- ano.res$`Pr(>F)`[1]
  pval.str <- paste("ANOVA p =", format(pval.vec[i], scientific = T, digits = 3))
  boxplot(r1[,i] ~ kres$cluster, col = 1:4, notch = T, 
          main = paste(i,"-th feature:",tmp.id,"\n",pval.str))
}
dev.off()

kcentroid.mtx <- mat.or.vec(388-197+1,4)
for(j in 1:4){
  kcentroid.mtx[,j] <- apply(r1[kres$cluster==j,197:388], 2, median)
}

plot(kcentroid.mtx[,1], type = "l")
for(j in 1:4){
  lines(kcentroid.mtx[,j], type = "l", col = j, lwd = 2)
}

pc.res <- princomp((r1[,197:388]))
plot(pc.res)
plot(pc.res$sdev^2)
plot(cumsum(pc.res$sdev^2)/sum(pc.res$sdev^2), ylab = "accumulated variance% explained by PCs")
abline(h=0.9,v=9)

library(psych)
pairs.panels(cbind(r1$tumor_area, pc.res$scores[,1:7]),
             method = "spearman")
pairs.panels(cbind(r1$stroma_area, pc.res$scores[,1:7]),
             method = "spearman")

plot(r1$stroma_area, pc.res$scores[,1], ylab = "1st PC")


