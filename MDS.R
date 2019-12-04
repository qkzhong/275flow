#Flow MDS
#
library(magrittr)
library(dplyr)
library(ggpubr)
library(readr)
#read document
dat <- read_csv('final_merge.csv')
dat1 <- subset(dat, select= c(-ID, -ID_1,-NetScore, -RATIO, -Lmax,-Vmax, -Lmean))
dat2 = t(dat1)
# Cmpute MDS
mds <- dat2 %>%
  dist() %>%          
  cmdscale() %>%
  as_tibble()
colnames(mds) <- c("Dim.1", "Dim.2")
# Plot MDS
ggscatter(mds, x = "Dim.1", y = "Dim.2", 
          label = rownames(dat2),
          size = 1,
          repel = TRUE)

# K-means clustering
clust <- kmeans(mds, 4)$cluster %>%
  as.factor()
mds <- mds %>%
  mutate(groups = clust)
# Plot and color by groups
ggscatter(mds, x = "Dim.1", y = "Dim.2", 
          label = rownames(dat2),
          color = "groups",
          palette = "jco",
          size = 1, 
          ellipse = TRUE,
          ellipse.type = "convex",
          repel = TRUE)
