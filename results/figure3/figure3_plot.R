library(dplyr)
library(gtools)
library(ggsci)
library(ggpubr)
library(ggplot2)
library(gridExtra)
library(extrafont)
library(rstatix)

###### function ######

theme_Publication <- function(base_size=14, base_family="helvetica") {
  library(grid)
  library(ggthemes)
  (theme_foundation(base_size=base_size, base_family=base_family)
    + theme(plot.title = element_text(face = "bold", size = rel(1.0), hjust = 0.5),
            text = element_text(),
            panel.background = element_rect(colour = NA),
            plot.background = element_rect(colour = NA),
            panel.border = element_rect(colour = NA),
            axis.title = element_text(face = "bold",size = rel(0.8)),
            axis.title.y = element_text(angle=90,vjust =2),
            axis.title.x = element_blank(),
            axis.text = element_text(),
            axis.line = element_line(colour="black"),
            axis.ticks = element_line(),
            panel.grid.major = element_line(colour="#f0f0f0"),
            panel.grid.minor = element_blank(),
            legend.position = "none",
            plot.margin=unit(c(10,5,5,5),"mm"),
            strip.background=element_rect(colour="#f0f0f0",fill="#f0f0f0"),
            strip.text = element_text(face="bold")
    ))
  
}

###### Code ######

setwd("/home/kwlee/Projects_gflas/Team_BI/Projects/DACO")

data_cas9<-read.delim(file.path("results/data","cas9_word2vec.csv"), header = T, sep = "\t",stringsAsFactors = F) %>% filter(Spearman > 0.5)
data_cas12a<-read.delim(file.path("results/data","cas12a_word2vec.csv"), header = T, sep = "\t",stringsAsFactors = F) %>% filter(Spearman > 0.5)

# kmer<-unique(data_cas9$kmer)
# c<-combinations(length(kmer),2, 1:length(kmer))
# comparsion_list <- matrix(kmer[c], ncol=2) %>% aaply(1, list)

train_data = c("WT-Kim", "WT-Wang", "WT-Xiang", "HF1-Wang", "esp-Wang", "hl60-wang")
figure_list<-list()

for (i in c(2, 4)){
  tmp <- data_cas9 %>% filter(Data == train_data[i])
  stat.test <- tmp %>% wilcox_test(Spearman ~ kmer, p.adjust.method = "fdr", detailed = T) %>% add_xy_position()
  figure_list[[i]] <- ggviolin(tmp, x = "kmer", y = "Spearman", fill = "kmer", add="boxplot", add.params = list(fill = "white")) + labs(y="Spearman correlation", title = train_data[i]) + 
    stat_pvalue_manual(stat.test, hide.ns = TRUE) + scale_fill_nejm() + theme_Publication()
  
}
g<-ggarrange(figure_list[[2]],figure_list[[4]], labels = c("A", "B"), ncol = 2, nrow = 1)
ggsave(plot = g, file=file.path(filename="results/figure3", "figure3.tiff"), scale = 1, width = 8.5, height = 4.5, dpi = 150, units="in", dev="tiff")
ggsave(plot = g, file=file.path(filename="results/figure3", "figure3.png"), scale = 1, width = 8.5, height = 4.5, dpi = 150, units="in", dev="png")
