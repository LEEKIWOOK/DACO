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
    + theme(plot.title = element_text(face = "bold", size = rel(0.8), hjust = 0.5),
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
cas9_comp<-read.delim(file.path("results/data","cas9_comp.csv"), header = T, sep = "\t",stringsAsFactors = F) %>% filter(Spearman > 0.1)
cas12_comp<-read.delim(file.path("results/data","cas12a_comp.csv"), header = T, sep = "\t",stringsAsFactors = F) %>% filter(Spearman > 0)
f6_data<-rbind(cas9_comp, cas12_comp)
train_data = c("WT-Kim", "WT-Wang", "WT-Xiang", "HF1-Wang", "esp-Wang", "AsCas12a-Kim")
tools = c("DeepSpCas9", "DeepHF-WT", "CrisprOn", "DeepHF-HF1", "DeepHF-esp", "DeepCpf1", "DACO")
figure_list<-list()

for (i in 1:6){
  
  tmp <- f6_data %>% filter(Train_data == train_data[i] & Test_data != train_data[i]) 
  #stat.test <- tmp %>% wilcox_test(Spearman ~ Model+Test_data, detailed = T) %>% add_xy_position
  tmp$Model<-factor(tmp$Model, levels=c(tools[i], tools[7]))
  
  figure_list[[i]] <- ggplot(tmp, aes(x=Model, y=Spearman, fill=Test_data)) + geom_bar(stat="identity", position="dodge") + labs(y="Spearman correlation", title = train_data[i]) + scale_fill_nejm() + theme_Publication()
    #stat_pvalue_manual(stat.test) 
}

g<-ggarrange(figure_list[[1]],figure_list[[2]],figure_list[[3]],figure_list[[4]],figure_list[[5]],figure_list[[6]], labels = c("A", "B", "C", "D", "E", "F"), ncol = 3, nrow = 2)
ggsave(plot = g, file=file.path(filename="results/figure6", "figure6.tiff"), scale = 1, width = 8.5, height = 4.5, dpi = 150, units="in", dev="tiff")
ggsave(plot = g, file=file.path(filename="results/figure6", "figure6.png"), scale = 1, width = 8.5, height = 4.5, dpi = 150, units="in", dev="png")
