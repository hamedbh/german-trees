print_conf_matrix <- function(conf_matrix) {
    stopifnot(
        "ggplot2" %in% installed.packages(), 
        "viridis" %in% installed.packages()
    )
    ggplot(data = as.data.frame(conf_matrix[["table"]]), 
           aes(x = Reference, y = Prediction, fill = Freq)) +
        geom_tile() + 
        scale_fill_viridis(option = "B", space = "Lab") +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                         size = 10, hjust = 1), 
              axis.text.y = element_text(size = 10), 
              axis.title.x = element_text(size = 10), 
              axis.title.y = element_text(size = 10), 
              panel.grid.major = element_blank(), 
              panel.border = element_blank(), 
              panel.background = element_blank(), 
              axis.ticks = element_blank()) +
        coord_fixed() + 
        geom_text(aes(Reference, Prediction, label = Freq), 
                  colour = "grey", size = 3)
}