library(dplyr)
library(ggplot2)
library(readr)


rss_output <- read_delim("data/rss_output.csv",";", escape_double = FALSE, trim_ws = TRUE)
rss_output %>% ggplot(aes(x=iteration, y=rss)) + geom_point() + ggtitle("RSS for each Iteration")
income %>% ggplot(aes(x=X1, y=X2)) + geom_point() + ggtitle("Years of Experience by Income") + xlab("years") + ylab("Income")


rss_output %>% filter(iteration <= 125) %>% ggplot(aes(x=iteration, y=rss)) + geom_point() + ggtitle("RSS for each Iteration less than 126")
