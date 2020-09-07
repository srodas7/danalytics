library(dplyr)
library(car)
library(psych)
library(rpart)
library(SnowballC)
library(dplyr)
library(e1071)
library(base)
library(Matrix)
library(data.table)
library(stats)
library(caret)
library(MASS)
library(ggplot2)
library(psych)
library(partykit)
library(lattice)
library(klaR)
library(C50)
library(gmodels)
library(Matrix)
library(class)
library(NLP)
library(tm)
library(wordcloud)
library(vcd)
library(adabag)
library(ipred)

###################### Regression Tree to Explore Borrowing Behavior ########################
########################## by Welfare Indicator on a Sector ###############################

#------------------ Regions that are not deprived of schooling and have longer years of
#------------------ of schooling may invest more in education

# education -- loan_amount~schooling, school attendance

# reduce dataset to only include loans requested for the Education Sector
edu <- t[t$sector=="Education",]

# linear regression to start off with
edu_lm <- lm(loan_amount~ED_SchoolingPop_Md, data = edu)
edu_lm2 <- lm(loan_amount~sqrt(mean_years_of_schooling), data = edu)
#### Result: not much insight at all
# Decision --> Try Regression Treee

# Regression Tree on full dataset
edu_rp <- rpart(loan_amount~mean_years_of_schooling+ED_SchoolingPop_Md, data = edu)
rpart.plot(edu_rp)

edu_rp
# n= 20034 
# 
# node), split, n, deviance, yval
# * denotes terminal node
# 
# 1) root 20034 24353430000  951.0158  
# 2) ED_SchoolingPop_Md< 4.125 7856  6537984000  745.2839 *
#   3) ED_SchoolingPop_Md>=4.125 12178 17268430000 1083.7330  
# 6) mean_years_of_schooling< 7.598858 8952  7996098000  951.4662  
# 12) ED_SchoolingPop_Md< 9.3 2131  2201674000  577.6044  
# 24) ED_SchoolingPop_Md>=8.675 1064    61136790  296.9220 *
#   25) ED_SchoolingPop_Md< 8.675 1067  1973124000  857.4977  
# 50) ED_SchoolingPop_Md< 8.1 807   921250400  326.8278 *
#   51) ED_SchoolingPop_Md>=8.1 260   119234500 2504.6150 *
#   13) ED_SchoolingPop_Md>=9.3 6821  5403513000 1068.2670 *
#   7) mean_years_of_schooling>=7.598858 3226  8681139000 1450.7670  
# 14) mean_years_of_schooling>=7.833811 3044  3596092000 1176.1660 *
#   15) mean_years_of_schooling< 7.833811 182  1016477000 6043.5440 *

summary(edu_rp)


#------------------ Summary

# Countries that are more deprived of an education and have lower schooling years tend to
# invest more into education than countries who have high and long school attendance.
# Large amount of loans requested for education are requested by those regions where more
# than 9.3 of the population is deprived of schooling, and where the mean years of schooling
# are less than 7.6



