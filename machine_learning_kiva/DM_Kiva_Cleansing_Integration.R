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


###################### Data Cleansing #############################
# load files
loans <- read.csv(file.choose())
str(loans)
region <- read.csv(file.choose())
theme <- read.csv(file.choose())
theme_region <- read.csv(file.choose())

#------------------Kiva Loans Dataset
loans <- read.csv(file.choose())
attach(loans)
use <- as.character(use)
region <- as.character(region)
posted_time <- as.Date.character(posted_time)
disbursed_time <- as.Date.character(disbursed_time)
funded_time <- as.Date.character(funded_time)
tags <- as.character(tags)
date <- as.Date.character(date)
borrower_genders <- as.character(borrower_genders)

#function to identify columns with NA values
getnulls <- function(df){
  a <- c()
  b <- NULL
  null_ind <- NULL
  for (i in c(1:ncol(df))) {
    a <- length(df[is.na(df[i]), i])
    b <- data.frame(Column = i, NullCount = a)
    null_ind <- rbind(null_ind, b)
  }
  return(null_ind)
}

#cleanse the borrowing_gender column
#function to get the substring of the end of the string
endasgender <- function(g){
  gender <- c()
  for(i in 1:nrow(g[18])) {
    gender[i] <- substring(g[i, 18], nchar(g[i, 18])-5)
  }
  return(gender)
}

#get the vector of substrings, add it as new column to vector, check table counts
substrvector <- endasgender(loans)
head(substrvector)
table(substrvector)
substrvector
#     , male female   male 
#4221  19122 513152 134710
loans$gender <- substrvector
head(loans)

#based on new gender column, create a new borrower_genders_test column to get the values for the genders
loans$borrower_genders_test <- ifelse(loans$gender == ", male", "male", ifelse(loans$gender == "male", "male",ifelse(loans$gender == "female", "female", NA)))
table(loans$borrower_genders_test)
#female   male 
#513152 153832 
# 513152 + 153832
#666984
a <-length(loans[loans$gender == "", 21])
666984 + a
#671205
colnames(loans)[22] <- "borrower_genders_clean"

# Explore Loan Use: create a corpus, cleanse and create a data matrix
use_corpus <- VCorpus(VectorSource(loans$use))
print(use_corpus)
nrow(loans)
inspect(use_corpus[1:10])
as.character(use_corpus[[9]])
lapply(use_corpus[1:10], as.character)

#Preprocess before DataTerm Matrix
use_clean_prepro <- tm_map(use_corpus, content_transformer(tolower))
use_clean_prepro <- tm_map(use_corpus, removeNumbers)
use_clean_prepro <- tm_map(use_corpus, removeWords, stopwords())
use_clean_prepro <- tm_map(use_corpus, stemDocument)
use_clean_prepro <- tm_map(use_corpus, stripWhitespace)
use_dtm_pp <- DocumentTermMatrix(use_clean_prepro)
wordcloud(use_clean_prepro, min.freq = 1000, random.order = FALSE)
use_freq_words <- findFreqTerms(use_dtm_pp, 100)

#Preprocess during DocumentTermMatrix()
use_dtm_np <- DocumentTermMatrix(use_corpus, control = list(tolower = TRUE, removeNumbers = TRUE, stopwords = TRUE, removePunctuation = TRUE, stemming = TRUE))
#Get frequent terms vector
use_freq_words_np <- findFreqTerms(use_dtm_np, 50)
length(use_freq_words_np)
use_freq_words_np_df <- data.frame(FreqWords = use_freq_words_np)
loansuse <- loans %>% select(use) %>% group_by(use) %>% summarize(usect = n()) %>% arrange(desc(usect))

#Write data frame for cleansed Kiva Loans and for frequent words
write.table(use_freq_words_np_df, file = "kiva_loans_use_freqwords.csv", quote = FALSE, sep = ",", col.names = TRUE, row.names = FALSE)
write.table(loans, file = "kiva_loans_cleansed.csv", quote = FALSE, sep = ",", col.names = TRUE, row.names = FALSE)
write.csv(kiva_loans, file="kiva_loans_cleansed.csv")

#cleaning region
l_region_factor <- as.factor(loans$region)
l_region_factor <- tolower(l_region_factor)
l_region_factor <- removeNumbers(l_region_factor)
l_region_factor <- removePunctuation(l_region_factor)
loans_region_clean <- data.frame(loans, RegionClean = l_region_factor)
loans_region_clean <- data.frame(loans[-c(6,17,18,21)], RegionClean = l_region_factor)

##remove use, tags, borrower_genders, gender (mid-cleansed)
loans_region_clean <- loans_region_clean[,c(1:8,19,9:18)]
loans_region_clean$RegionClean2 <- as.character(loans_region_clean$RegionClean)
loans_region_clean$RegionClean2 <- trimws(loans_region_clean$RegionClean)
loans_region_clean$RegionClean2 <- as.factor(loans_region_clean$RegionClean2)
head(levels(loans_region_clean$RegionClean2), n=50)


#explore kiva loans csv
loans %>% group_by(country) %>% summarize(LoanCt = length(country), Sum = sum(loan_amount), StdDev = sd(loan_amount), Mean = mean(loan_amount, na.rm = TRUE), Median = median(loan_amount), MinAmt = min(loan_amount), MaxAmt = max(loan_amount)) %>% top_n(20,wt=LoanCt) %>% ungroup() %>% ggplot(aes(x = reorder(country, LoanCt), y = LoanCt)) + geom_bar(stat="identity", fill="lightblue", colour="black") + coord_flip()

##Is there a common industry sector that loans get requested for?
loanct_by_sector <- loans %>% group_by(sector) %>% summarize(ct = length(activity)) %>% ungroup() %>% ggplot(aes(x = reorder(sector,ct), y = ct)) + geom_bar(stat="identity", fill="blue", colour = "black") + coord_flip() + theme_bw(base_size = 10) + labs(x="Sector",y="Loan Count")

##loans per activity
loanct_by_sector_activity <- loans %>% group_by(activity) %>% summarize(ct = length(id)) %>% top_n(50, wt=ct) %>% ungroup() %>% ggplot(aes(x = reorder(activity,ct), y = ct)) + geom_bar(stat="identity", fill= "blue", colour = "black") + coord_flip() + theme_bw(base_size = 10) + labs(x="Activity",y="Loan Count")


#------------------Kiva Region Dataset
str(region)
head(region[region$country == "El Salvador", ])
#there are different MPI for regions
alb <- region[region$ISO == "ALB", ]
#some have NAs instead, probably bc there are no loans from those places
getnulls(region)
#Column NullCount
#1      1         0
#2      2         0
#3      3         0
#4      4         0
#5      5         0
#6      6      1788
#7      7         0
#8      8      1880
#9      9      1880
#there are 103 countries in region dataset, but only 87 are used in loans dataset


#------------------Kiva Theme Dataset
#Some exploring on theme dataset
dup <- duplicated(theme$Loan.Theme.ID)
dub_df <- theme[dup, ]
dim(dub_df)
dup2 <- duplicated(theme$Loan.Theme.Type)
dup_df <- theme[dup2, ]
head(dup_df %>% arrange(dup_df$Loan.Theme.Type), n = 50)
dim(dub_df)
n <- theme[theme$Loan.Theme.ID == "", ]
nrow(n)
#[1] 14813
n <- theme[theme$Loan.Theme.Type == "", ]
nrow(n)
#[1] 14813
n <- theme[is.na(theme$Partner.ID), ]
nrow(n)
#[1] 14813


#------------------ OPHI MPI Dataset
mpi <- read.csv(file.choose())
mpi <- data.frame(mpi[,1:16])
mpi$ISO <- as.character(mpi$ISO)
region$ISO <- as.character(region$ISO)
region$MPIcountry <- NA


###################### Imputing Missing Values #############################

#------------------ Kiva Loans Missing Values
kiva_loans <- loans
# Remove original gender, halfway cleansed gender, and add cleansed region from loans_region_cleansed dataset
kiva_loans <- kiva_loans[,-c(18,21)]
kiva_loans <- cbind(kiva_loans, "RegionClean" = loans_region_clean$RegionClean2)
names(kiva_loans)[21] <- "region_clean"
names(kiva_loans)[20] <- "borrower_gender"
kiva_loans <- kiva_loans[,-9]

# Get most frequent factor of gender by country/region or do a decision tree on gender to see 
kiva_loans_nb <- naiveBayes(borrower_gender~country+sector+activity, data = kiva_loans)
kiva_loans_dt <- C5.0(borrower_gender~country+sector+activity, data = kiva_loans)
### Country is used 100%
# Nevermind, just use the most frequent gender and populate the rest, it was less than 2%
table(kiva_loans$country, kiva_loans$borrower_gender)
prop.table(table(kiva_loans$borrower_gender))
prop.table(table(kiva_loans$borrower_gender))*100
#female     male 
#76.93618 23.06382 
### female it is
kiva_loans$borrower_gender[is.na(kiva_loans$borrower_gender)] <- "female"
table(kiva_loans$borrower_gender)
#female   male 
#517373 153832
prop.table(table(loans$borrower_genders_clean))
#female      male 
#0.7693618 0.2306382
prop.table(table(kiva_loans$borrower_gender))
#female      male 
#0.7708122 0.2291878

write.table(kiva_loans, file="kiva_loans_final.csv", quote = FALSE, sep = ",", col.names = TRUE, row.names = FALSE)

#------------------ Country Stats Missing Values
country_stats <- read.csv(file.choose())
### Create lm that can predict the missing number for population below poverty line because that attribute has 22 null values, all others only have 4-6 nulls (ok)
popbpl_lm <- lm(population_below_poverty_line~population+hdi+life_expectancy+expected_years_of_schooling+mean_years_of_schooling+gni,data = country_stats)
stepwise(popbpl_lm)
popbpl_lm_red <- lm(formula = population_below_poverty_line ~ hdi + expected_years_of_schooling + gni, data = country_stats)
pop <- powerTransform(cbind(population_below_poverty_line,hdi,expected_years_of_schooling,
                            gni)~1, data = country_stats)
pop <- powerTransform(cbind(population_below_poverty_line,hdi,expected_years_of_schooling,
                            gni)~1, data = country_stats)
summary(pop)
anova(popbpl_lm, popbpl_lm_red)
#p-value is > alpha. Fail to reject the null, so we keep g_reduced
popbpl_lm_red_tr <- lm(sqrt(population_below_poverty_line)~(hdi^2)+(expected_years_of_schooling^2)+log(gni), data = country_stats)
summary(popbpl_lm_red_tr)
popbpl_lm_red_tr <- lm(log(population_below_poverty_line)~(hdi^2)+(expected_years_of_schooling^2)+log(gni), data = country_stats)
summary(popbpl_lm_red_tr)
popbpl_lm_red_tr <- lm(log(population_below_poverty_line)~hdi+expected_years_of_schooling+gni, data = country_stats)
summary(popbpl_lm_red_tr)
popbpl_lm_red_tr <- lm(population_below_poverty_line~(hdi^2)+(expected_years_of_schooling^2)+log(gni), data = country_stats)
summary(popbpl_lm_red_tr)
### The following one gave the better r2 and p-values
popbpl_lm_red_tr <- lm(population_below_poverty_line~(hdi^2)+(expected_years_of_schooling^2)+gni, data = country_stats)
summary(popbpl_lm_red_tr)

b <- predict(object = popbpl_lm_red_tr, newdata = country_stats[,c(8,10,12)])
dim(country_stats)
tail(country_stats)
length(b)
testing <- cbind(country_stats, Pred = b)

pred_ind <- function(df){
  for(i in c(1:nrow(df))){
    if(is.na(df[i,7])){
      df[i,7] <- df[i,14]
    }
  }
  return(df)
}

trywhy <- pred_ind(testing)
head(trywhy)
tail(trywhy)
getnulls(trywhy)
country_stats <- trywhy[,-14]

write.csv(country_stats, file="country_stats_cleansed_nonulls.csv")
write.table(country_stats, file="kiva_country_stats_cleansed.csv", quote = FALSE, sep = ",", col.names = TRUE, row.names = FALSE)



#------------------ MPI Missing Values
mmpi <- read.csv(file.choose()) #this is the Country_Region_MPI.csv file
names(mmpi)
#[1] "ISO"             "Country"         "SubNatRegion"    "WorldRegion"     "CountryMPI"      "RegionMPI"      
#[7] "ED_SchoolingPop" "ED_SchoolAttPop" "HE_ChildMortPop" "HE_NutrPop"      "LS_ElectrPop"    "LS_ImprSanitPop"
#[13] "LS_DrinkWatPop"  "LS_FloorPop"     "LS_CookFuelPop"  "LS_AssetOwnPop" 
#8-11,14,15
#do the ave on country, do if else, do ave on region, do if else
#"ED_SchoolAttPop","HE_ChildMortPop","HE_NutrPop","LS_ElectrPop","LS_FloorPop","LS_CookFuelPop"

# "ED_SchoolAttPop" column 8 --> 17 nulls
eight_c_df <- aggregate(data=mmpi, ED_SchoolAttPop~Country, median, na.rm=TRUE)
eight_c_v <- ave(mmpi$ED_SchoolAttPop, mmpi$Country, FUN=function(x) median(x, na.rm = TRUE))
eight_r_v <- ave(mmpi$ED_SchoolAttPop, mmpi$WorldRegion, FUN=function(x) median(x, na.rm = TRUE))
mmpi$ED_SchoolAttPop <- ifelse(is.na(mmpi$ED_SchoolAttPop), eight_c_v, mmpi$ED_SchoolAttPop)
#getnulls = 17
mmpi$ED_SchoolAttPop <- ifelse(is.na(mmpi$ED_SchoolAttPop), eight_r_v, mmpi$ED_SchoolAttPop)
#getnulls = 0

# "HE_ChildMortPop" column 9 --> 24 nulls
nine_c_v <- ave(mmpi$HE_ChildMortPop, mmpi$Country, FUN=function(x) median(x, na.rm = TRUE))
nine_r_v <- ave(mmpi$HE_ChildMortPop, mmpi$WorldRegion, FUN=function(x) median(x, na.rm = TRUE))
mmpi$HE_ChildMortPop <- ifelse(is.na(mmpi$HE_ChildMortPop), nine_c_v, mmpi$HE_ChildMortPop)
#getnulls(mmpi) = 24
mmpi$HE_ChildMortPop <- ifelse(is.na(mmpi$HE_ChildMortPop), nine_r_v, mmpi$HE_ChildMortPop)
#getnulls(mmpi) = 0

# "HE_NutrPop" column 10 --> 132 nulls
ten_c_v <- ave(mmpi$HE_NutrPop, mmpi$Country, FUN = function(x) median(x, na.rm = TRUE))
ten_c_r <- ave(mmpi$HE_NutrPop, mmpi$WorldRegion, FUN = function(x) median(x, na.rm = TRUE))
#getulls --> 132
mmpi$HE_NutrPop <- ifelse(is.na(mmpi$HE_NutrPop), ten_c_r, mmpi$HE_NutrPop)
#getulls --> 0

# "LS_ElectrPop" column 11 --> 18 nulls
eleven_c_v <- ave(mmpi$LS_ElectrPop, mmpi$Country, FUN = function(x) median(x, na.rm = TRUE))
eleven_r_v <- ave(mmpi$LS_ElectrPop, mmpi$WorldRegion, FUN = function(x) median(x, na.rm = TRUE))
mmpi$LS_ElectrPop <- ifelse(is.na(mmpi$LS_ElectrPop), eleven_c_v, mmpi$LS_ElectrPop)
#getnulls(mmpi) --> 18
mmpi$LS_ElectrPop <- ifelse(is.na(mmpi$LS_ElectrPop), eleven_r_v, mmpi$LS_ElectrPop)
#getnulls(mmpi) --> 0

# "LS_FloorPop" column 14 --> 44 nulls
fourtn_c_v <- ave(mmpi$LS_FloorPop, mmpi$Country, FUN = function(x) median(x, na.rm = TRUE))
fourtn_r_v <- ave(mmpi$LS_FloorPop, mmpi$WorldRegion, FUN = function(x) median(x, na.rm = TRUE))
mmpi$LS_FloorPop <- ifelse(is.na(mmpi$LS_FloorPop), fourtn_c_v, mmpi$LS_FloorPop)
#getnulls(mmpi) --> 44
mmpi$LS_FloorPop <- ifelse(is.na(mmpi$LS_FloorPop), fourtn_r_v, mmpi$LS_FloorPop)
#getnulls(mmpi) --> 0


# "LS_CookFuelPop" column 15 --> 25 nulls
fiftn_c_v <- ave(mmpi$LS_CookFuelPop, mmpi$Country, FUN = function(x) median(x, na.rm = TRUE))
fiftn_r_v <- ave(mmpi$LS_CookFuelPop, mmpi$WorldRegion, FUN = function(x) median(x, na.rm = TRUE))
mmpi$LS_CookFuelPop <- ifelse(is.na(mmpi$LS_CookFuelPop), fiftn_c_v, mmpi$LS_CookFuelPop)
#getnulls(mmpi) --> 25
mmpi$LS_CookFuelPop <- ifelse(is.na(mmpi$LS_CookFuelPop), fiftn_r_v, mmpi$LS_CookFuelPop)
#getnulls(mmpi) --> 0

#------------------ MPI Country Median & Mean for Indicators
mmpi_country <- mmpi %>% group_by(ISO,Country) %>% 
  summarize(ED_SchoolingPop_Md = median(ED_SchoolingPop), ED_SchoolAttPop_Md = median(ED_SchoolAttPop), 
            HE_ChildMortPop_Md= median(HE_ChildMortPop), HE_NutrPop_Md = median(HE_NutrPop), LS_ElectrPop_Md = median(LS_ElectrPop),
            LS_ImprSanitPop_Md = median(LS_ImprSanitPop), LS_DrinkWatPop_Md = median(LS_DrinkWatPop), LS_FloorPop_Md = median(LS_FloorPop),
            LS_CookFuelPop_Md = median(LS_CookFuelPop), LS_AssetOwnPop_Md = median(LS_AssetOwnPop), ED_SchoolingPop_Av = mean(ED_SchoolingPop),
            ED_SchoolAttPop_Av = mean(ED_SchoolAttPop), HE_ChildMortPop_Av= mean(HE_ChildMortPop), HE_NutrPop_Av = mean(HE_NutrPop),
            LS_ElectrPop_Av = mean(LS_ElectrPop),LS_ImprSanitPop_Av = mean(LS_ImprSanitPop), LS_DrinkWatPop_Av = mean(LS_DrinkWatPop),
            LS_FloorPop_Av = mean(LS_FloorPop),LS_CookFuelPop_Av = mean(LS_CookFuelPop), LS_AssetOwnPop_Av = mean(LS_AssetOwnPop))
mmpi_country_md <- mmpi_country[1:12]


write.csv(mmpi_country, file="mpi_medians_means_bycountry.csv")
write.csv(mmpi, file="mpi_cleansed_nonulls.csv")

###################### Integrating Datasets #############################

testing <- kiva_region %>% left_join(mpi[c(1,4:5)], by=("ISO"="ISO"), incomparables=NA)
intm <- testing %>% left_join(mpi[c(1:3,6:16)], by=c("ISO"="ISO", "country"="Country", "region"="SubNatRegion"), incomparables=NA)
be <- intm %>% distinct()
region_mpi_joined <- be
testing <- region_mpi_joined %>% left_join(country_stats, by=c("ISO"="country_code3"), incomparables = NA)
region_mpi_countrystats_joined <- testing
testing <- theme_region_red %>% left_join(region_mpi_countrystats_joined[ ,c(2,10,11,25:33)], by=c("ISO"="ISO"), incomparables = NA)
t <- testing %>% distinct()
region_theme_region_mpi_countrystats_joined <- t


#------------------ Theme Region <-- CountryMPI
theme_region_red <- theme_region[,-9]
theme_region_red <- theme_region_red[,-13]
intm <- mmpi[c(1,2,4,5)] %>% distinct()
lvlone <- theme_region_red %>% left_join(intm, by=c("ISO"="ISO"), incomparables = NA)
lvlone <- lvlone %>% distinct()
write.csv(lvlone, file="theme_by_region.csv")

lvltwo <- lvlone %>% left_join(t, by=c("mpi_region"="country_region"))
lvltwo <- lvlone %>% left_join(t[7:17], by=c("mpi_region"="country_region"))
b <- lvltwo %>% left_join(country_stats[6:13], by=c("country"="kiva_country_name"), incomparables = NA)
lvlthree <- lvltwo %>% left_join(country_stats[6:13], by=c("country"="kiva_country_name"), incomparables = NA)
write.csv(lvlthree, file="theme_region_mpi_ind_stats.csv")


#------------------ Loans <-- CountryMPI & Theme
countrycode <- country_stats[1:3] %>% distinct()
intrm <- countrycode %>% left_join(kiva_region[2:3], by=c("country_code3"="ISO"))
intrm <- intrm %>% distinct()
kiva_loans_mpi <- kiva_loans %>% left_join(intrm[1:3], by=c("country_code"="country_code"), incomparables = NA)
intm <- kiva_region[c(2,3,5,10)] %>% distinct()
kiva_loans_mpi <- kiva_loans_mpi %>% left_join(intm, by=c("country_code3"="ISO"), incomparables = NA)
kiva_loans_mpi_theme <- kiva_loans_mpi %>% left_join(theme, by=c("id"="id"), incomparables=NA)
k <- kiva_loans_mpi_theme %>% left_join(mmpi_country_md[c(1,3:12)], by=c("country_code3"="ISO"), incomparables = NA)
kiva_loans_mpi_theme_indicators <- k
kiva_loans_mpi_theme_indicators <- kiva_loans_mpi_theme %>% left_join(mmpi_country_md[c(1,3:12)], by=c("country_code3"="ISO"), incomparables = NA)
k <- kiva_loans_mpi_theme_indicators %>% left_join(country_stats[c(3,4,6:12)], by=c("country_code3"="country_code3"), incomparables = NA)
kiva_loans_mpi_theme_indicators_stats <- kiva_loans_mpi_theme_indicators %>% left_join(country_stats[c(3,4,6:12)], by=c("country_code3"="country_code3"), incomparables = NA)

write.csv(kiva_loans_mpi_theme_indicators_stats, file="kiva_loans_mpi_theme_indicators_stats.csv")









