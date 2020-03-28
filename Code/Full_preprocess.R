##Preprocess
library(dummies)

pitches.dataset <- read.csv( file = "C:/Users/ugur_/Desktop/MSCBD/Thesis/downloads/pitches.csv")
str(pitches.dataset)
#remove unnecesseary pitch types like->intentional ball aloowing walk to batter
#AB/FA/UN/IN
#removing unknown data//wong type pitch // empty rows
clean.dataset <- (na.omit(pitches.dataset))
clean.dataset <- clean.dataset[clean.dataset$pitch_type != "UN" ,]
clean.dataset <- clean.dataset[clean.dataset$pitch_type != "FA" ,]
clean.dataset <- clean.dataset[clean.dataset$pitch_type != "AB" ,]
clean.dataset <- clean.dataset[clean.dataset$pitch_type != "IN" ,]
#change FO to PO (pitchout)
clean.dataset[clean.dataset$pitch_type == "FO", "pitch_type"]<-"PO"
#changed FT to SI(sinker)
clean.dataset[clean.dataset$pitch_type == "FT", "pitch_type"]<-"SI"

games.dataset <- read.csv( file = "C:/Users/ugur_/Desktop/MSCBD/Thesis/downloads/games.csv")
atbat.dataset <- read.csv( file = "C:/Users/ugur_/Desktop/MSCBD/Thesis/downloads/atbats.csv")
player.dataset<- read.csv( file = "C:/Users/ugur_/Desktop/MSCBD/Thesis/downloads/player_names.csv")
###

total<- merge(clean.dataset, atbat.dataset, by="ab_id")
total<- merge(total,games.dataset, by="g_id")
total<-merge(total , player.dataset, by.x = "batter_id",by.y = "id")
colnames(total)[colnames(total) == "first_name"]<-"batter_Fname"
colnames(total)[colnames(total) == "last_name"]<-"batter_Lname"
total<-merge(total , player.dataset, by.x = "pitcher_id",by.y = "id")
colnames(total)[colnames(total) == "first_name"]<-"pitcher_Fname"
colnames(total)[colnames(total) == "last_name"]<-"pitcher_Lname"

total<-total[,c("g_id","pitcher_id","inning","pitch_type","p_score","b_score","outs","pitch_num","on_1b","on_2b","on_3b","spin_rate","spin_dir","start_speed","end_speed","type")]
total<-cbind(total, dummy(total$pitch_type, sep = "_"))
df <- cbind(total)
df <- df[-c(1:nrow(total)),]
ids<-unique(total$g_id)
for(i in 1:length(ids)){
  tst<-total[total$g_id == ids[i],]
  inner_ids<-unique(tst$pitcher_id)
  for(j in 1:length(inner_ids)){
    df1<-tst[tst$pitcher_id == inner_ids[j],]
    df1$Game_pt_pCount<- nrow(df1)
    df1$Game_pt_t_s<- nrow(df1[df1$type == "S",])
    
    df1$total_CH<- nrow(df1[df1$pitch_type == "CH",])
    df1$total_CU<-nrow(df1[df1$pitch_type == "CU",])
    df1$total_EP<-nrow(df1[df1$pitch_type == "EP",])
    df1$total_FC<-nrow(df1[df1$pitch_type == "FC",])
    df1$total_FF<-nrow(df1[df1$pitch_type == "FF",])
    df1$total_PO<-nrow(df1[df1$pitch_type == "PO",])
    df1$total_FS<-nrow(df1[df1$pitch_type == "FS",])
    df1$total_KC<-nrow(df1[df1$pitch_type == "KC",])
    df1$total_KN<-nrow(df1[df1$pitch_type == "KN",])
    df1$total_SC<-nrow(df1[df1$pitch_type == "SC",])
    df1$total_SI<-nrow(df1[df1$pitch_type == "SI",])
    df1$total_SL<-nrow(df1[df1$pitch_type == "SL",])
    df<-rbind(df,df1)
  }
  
}

df<-cbind(df, dummy(df$pitch_type, sep = "_"))
df<-df[,-which(names(df) %in% c("g_id","pitcher_id","pitch_type"))]

minisamp<-df[0:50000,]
write.csv(df,'C:\\Users\\ugur_\\Desktop\\MSCBD\\Thesis\\layer2\\final.csv', row.names = FALSE)
