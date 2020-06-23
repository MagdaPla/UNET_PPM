# CÀLCUL DE LES CAPÇADES DELS ARBRES A PARTIR D'UN MDE DEL PROCEESAMENT D'IMATGES DRON 

library(raster)
library(itcSegment)
library(rgdal)

# definim directori de treball
setwd("./img/MDE/retall")

# carreguem o visualitzem el MDE en format ASC d'arcgis amb capçalera
chm <- raster("nom_mde.asc")
plot(chm)

# generem les corones dels arbres
se<-itcIMG(chm,epsg=32663,searchWinSize = 7, TRESHSeed = 0.45,
           TRESHCrown = 0.55, DIST = 10, th = 0, ischm = FALSE)

summary(se)
plot(se,axes=T)

arbres<-se[3]

# exportem la capa generada a shp
writeOGR(arbres,dsn="C:/temp",layer="nom_capa", driver="ESRI Shapefile")
