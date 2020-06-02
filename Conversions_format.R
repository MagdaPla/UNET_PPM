# per probar el codi senzill de carvana canviem els formats originals a jpg i gif

# llegim els tif

library(magick)
library(tiff)

setwd("C:/Taller/UOC/Aules/TFM/Exemples/UNET_PPM")

#convertim mascares 

for (tile in 1:144){
  mascara <- image_read(paste("C:/Taller/UOC/Aules/TFM/Exemples/UNET_PPM/masc/",tile,".tif", sep = ""))
  image_write(mascara, path=(paste("C:/Taller/UOC/Aules/TFM/Exemples/UNET_PPM/masc_gif/",tile,".gif",sep = "")), format = "gif")
}


#convertim rgb 

for (tile in 1:144){
  rgb <- image_read(paste("C:/Taller/UOC/Aules/TFM/Exemples/UNET_PPM/rgb/",tile,".tif", sep = ""))
  image_write(rgb, path=(paste("C:/Taller/UOC/Aules/TFM/Exemples/UNET_PPM/rgb_jpg/",tile,".jpg",sep = "")), format = "jpg")
}
