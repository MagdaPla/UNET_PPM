# per probar el codi senzill de carvana canviem els formats originals que hem generat en TIFF a jpg i gif


library(magick)
library(tiff)

#convertim mascares 

for (tile in 1:144){
  mascara <- image_read(paste("C:/Taller/UOC/Aules/TFM/Naves/rgb/masc/",tile,".tif", sep = ""))
  image_write(mascara, path=(paste("C:/Taller/UOC/Aules/TFM/Naves/rgb/masc0_gif/",tile,".gif",sep = "")), format = "gif")
}


#convertim rgb 

for (tile in 1:144){
  rgb <- image_read(paste("C:/Taller/UOC/Aules/TFM/Naves/rgb/rgb/",tile,".tif", sep = ""))
  image_write(rgb, path=(paste("C:/Taller/UOC/Aules/TFM/Naves/rgb/rgb0_jpg/",tile,".jpg",sep = "")), format = "jpg")
}
