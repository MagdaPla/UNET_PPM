# UNET_PPM
Repositori de formació per la preparació d'un Treball de Final de Màster en Bioinformàtica i Bioestadística de la UOC.

La informació d'aquest repositori correspon al codi per a la segmentació per a detectar zones danyades per la processionària del pi en imatges RGB captades des de drons amb Deep Learning, utilitzant el segmentador U-net (Keras i Tensorflow) amb R.

El contingut concret és:
1-UNET_PPM.ipynb correspon al codi per a l'entrenament i predicicó del model. Preparat Google Colab (R runtime). 
2-tree_crown.R: codi en R per a poder generar les capçades dels arbres a partir d'un MDE 
3-Conversion_formats.R: codi per aconvertir les imatges (RGB i màscares) a TIF de 24 bits i GIF respectivament per poder-los entrar al model. 
4-retalla_exporta.bat: codi per a retallar els tiles a entrar el model tant de les RGB com de les màscares a partir de la malla de 128x128 pixels que s'ha generat en format vectorial. Tot aquest procés s'executa amb el programa de SIG i Teledetecció MiraMon.
5- Els directoris amb les dades d'entrenament "rgb" i "masc" i unes exemples de noves imatges rgb per a realitzar prediccions a "rgb_to_prredict"

El codi pot evolucionar una vegada lliurat i presentat el TFM el mes de juliol del 2020.

Magda Pla
magda.pla@ctfc.cat
