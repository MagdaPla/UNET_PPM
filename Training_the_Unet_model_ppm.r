# This code runs only on Windows because of specific parallel backend. 
# set working directory
setwd("C:/Taller/UOC/Aules/TFM/Exemples/UNET_PPM")

# remove previous trained weights if any
unlink(list.files(path = "./weights_r/",full.names = TRUE))

# training parameters
epochs = 100
batch_size <- 24
DRAW_SAMPLES = TRUE 
no_cores = 12
lr_rate = 0.0001

library(keras)
library(tensorflow)
library(reticulate)
library(raster)
library(abind)
library(foreach)
library(parallel)
library(doParallel)


set.seed(104)


# for reproducibility
tf$set_random_seed(100)

# if it doesnt work change the 1 value
gpu_options <- tf$GPUOptions(allow_growth=TRUE, per_process_gpu_memory_fraction = 1) #tf$GPUOptions(per_process_gpu_memory_fraction = 0.3)
config <- tf$ConfigProto(gpu_options = gpu_options)


session_conf <- config
sess <- tf$Session(graph = tf$get_default_graph(), config = session_conf)

# Parameters -----------------------------------------------------

# directory of the image and object masks
images_dir <- "./rgb/" 
masks_dir <- "./masc/"

# 
if (DRAW_SAMPLES) {
  
  unlink(list.files(path = "./valid_masc/",full.names = TRUE))
  unlink(list.files(path = "./valid_rgb/",full.names = TRUE))
  
  # number of image for training
  train_samples <- length(list.files(images_dir)) # 669
  train_index <- sample(1:train_samples, round(train_samples * 0.8)) # 80%
  val_index <- c(1:train_samples)[-train_index]
  
  
  # sauver les images de validation dans
  valid_save=list.files(images_dir,full.names = TRUE)
  valid_save=valid_save[val_index]
  file.copy(from=valid_save,to="./valid_rgb/")
  
  valid_save=list.files(masks_dir,full.names = TRUE)
  valid_save=valid_save[val_index]
  file.copy(from=valid_save,to="./valid_masc/")
  
  save(train_index, val_index, file = "./train_val_indices.RData")
  
} else {
  load("./train_val_indices.RData", verbose=T)
}



# Loss function -----------------------------------------------------

K <- backend()
K$set_session(sess)


dice_coef <- custom_metric("custom", function(y_true, y_pred, smooth = 1.0) {
  y_true_f <- k_flatten(y_true)
  y_pred_f <- k_flatten(y_pred)
  intersection <- k_sum(y_true_f * y_pred_f)
  result <- (2 * intersection + smooth) / 
    (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
  return(result)
})

bce_dice_loss <- function(y_true, y_pred) {
  result <- loss_binary_crossentropy(y_true, y_pred) +
    (1 - dice_coef(y_true, y_pred))
  return(result)
}


# U-net 128 -----------------------------------------------------

get_unet_128 <- function(input_shape = c(128, 128, 3),
                         num_classes = 1) {
  
  inputs <- layer_input(shape = input_shape)
  # 128
  
  down1 <- inputs %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") 
  down1_pool <- down1 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  # 64
  
  down2 <- down1_pool %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") 
  down2_pool <- down2 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  # 32
  
  down3 <- down2_pool %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") 
  down3_pool <- down3 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  # 16
  
  down4 <- down3_pool %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") 
  down4_pool <- down4 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  # 8
  
  center <- down4_pool %>%
    layer_conv_2d(filters = 1024, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 1024, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") 
  # center
  
  up4 <- center %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down4, .), axis = 3)} %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  # 16
  
  up3 <- up4 %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down3, .), axis = 3)} %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  # 32
  
  up2 <- up3 %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down2, .), axis = 3)} %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  # 64
  
  up1 <- up2 %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down1, .), axis = 3)} %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  # 128
  
  classify <- layer_conv_2d(up1,
                            filters = num_classes, 
                            kernel_size = c(1, 1),
                            activation = "sigmoid")
  
  
  model <- keras_model(
    inputs = inputs,
    outputs = classify
  )
  
  model %>% compile(
    optimizer = optimizer_rmsprop(lr = 0.0001),
    loss = bce_dice_loss,
    metrics = c(dice_coef)
  )
  
  return(model)
}

model <- get_unet_128()

# to use previous trained weights use the function load_model_weights_hdf5
# for example :
# load_model_weights_hdf5(model, "./weights_r_save/unet64_178.h5")


## define number of clusters
cl <- makePSOCKcluster(no_cores) 

clusterEvalQ(cl, {
  
  library(abind)     
  library(raster)
  library(reticulate)
  
  # Read and augmentation functions -----------------------------------------------------
  
  imagesRead <- function(image_file,mask_file)
  {
    
    img <- brick(image_file)
    mask <- raster(mask_file)
    
    return(list(img = img, mask = mask))
  }
  
  # randomHorizontalFlip rotations and inversion + rotations
  randomHorizontalFlip <- function(img,mask,u = 0) {
    if (rnorm(1) < u) return(list(img = img, mask = mask))
    r_angle=sample(c(2,3,4,5,6,7,8),1)
    if(r_angle==2) {return(list(img = flip(t(img),direction = 1), mask = flip(t(mask),direction = 1)))}
    if(r_angle==3) {return(list(img = flip(t(flip(t(img),direction = 1)),direction = 1), mask = flip(t(flip(t(mask),direction = 1)),direction = 1)))}
    if(r_angle==4) {return(list(img = flip(t(img),direction = 2), mask = flip(t(mask),direction = 2)))}
    if(r_angle==5) {return(list(img = flip(img,direction = 1), mask = flip(mask,direction = 1)))}
    if(r_angle==6) {return(list(img = flip(t(flip(img,direction = 1)),direction = 1), mask = flip(t(flip(mask,direction = 1)),direction = 1)))}
    if(r_angle==7) {return(list(img = flip(t(flip(t(flip(img,direction = 1)),direction = 1)),direction = 1), mask = flip(t(flip(t(flip(mask,direction = 1)),direction = 1)),direction = 1)))}
    if(r_angle==8) {return(list(img = flip(t(flip(img,direction = 1)),direction = 2), mask = flip(t(flip(mask,direction = 1)),direction = 2)))}
  }
  
  # add a shift to the bands
  randomVariability = function(img, u = 0, variability = c(90, 110)) {
    if (rnorm(1) < u) return(img)
    variability_shift = runif(1, variability[1], variability[2])/100
    img = img * variability_shift
    return(img)
  }
  
  
  img2arr <- function(image) {
    image <- as.array(image)
    result <- aperm(image, c(2,1,3))
    result <- result/255 # to have values between 0 and 1
    array_reshape(result,  c(1, dim(image)[1], dim(image)[2], dim(image)[3]))
  }
  
  
  mask2arr <- function(mask) {
    mask=as.array(mask[[1]])
    result <- aperm(mask, c(2,1,3))
    result=result[,,1]
    array_reshape(result,  c(1, dim(mask)[1], dim(mask)[2], dim(mask)[3]))
  }
  
})


registerDoParallel(cl)


train_generator <- function(images_dir, 
                            samples_index,
                            masks_dir, 
                            batch_size) {
  images_iter <- list.files(images_dir, 
                            pattern = ".tif", 
                            full.names = TRUE)[samples_index] # for current epoch
  images_all <- list.files(images_dir, 
                           pattern = ".tif",
                           full.names = TRUE)[samples_index]  # for next epoch
  masks_iter <- list.files(masks_dir, 
                           pattern = ".tif",
                           full.names = TRUE)[samples_index] # for current epoch
  masks_all <- list.files(masks_dir, 
                          pattern = ".tif",
                          full.names = TRUE)[samples_index] # for next epoch
  
  function() {
    
    # start new epoch
    if (length(images_iter) < batch_size) {
      images_iter <<- images_all
      masks_iter <<- masks_all
    }
    
    batch_ind <- sample(1:length(images_iter), batch_size)
    
    batch_images_list <- images_iter[batch_ind]
    images_iter <<- images_iter[-batch_ind]
    batch_masks_list <- masks_iter[batch_ind]
    masks_iter <<- masks_iter[-batch_ind]
    
    
    x_y_batch <- foreach(i = 1:batch_size) %dopar% {
      x_y_imgs <- imagesRead(image_file = batch_images_list[i],
                             mask_file = batch_masks_list[i])
      
      # flip all side and invert
      x_y_imgs <- randomHorizontalFlip(x_y_imgs$img,x_y_imgs$mask)
      
      # add some variability to the values
      x_y_imgs$img = randomVariability(x_y_imgs$img, u = 0, variability = c(90, 110))
      
      # return as arrays
      x_y_arr <- list(x = img2arr(x_y_imgs$img),
                      y = mask2arr(x_y_imgs$mask))
    }
    
    x_y_batch <- purrr::transpose(x_y_batch)
    
    x_batch <- do.call(abind, c(x_y_batch$x, list(along = 1)))
    
    y_batch <- do.call(abind, c(x_y_batch$y, list(along = 1)))
    
    result <- list(keras_array(x_batch), keras_array(y_batch))
    return(result)
  }
}

val_generator <- function(images_dir, 
                          samples_index,
                          masks_dir, 
                          batch_size) {
  images_iter <- list.files(images_dir, 
                            pattern = ".tif", 
                            full.names = TRUE)[samples_index] # for current epoch
  images_all <- list.files(images_dir, 
                           pattern = ".tif",
                           full.names = TRUE)[samples_index]  # for next epoch
  masks_iter <- list.files(masks_dir, 
                           pattern = ".tif",
                           full.names = TRUE)[samples_index] # for current epoch
  masks_all <- list.files(masks_dir, 
                          pattern = ".tif",
                          full.names = TRUE)[samples_index] # for next epoch
  
  function() {
    
    # start new epoch
    if (length(images_iter) < batch_size) {
      images_iter <<- images_all
      masks_iter <<- masks_all
    }
    
    batch_ind <- sample(1:length(images_iter), batch_size)
    
    batch_images_list <- images_iter[batch_ind]
    images_iter <<- images_iter[-batch_ind]
    batch_masks_list <- masks_iter[batch_ind]
    masks_iter <<- masks_iter[-batch_ind]
    
    
    x_y_batch <- foreach(i = 1:batch_size) %dopar% {
      x_y_imgs <- imagesRead(image_file = batch_images_list[i],
                             mask_file = batch_masks_list[i])
      # without augmentation
      ########################################
      ########################################
      # return as arrays
      x_y_arr <- list(x = img2arr(x_y_imgs$img),
                      y = mask2arr(x_y_imgs$mask))
    }
    
    x_y_batch <- purrr::transpose(x_y_batch)
    
    x_batch <- do.call(abind, c(x_y_batch$x, list(along = 1)))
    
    y_batch <- do.call(abind, c(x_y_batch$y, list(along = 1)))
    
    result <- list(keras_array(x_batch), keras_array(y_batch))
    return(result)
  }
}

train_iterator <- py_iterator(train_generator(images_dir = images_dir,
                                              masks_dir = masks_dir,
                                              samples_index = train_index,
                                              batch_size = batch_size))

val_iterator <- py_iterator(val_generator(images_dir = images_dir,
                                          masks_dir = masks_dir,
                                          samples_index = val_index,
                                          batch_size = batch_size))



# Training -----------------------------------------------------


# callbacks
callbacks_list <- list(
  callback_model_checkpoint(filepath = "weights_r/unet64_{epoch:03d}.h5",
                            monitor = "val_custom",
                            save_best_only = FALSE,
                            save_weights_only = TRUE,
                            mode = "max" ,period =1,save_freq = NULL,)
)


model %>% fit_generator(
  generator=train_iterator,
  steps_per_epoch = as.integer(length(train_index) / batch_size),
  epochs = epochs,
  validation_data = val_iterator,
  validation_steps = as.integer(length(val_index) / batch_size),
  verbose = 1,  callbacks = callbacks_list
)
