library(torch)
library(luz)
library(torchvision)
library(torchdatasets)
library(purrr)

train_ind <- 1:1e4
valid_ind <- 10001:1.5e4
test_ind <- 15001:2e4

add_channel_dim <- function(img){
  img$unsqueeze(1)
}

crop_axes <- function(img){
  transform_crop(img, 
                 top = 0, 
                 left = 21, 
                 height = 131, 
                 width = 130)
}

root <- file.path(tempdir(), "correlation")

train_ds <- guess_the_correlation_dataset(
  root = root,
  transform = function(img){
    crop_axes(img) %>% add_channel_dim()
  },
  indexes = train_ind,
  download = TRUE
)

valid_ds <- guess_the_correlation_dataset(
  root = root,
  transform = function(img){
    crop_axes(img) %>% add_channel_dim()
  },
  indexes = valid_ind,
  download = FALSE
)

test_ds <- guess_the_correlation_dataset(
  root = root,
  transform = function(img){
    crop_axes(img) %>% add_channel_dim()
  },
  indexes = test_ind,
  download = FALSE
)

length(train_ds) == length(train_ind)
length(valid_ds) == length(valid_ind)
length(test_ds) == length(test_ind)

train_dl <- dataloader(train_ds, batch_size = 64, shuffle = TRUE)

length(train_dl)

batch <- dataloader_make_iter(train_dl) %>% dataloader_next()

dim(batch$x)
dim(batch$y)

par(mfrow = c(8,8), mar = rep(0,4))

images<- as.array(batch$x$squeeze(2))

images %>% 
  array_tree(1) %>% 
  map(as.raster) %>% 
  iwalk(~{plot(.x)})

valid_dl <- dataloader(valid_ds, batch_size = 64)

test_dl <- dataloader(test_ds, batch_size = 64)

torch_manual_seed(777)

net <- nn_module(
  
  "corr-cnn",
  
  initialize = function() {
    
    self$conv1 <- nn_conv2d(in_channels = 1, out_channels = 32, kernel_size = 3)
    self$conv2 <- nn_conv2d(in_channels = 32, out_channels = 64, kernel_size = 3)
    self$conv3 <- nn_conv2d(in_channels = 64, out_channels = 128, kernel_size = 3)
    
    self$fc1 <- nn_linear(in_features = 14 * 14 * 128, out_features = 128)
    self$fc2 <- nn_linear(in_features = 128, out_features = 1)
    
  },
  
  forward = function(x) {
    
    x %>% 
      self$conv1() %>%
      nnf_relu() %>%
      nnf_avg_pool2d(2) %>%
      
      self$conv2() %>%
      nnf_relu() %>%
      nnf_avg_pool2d(2) %>%
      
      self$conv3() %>%
      nnf_relu() %>%
      nnf_avg_pool2d(2) %>%
      
      torch_flatten(start_dim = 2) %>%
      self$fc1() %>%
      nnf_relu() %>%
      
      self$fc2()
  }
)

model <- net()

model(batch$x)

fitted <- net %>%
  setup(
    loss = function(y_hat, y_true){
      nnf_mse_loss(y_hat, y_true$unsqueeze(2))
    },
    optimizer = optim_adam
  ) %>%
  fit(train_dl, epochs = 10, valid_data = test_dl)

preds <- predict(fitted, test_dl) %>% as.numeric()

test_dl <- dataloader(test_ds, batch_size = 5e3)

targets <- ((test_dl %>% dataloader_make_iter() %>% dataloader_next())$y) %>% as.numeric()

df <- data.frame(targets = targets, preds = preds)

library(ggplot2)

ggplot(df, aes(x = targets, y = preds)) +
  geom_point(size = 0.1) + 
  theme_classic() + 
  labs(title = "Guess the correlation",
       subtitle = "torch for R exercise",
       x = "True correlations",
       y = "model predictions")











