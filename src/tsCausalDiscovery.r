library(seqICP)
library(igraph)
# set.seed(5)

source("/home/haikim20/angelicathesis/src/seqICP.r")

# Function to generate a random DAG and permute the node labels
generate_random_dag <- function(num_nodes, edge_prob) {
  # Create the initial adjacency matrix
  mat <- matrix(0, nrow = num_nodes, ncol = num_nodes)
  # guarantees acyclicity
  for (i in 1:(num_nodes-1)) {
    for (j in (i+1):num_nodes) {
      if (runif(1) < edge_prob) {
        mat[i, j] <- 1 #i = parent, j = child
      }
    }
  }
  
  # Generate a random permutation of node labels
  perm <- sample(num_nodes)
  
  # Permute both the rows and the columns of the matrix
  mat <- mat[perm, perm]
  
  return(mat == 1)
}

generate_coefficients <- function(dag, Yidx, coeff_min=NULL, coeff_max=NULL) {
  coefficient_matrix <- matrix(0, nrow = nrow(dag), ncol = ncol(dag))
  
  # Apply runif to each element individually
  for (i in 1:nrow(dag)) {
    for (j in 1:ncol(dag)) {
      if (dag[i, j] || i==j) {
        repeat {
          if(!is.null(coeff_min) & !is.null(coeff_max)){
            random_value <- runif(1,coeff_min, coeff_max)  
          }
          else{
            # mixed coeff strength for target variable
            if (j == Yidx) {
              strength <- sample(c("weak", "moderate", "strong"), 1)
              if (strength == "weak") {
                random_value <- runif(1, 0.01, 0.1)
              } else if (strength == "moderate") {
                random_value <- runif(1, 0.1, 0.3)
              } else { # strong
                random_value <- runif(1, 0.3, 0.5)
              }
            }  
            else {
              random_value <- runif(1, 0, 0.5)
            }
          }
          if (random_value != 0) {
            sign <- sample(c(-1, 1), size = 1, replace = TRUE)
            coefficient_matrix[i, j] <- random_value*sign
            break
          }
        }
      }
    }
  }
  return(coefficient_matrix)
}

# generate simulated time series
simulate_ar_time_series <- function(dag, Yidx, max_lag, num_samples, 
                                    data_mean, data_var, noise_var,
                                    coeff_min=NULL, coeff_max=NULL, 
                                    intervention_strength=30) {
  num_nodes <- nrow(dag)
  data <- matrix(data=0, nrow = num_samples, ncol = num_nodes)
  
  # Randomly initialize values for t=1
  data[1,] <- matrix(rnorm(num_nodes, data_mean, data_var),
                     nrow = 1, ncol = num_nodes) + rnorm(num_nodes, 0, sqrt(noise_var))
  coeffs <- lapply(1:(max_lag+1), function(x) generate_coefficients(dag, Yidx, coeff_min, coeff_max))
  diag(coeffs[[1]]) <- 0 # no instantaneous effects for itself             
  
  intervention_vars <- sample(setdiff(1:num_nodes, Yidx), num_nodes-1, replace=FALSE)
  intervention_timesteps <- sample(2:num_samples, num_nodes-1, replace = FALSE)
  
  # sanity checks for interventions
  if(Yidx %in% intervention_vars){
    stop("interventions on a target variable")
  }
  
  # extract topological ordering
  graph <- graph_from_adjacency_matrix(dag, mode="directed", diag=FALSE)
  order <- topo_sort(graph)
  # find root nodes with no parents (i.e. columns that are all false)
  root <- which(apply(dag, 2, function(x) all(!x)), arr.ind = TRUE)
  for (t in 2:num_samples) {
    for (i in order) {
      val <- 0
      #perform interventions
      if(t %in% intervention_timesteps && i == intervention_vars[which(intervention_timesteps == t)]){
        data[t, i] <- intervention_strength
      }
      else if(i %in% root){
        for(k in 1:min(max_lag, t-1)){
          val <- val + data[t-k, i] * coeffs[[k+1]][i, i] #autoregressive values only cuz it's root
        } 
        data[t,i] <- val + rnorm(1,0,sqrt(noise_var))
      }
      else{
        for(k in 0:min(max_lag, t-1)){
          val <- val + data[t-k, ] %*% coeffs[[k+1]][, i] #non-parents will be ignored by the zero coef
        } 
        data[t,i] <- val + rnorm(1,0,sqrt(noise_var))
      }
    }
  }
  
  # sanity checks on data
  check_intervention <- all(sapply(data[cbind(intervention_timesteps, intervention_vars)],
                                   function(x) x == intervention_strength))
  if(!check_intervention){
    stop("wrong interventions")
  }
  if(any(is.na(data)) || any(is.infinite(as.matrix(data)))){
    stop("infinity or NAs in data")
  }
  
  results <- list(data = data, coeffs=coeffs,
                  intervention_vars = intervention_vars,
                  intervention_timesteps=intervention_timesteps)
  class(results) <- "sim"
  return(results)
}

run_causal_experiment <- function(iter=1, dag, num_nodes, num_samples, data_mean, data_var, 
                                  noise_var, intervention_strength, 
                                  coeff_min=NULL, coeff_max=NULL, 
                                  data_mean_range=NULL, data_var_range=NULL, 
                                  noise_var_range=NULL, edge_prob = 0.5, 
                                  max_lag = 1, alpha = 0.05, B = 1000) {
  Yidx <- which.max(colSums(dag))
  # Yidx <- sample(1:num_nodes, 1)
  
  # Generate time series data
  if(!is.null(noise_var_range)){
    noise_var <- runif(1, noise_var_range[1], noise_var_range[2])
  }
  if(!is.null(data_mean_range)){
    data_mean <- runif(1, data_mean_range[1], data_mean_range[2])
  }
  if(!is.null(data_var_range)){
    data_var <- runif(1, data_var_range[1], data_var_range[2])
  }
  
  sim.res <- simulate_ar_time_series(dag=dag, Yidx=Yidx, 
                                     max_lag=max_lag, num_samples=num_samples, 
                                     data_mean=data_mean, data_var = data_var, noise_var = noise_var, 
                                     coeff_min=coeff_min, coeff_max=coeff_max, 
                                     intervention_strength = intervention_strength)
  data <- data.frame(sim.res$data)
  coeffs <- sim.res$coeffs
  intervention_vars <- sim.res$intervention_vars
  intervention_timesteps <- sim.res$intervention_timesteps
  
  Xmatrix <- as.matrix(data[,-Yidx])
  Y <- as.matrix(data[,Yidx])
  
  # Run seqICP
  # print(paste("lag:", max_lag))
  # print(coeffs)
  n_new <- num_samples-max_lag
  seqICP.result <- seqICP(X = Xmatrix, Y, model="ar", test = "block.decoupled",
                          par.test = list(grid = seq(1, n_new,
                                                     length.out=floor(log(n_new))),
                                          complements = TRUE, link = sum,
                                          alpha = alpha, B = B),
                          par.model= list(pknown = TRUE, p=max_lag),
                          stopIfEmpty=TRUE, silent=FALSE)
  # summary(seqICP.result)
  
  #Extract CIs
  p <- ncol(Xmatrix)
  parent_set <- seqICP.result$parent.set #instantaneous parents
  estimated_parents <- rep(FALSE, p)
  estimated_parents[parent_set] <- TRUE
  estimated_parents <- c(estimated_parents, 
                         !(seqICP.result$coefficients[-(1:(p+1)),2] < 0 & 
                             seqICP.result$coefficients[-(1:(p+1)),3] >0)) #append lagged terms as parents
  true_parents <- c(dag[-Yidx,Yidx], 
                    c(sapply(coeffs[-1], function(x)c(x[Yidx,Yidx], x[-Yidx,Yidx]))) != 0)
  
  # extract coefficients
  estimated_coeffs <- seqICP.result$coefficients[-1,1]
  true_coeffs <- c(coeffs[[1]][-Yidx,Yidx], 
                   c(sapply(coeffs[-1], function(x)c(x[Yidx,Yidx], x[-Yidx,Yidx]))))
  lower_bound =  seqICP.result$coefficients[-1,2] 
  upper_bound = seqICP.result$coefficients[-1,3]
  
  # sanity checks
  if(length(estimated_parents)!=length(estimated_coeffs) && length(true_parents)!=length(true_coeffs)){
    stop("dimensions of parent and coeff sets don't match")
  }
  
  #adjust the output for (incorrect) # lags detected by the model
  # est_max_lag <- unique(seqICP.result$test.results$p)
  # if(length(est_max_lag) > 1){
  #     print(seqICP.result$test.results$p)
  #   stop("multiple lags found?")
  # }
  diff <- length(estimated_parents) - length(true_parents)                          
  if(diff>0){
    true_parents <- c(true_parents, rep(FALSE, abs(diff)))
    true_coeffs <- c(true_coeffs, rep(0, abs(diff)))
  }
  if(diff<0){
    estimated_parents <- c(estimated_parents, rep(FALSE, abs(diff)))
    estimated_coeffs <- c(estimated_coeffs, rep(0, abs(diff)))
    lower_bound <- c(lower_bound, rep(0,abs(diff)))
    upper_bound <- c(upper_bound, rep(0,abs(diff)))
  }
  
  TP <- sum(true_parents & estimated_parents)
  FP <- sum(!true_parents & estimated_parents)
  TN <- sum(!true_parents & !estimated_parents)
  FN <- sum(true_parents & !estimated_parents)
  
  accuracy <- (TP + TN) / (TP + FP + FN + TN)
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  specificity <- TN / (TN + FP)                      
  F1 <- 2 * (precision * recall) / (precision + recall)
  num_parents <- sum(true_parents)
  
  # Find ancestors of the target node
  graph <- graph_from_adjacency_matrix(dag, mode = "directed")
  all_nodes <- V(graph)
  ancestors <- subcomponent(graph, Yidx, mode = "in")
  parents <- which(dag[,Yidx]==1)
  
  metrics <- data.frame(num_nodes = num_nodes, num_samples=num_samples, 
                        true_max_lag = max_lag, 
                        data_mean=data_mean, data_var=data_var, noise_var=noise_var, intervention_strength = intervention_strength,
                        model_rejected = seqICP.result$modelReject,
                        hidden_ancestors = length(setdiff(ancestors, c(Yidx, parents)))>0,
                        num_parents = num_parents, TP = TP,TN = TN, FP = FP, FN = FN,
                        accuracy=accuracy, specificity = specificity, precision=precision, recall=recall, F1=F1)
  
  labels <- character(length = length(true_parents))
  labels[true_parents & estimated_parents] <- "True Positive"
  labels[!true_parents & estimated_parents] <- "False Positive"
  labels[!true_parents & !estimated_parents] <- "True Negative"
  labels[true_parents & !estimated_parents] <- "False Negative"
  
  coeff_label <- data.frame(num_nodes = num_nodes, num_samples=num_samples, 
                            true_max_lag = max_lag, 
                            num_parents = num_parents, model_rejected = seqICP.result$modelReject,
                            hidden_ancestors = length(setdiff(ancestors, c(Yidx, parents)))>0,
                            data_mean=data_mean, data_var=data_var, 
                            noise_var = noise_var, intervention_strength = intervention_strength,
                            label = labels, true_coeffs = true_coeffs, estimated_coeffs = estimated_coeffs,
                            error = true_coeffs-estimated_coeffs,
                            lower_bound =  lower_bound, upper_bound = upper_bound,
                            stringsAsFactors = FALSE)
  
  results <- list(metrics, coeff_label)
  class(results) <- "exp"
  return(results)
}

args <- commandArgs(trailingOnly = TRUE)

if (length(args) == 0) {
  stop("No arguments provided")
} else {
  # Convert the first argument to integer
  input_arg <- as.integer(args[1])
}
# Assuming the first argument is what we want and it should be an integer
if (length(args) == 0) {
  stop("No arguments provided")
} else {
  # Convert the first argument to integer
  noise_var <- as.numeric(args[1])
  seed <- as.numeric(args[2])  
}                      

if(!exists(".Random.seed")) {
  # Initialize .Random.seed by generating a random number
  runif(1)
}                

# generate a random DAG                            
p <- 0.8                                
original_rng_state <- .Random.seed
set.seed(seed)             
d <- generate_random_dag(5, p)                            
.Random.seed <- original_rng_state     

param <- list(dag=d, num_nodes = 5, num_samples=1000, coeff_min=0.3, coeff_max=0.5, 
       data_mean = 2, data_var = 1, noise_var = noise_var, intervention_strength=30, max_lag=2)

metrics <- data.frame()
coeffs_results <- data.frame()
                            
num_sim <- 30       
                            
for (i in 1:num_sim) {
    exp.res <- do.call(run_causal_experiment, param)
    metrics <- rbind(metrics, cbind(sim_idx=i, seed = seed, edge_prob = p, exp.res[[1]])) 
    coeffs_results <- rbind(coeffs_results, cbind(sim_idx=c(i), seed = seed, edge_prob = p, exp.res[[2]]))
}

write.csv(metrics, paste0("node5_noise", param$noise_var,"_dense_g",seed, "_metrics.csv"), row.names = FALSE)
write.csv(coeffs_results, paste0("node5_noise", param$noise_var, "_dense_g",seed, "_coeffs.csv"), row.names = FALSE)
print(d)