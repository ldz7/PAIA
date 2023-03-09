rm(list=ls())

library(grpreg)


# Function for Block diagonal matrix generation (with adjustment of the order of columns)
# xlist: list of covariate matrix
# n: vector of sample size (n^{(1)}, n^{(2)}, ..., n^{(m)})
# p: number of covariates
# m: number of datasets
bdiag <- function(xlist, n, p, m=1) {
    x <- as.matrix(do.call('rbind', xlist))
    xx <- c()
    if (m > 1) {
        large_matrix_col_idx <- m * seq(0, p - 1)   # c(0, m, m*2, m*3, ..., m*(p-1))
        original_matrix_row_index <- c(0, cumsum(n))    # c(0, n[1], n[1]+n[2], n[1]+n[2]+n[3], ...)
        for (i in 1:m) {
            xx0 <- matrix(0, n[i], m * p)
            xx0[, i + large_matrix_col_idx] <- x[(original_matrix_row_index[i]+1):original_matrix_row_index[i+1], ]
            xx <- rbind(xx, xx0)
        }
    } else xx <- x
    return(xx)
}




dataset1 <- read.csv('dataset1.csv')
dataset2 <- read.csv('dataset2.csv')
dataset3 <- read.csv('dataset3.csv')
nn <- c(nrow(dataset1), nrow(dataset2), nrow(dataset3))
m <- length(nn)
p <- ncol(dataset1) - 1 # minus 1 if response is in the matrix/data.frame


data.reorg <- list()
data.reorg$XX <- bdiag(list(dataset1[, 1:p], dataset2[, 1:p], dataset3[, 1:p]), 
                       nn, p, m)
data.reorg$X <- do.call('rbind', list(dataset1[, 1:p], dataset2[, 1:p], dataset3[, 1:p]))
data.reorg$Y <- c(dataset1[, p + 1], dataset2[, p + 1], dataset3[, p + 1])


# given according to the result of Step 1 (prior information extraction)
# e.g. 1:5 means the first five covariates are in the prior set
prior_set_index <- 1:5 

m1_group <- factor(rep(1:p, rep(m, p)))
m1_prior <- 1:p
m1_prior[prior_set_index] <- 0
m1_prior <- rep(m1_prior, each=m)



# foldid generation for cross validation
nfolds <- 5
foldid1 <- sample(rep(seq(nfolds), length=nn[1]))
foldid2 <- sample(rep(seq(nfolds), length=nn[2]))
foldid3 <- sample(rep(seq(nfolds), length=nn[3]))
foldid <- c(foldid1, foldid2, foldid3)

eta_grid <- seq(from=0, to=10, by=0.01)


##############################################################
####### M1 ：Prior Assisted Integrative Analysis #############
##############################################################

data.train <- data.frame(data.reorg[["XX"]], data.reorg[["Y"]])

# Equation (3)
prior <- cv.grpreg(as.matrix(data.train[, 1:(m*p)]), 
                   as.matrix(data.train[, m*p + 1]),
                   family="gaussian",
                   group=m1_prior,
                   penalty="grLasso", 
                   fold=foldid)

# \widehat{\dot{\mathbb{Y}}} construction
y_hat_p <- predict(prior,
                   as.matrix(data.train[, 1:(m*p)]),
                   type="response",
                   lambda=prior$lambda.min)

m1_cve_array <- array(NA, length(eta_grid)) # mean cross-validated error
m1_beta_array <- array(NA, c(length(eta_grid), m*p + 1))

for (i in 1:length(eta_grid)) {
    # \widetilde{\mathbb{Y}} construction
    y_tilde <- (as.matrix(data.train[, m*p + 1]) + eta_grid[i]*y_hat_p) / (1+eta_grid[i])
    # Equation (4)
    m1_model <- cv.grpreg(as.matrix(data.train[, 1:(m*p)]), 
                          y_tilde,
                          group=m1_group,
                          penalty="grLasso", 
                          fold=foldid)
    
    m1_beta_array[i,] <- predict(m1_model, lambda=m1_model$lambda.min, 
                                 type='coefficients')
    m1_cve_array[i] <- min(m1_model$cve)
}

# the best estimated coefficients, in the form of
# c(\beta_1^{(1)}, \beta_1^{(2)}, \beta_1^{(3)}, \beta_2^{(1)}, \beta_2^{(2)}, \beta_2^{(3)}, \ldots)
m1_beta_min <- m1_beta_array[which.min(m1_cve_array),][-1]

write.csv(m1_beta_min, "m1_coef.csv", row.names=FALSE)


