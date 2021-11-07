library(MFPCA)

mfpca_train = function(multivar, argvals){
    
    # multivariate FPCA based on results from uPACE
    mFPCA = function(Xi, phi, p , L, I=I){
        
        # eigenanalysis on matrix M
        M = t(Xi)%*%Xi/(I-1)
        eigen.M = eigen(M)
        values = eigen.M$values
        pve = cumsum(values)/sum(values)
        Cms = eigen.M$vectors
        index = unlist(lapply(1:length(L), function(x) rep(x, L[x])))
        
        # MFPCA score
        rho = mfpca_score(Xi, Cms)
        
        # MFPCA eigenfunction
        psis = NULL
        for(j in 1:p){
            psi = NULL
            for(m in 1:dim(Cms)[2]){
                psi = cbind(psi, phi[[j]]%*%Cms[which(index==j),m])
            }
            psis[[j]] = psi
        }
        
        out = list(eigenvalue = values, Cms = Cms, pve = pve, rho = rho, psis=psis)
        return(out)
    }  
        
    multivar = as.array(multivar)
    argvals = as.numeric(argvals)
    multivar = ifelse(multivar==0, NA, multivar)
    
    I = dim(multivar)[1]
    Q = dim(multivar)[3]
    
    # univariate FPCA via PACE
    Xi.train = L = phi.train =  NULL
    for(p in 1:Q){
        tmp.ufpca = uPACE(multivar[,,p], argvals)
        Xi.train = cbind(Xi.train, tmp.ufpca$scores) # FPC scores
        L = c(L, dim(tmp.ufpca$scores)[2])
        phi.train[[p]] = t(tmp.ufpca$functions@X) # FPC eigenfunctions
    }

    # multivariate FPCA
    mFPCA.train = mFPCA(Xi=Xi.train, phi=phi.train, p=Q, L=L ,I=I)
    rho.train = mFPCA.train$rho  #MFPC scores
    pve = mFPCA.train$pve
    Cms = mFPCA.train$Cms
    psis = mFPCA.train$psis
    psi = array(dim=c(Q,dim(psis[[1]])[1],dim(psis[[1]])[2]))
    for(p in 1:Q){
        psi[p,,] = psis[[p]]
    }
    
    return(list("scores"=rho.train, "Cms"=Cms, "psi"=psi))
}


mfpca_test = function(multivar.train, multivar.test, argvals, Cms, psi){
    multivar.train = as.array(multivar.train)
    multivar.test = as.array(multivar.test)
    multivar.train = ifelse(multivar.train==0, NA, multivar.train)
    multivar.test = ifelse(multivar.test==0, NA, multivar.test)
    argvals = as.numeric(argvals)
    Q = dim(multivar.test)[3]
    
    # univariate FPC 
    Xi.test = NULL
    meanf = NULL
    for(qq in 1:Q){
        tmp.ufpca = uPACE(multivar.train[,,qq], argvals, multivar.test[,,qq])
        Xi.test = cbind(Xi.test, tmp.ufpca$scores) # dynamic FPC scores for test subjects 
        meanf[[qq]] = tmp.ufpca$mu@X
    }
    
    # estimate MFPC scores for test subjects
    rho.test = mfpca_score(Xi.test, Cms)
    long.pred = mfpca_pred(rho.test, meanf, psi)
    
    return(list("scores"=rho.test, "long_pred"=long.pred))
}



# mfpc score calculation
mfpca_score = function(predXi, Cms){
    rho = matrix(NA, nrow = nrow(predXi), ncol=dim(Cms)[2])
    for(i in 1:nrow(predXi)){
        for(m in 1:dim(Cms)[2]){
            rho[i,m] = predXi[i,]%*%Cms[,m]
        }
    }
    return(rho)
}


# mfpc trajectories prediction
mfpca_pred = function(score, meanf, psi, n.rho=NULL){
    p = dim(psi)[1]
    n = nrow(score)
    
    if(is.null(n.rho)){
        n.rho = ncol(score)
    }
    
    out = NULL
    for(m in 1:p){
        out[[m]] = matrix(meanf[[m]], nrow=n, ncol=length(meanf[[m]]), byrow = T ) + score[,1:n.rho]%*%t(psi[m,,1:n.rho])
    }
    
    long.pred = array(data=NA, dim=c(dim(out[[1]])[1],dim(out[[1]])[2],p))
    for(m in 1:p){
        long.pred[,,m] = out[[m]]
    }
    return(long.pred)
}


# univariate FPCA via principal analysis by conditional estimation(PACE)
uPACE = function(testData, domain, predData=NULL, nbasis = 5, pve = 0.99, npc = NULL){
    tmp = funData(domain, testData)
    if(is.null(predData)){
        tmp2 = NULL
    }else{
        tmp2 = funData(domain, predData)
    }
    res = PACE(tmp, tmp2, pve=pve, npc= npc, nbasis=nbasis)
    return(res)
}



