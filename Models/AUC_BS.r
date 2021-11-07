library(survival)
library(tdROC)

AUC_1 = function(surv, event, time, pt){
    roc = tdROC( X = surv, Y = time, 
                 delta = event,
                 tau = pt, span = 0.05,
                 nboot = 0, alpha = 0.05,
                 n.grid = 1000, cut.off = 0.5)
    
    return(roc$AUC$value)
}


AUC = function(surv, event, time, predtimes){
    aucs = rep(NA,length(predtimes))
    
    # if nonPH, surv should be matrix of Ixlength(predtimes)
    if(is.matrix(surv)){
        for(i in 1:length(predtimes)){
            aucs[i] = AUC_1(1-surv[,i], event, time, predtimes[i])
        }
    }else if(is.vector(surv)){
        for(i in 1:length(predtimes)){
            aucs[i] = AUC_1(1-surv, event, time, predtimes[i])
        }
    }
    
    return(list("auc"=aucs))
}


Brier = function(surv, event, time, event_train, time_train, LT, DeltaT){
    #estimate km curve for BS calculation
    train.surv = cbind.data.frame("event"=event_train, "time"=time_train)
    km = survfit(Surv(time, event)~1, data=train.surv)
    survest = stepfun(km$time, c(1, km$surv))
    
    BS = rep(NA, length(DeltaT))
    for(i in 1:length(DeltaT)){
        pt = LT + DeltaT[i]
        N_vali = length(event)
        
        #BRIER SCORE
        D = rep(0, N_vali) 
        D[time<=pt & event==1] = 1
        
        pi = 1-surv[,i]
        
        km_pts = survest(time)/survest(LT)
        W2 <- D/km_pts
        W1 <- as.numeric(time>pt)/(survest(pt)/survest(LT))
        W <- W1 + W2
        
        BS_pts <- W * (D - pi)^2
        BS[i] = sum(na.omit(BS_pts)) / N_vali
    }
    return(BS)
}





