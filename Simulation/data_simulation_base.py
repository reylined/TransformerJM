import numpy as np
import pandas as pd
import scipy.integrate as integrate
import scipy.optimize as optimize




def simulate_JM_base(I, obstime, miss_rate=0.1, opt="none", seed=None):
    
    if seed is not None:
        np.random.seed(seed)
    
    J = len(obstime)
    
    #### longitudinal submodel ####
    beta0 = np.array([1.5,2,0.5],)
    beta1 = np.array([2,-1,1])
    betat = np.array([1.5, -1, 0.6]) 
    b_var = np.array([1,1.5,2])
    e_var = np.array([1,1,1])
    rho = np.array([-0.2,0.1,-0.3])
    b_Sigma = np.diag(b_var)
    b_Sigma[0,1] = b_Sigma[1,0] = np.sqrt(b_var[0]*b_var[1])*rho[0]
    b_Sigma[0,2] = b_Sigma[2,0] = np.sqrt(b_var[0]*b_var[2])*rho[1]
    b_Sigma[1,2] = b_Sigma[2,1] = np.sqrt(b_var[1]*b_var[2])*rho[2]
    

    X = np.random.normal(3,1,size=I)
    ranef = np.random.multivariate_normal(mean=[0,0,0], cov=b_Sigma, size=I)
    mean_long = beta0 + np.outer(X,beta1)
    eta_long = mean_long + ranef


    if opt=="none":
        gamma = np.array([-4,-2])
        alpha = np.array([0.2,-0.2,0.4])
        x1 = np.random.binomial(n=1,p=0.5,size=I)
        x2 = np.random.normal(size=I)
        W = np.stack((x1,x2), axis=1)
        eta_surv = W@gamma + eta_long@alpha
        base = W[...,np.newaxis]
        
    if opt=="interaction":
        gamma = np.array([-4,-2,3])
        alpha = np.array([0.2,-0.2,0.4])
        x1 = np.random.binomial(n=1,p=0.5,size=I)
        x2 = np.random.normal(size=I)
        x3 = x1*x2
        W = np.stack((x1,x2,x3), axis=1)
        eta_surv = W@gamma + eta_long@alpha
        base = np.stack((x1,x2), axis=1)
        base = base[...,np.newaxis]


    #Simulate Survival Times using Inverse Sampling Transform
    scale = np.exp(-7)
    U = np.random.uniform(size=I)
    alpha_beta = alpha@betat
    
    def CHF(tau):
        def h(t):
            return scale * np.exp(eta_surv[i] + alpha_beta*t)
        return np.exp(-1 * integrate.quad(lambda xi: h(xi),0,tau)[0])
        
    Ti = np.empty(I)
    Ti[:] = np.NaN
    for i in range(0,I):
        Ti[i] = optimize.brentq(lambda xi: U[i]-CHF(xi), 0, 100)
    
    
    #Get true survival probabilities
    true_prob = np.ones((I, len(obstime)))
    for i in range(0,I):
        for j in range(1,len(obstime)):
            tau = obstime[j]
            true_prob[i,j] = CHF(tau)

    C = np.random.uniform(low=obstime[3], high=obstime[-1]+25, size=I)
    C = np.minimum(C, obstime[-1])
    event = Ti<C
    true_time = np.minimum(Ti, C)

    # round true_time up to nearest obstime
    time = [np.min([obs for obs in obstime if obs-t>=0]) for t in true_time]
    
    
    subj_obstime = np.tile(obstime, reps=I)
    pred_time = np.tile(obstime, reps=I)
    mean_long = np.repeat(mean_long, repeats=J, axis=0)
    eta_long = np.repeat(eta_long, repeats=J, axis=0)
    long_err = np.random.multivariate_normal(mean=[0,0,0], cov=np.diag(e_var), size=I*J)
    Y = np.empty((I*J,3))
    Y_pred = np.empty((I*J,3))
    for i in range(0,3):
        Y[:,i] = eta_long[:,i] + betat[i]*subj_obstime + long_err[:,i]
        Y_pred[:,i] = eta_long[:,i] + betat[i]*pred_time + long_err[:,i]
    true_prob = true_prob.flatten()
    ID = np.repeat(range(0,I), repeats=J)
    visit = np.tile(range(0,J), reps=I)
    data = pd.DataFrame({"id":ID, "visit":visit, "obstime":subj_obstime, "predtime":pred_time,
                        "time":np.repeat(time,repeats=J),
                        "event":np.repeat(event,repeats=J),
                        "Y1":Y[:,0],"Y2":Y[:,1],"Y3":Y[:,2],
                        "X1":np.repeat(base[:,0],repeats=J),
                        "X2":np.repeat(base[:,1],repeats=J),
                        "pred_Y1":Y_pred[:,0],"pred_Y2":Y_pred[:,1],
                        "pred_Y3":Y_pred[:,2],"true":true_prob})
    
    
    return data
    
