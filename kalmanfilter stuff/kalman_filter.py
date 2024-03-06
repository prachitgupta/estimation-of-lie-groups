import numpy as np 
import matplotlib.pyplot as plt

class Kalman_Filter():

    ## kalman filter is to be executed in two stages -> 1)Prediction and 2) Update
    ## A = state transition matrix from t-1 to t
    ## B = control input matric ut -> xt
    ## u = control input
    ## X = known estimate at t-1|t-1 known precisely at t=0 and estimated for every subsequent step using kf
    ## P = state covariance 
    ## Q = process noise covariance
    ## H = transition matrix state_space - measurement space
    ## R = measurement noise covariance
    ## Z = sensor measurement

    ##prediction  , apriori estimate of mean and covariance of state vector at time step t from known estimates at t-1
    def prediction(self,A,B,u,X,P,Q):
        ## predict Xt|t-1 from Xt-1|t-1 i.e state mean
        Xpred = np.dot(A,X) + np.dot(B,u)
        ## predict Pt|t-1 from Pt-1|t-1 i.e state covariance
        Ppred = np.dot(A,np.dot(P,np.transpose(A))) + Q
        return Xpred,Ppred

    ##UPDATE  uses prediction for t and noisy sensor measurment at t to estimate state completely at t
    def update(self,A,B,u,Xpred,Ppred,H,R,Z):
        ## estimate measurment covariance with transition matrix H and measuremet noise covariance
        S = np.dot(H,np.dot(Ppred,np.transpose(H))) + R
        ##calculate Kalman gain K
        K = np.dot(Ppred , np.dot(np.transpose(H),np.linalg.inv(S))) 
        ## ESTIMATE Xt|t from Xt|t-1(prediction) and Zt(measurement) i.e state mean
        X_estimated = Xpred + np.dot(K,(Z - np.dot(H,Xpred)))
        ## ESTIMATE Pt|t from Pt|t-1(prediction) and measurement covariance i.e state covariance
        P_estimated = Ppred - np.dot(K, np.dot(H,Ppred))
        return X_estimated,P_estimated
    
###implement train example in ramsay paper 
dt = 0.1
T_simulation = 10
steps = int(T_simulation/dt)
##state space pos and velocity at eact time step
X_space = np.zeros((2,steps+1))
##measurement space pos and velocity measurements at eact time step
Z_space = np.zeros((2,steps+1)) 
##estimate from kalman filter
E_space = np.zeros((2,steps+1))
##Prediction space
P_space = np.zeros((2,steps+1)) 
##initial state (pos and velocitty).flatten()
X0 = np.array([[5],[0]])
##state transition Ft
Ft = np.array([[1,dt],[0,1]])
## control matrix Bt
Bt = np.array([[(np.square(dt))/2.0],[dt]])
##control input
force,mass = 10,1.0
ut = force/mass
## process gaussian noise wt (mean = 0, sigma = 1)
wt = np.random.randn(Ft.shape[0],1)
##process noise covariance matrix
Q = np.square(wt)*np.eye(Ft.shape[0])
##state covariance assumed arbitary for initial state
P = np.diag((0.01,0.01))
##measurement state transition matrix from xt -> zt
H = np.array([[1,0],[0,1]])
## measurement gaussian noise wt (mean = 0, sigma = 1)
vt = np.random.randn(Ft.shape[0],1)
##process noise covariance matrix
R = np.square(vt)*np.eye(Ft.shape[0])

## initialize actual state and state estimation to known X0
Xact = X0 
Xestimated = X0
##initial sensor measurement, 
Z = np.dot(H,Xact) + 1*vt
## run simulation for time steps , generating Xactual from state space and X measurement using KF
for k in range(1,steps+1):
    ##Xactual compute
    Xact = np.dot(Ft,Xact) + np.dot(Bt,ut) + 0.1*wt  
    X_space[:,k] = Xact.flatten()
    ##Kalman filter 
    kf = Kalman_Filter()
    ##predict x using priori known estimates
    Xpred, Ppred = kf.prediction(Ft ,Bt, ut, Xestimated, P, Q)
    P_space[:,k] = Xpred.flatten()
    ###update estimate based on measurement and prediction
    Xestimated, P_est = kf.update(Ft, Bt, ut, Xpred, Ppred, H, R, Z)
    E_space[:,k] = Xestimated.flatten()
    #measurements assumed for now expected plus some gaussian noise
    Z = np.dot(H,Xact) + 35*vt
    Z_space[:,k] = Z.flatten()
   
position_desired = X_space[0, :]
measured_pos = Z_space[0,:]
estimated_pos = E_space[0,:]
predicted_pos = P_space[0,:]
# Generate time array
time = np.linspace(0, T_simulation, steps+1)

##plots data
plt.figure(facecolor= "black")

# Plot each curve with a different color and add legend
plt.plot(time,position_desired, label='ground truth', color='orange')
plt.plot(time,estimated_pos, label='kalman estimates', color='black')
plt.scatter(time,measured_pos, label='sensor_data', color='blue')
# plt.plot(time, predicted_pos, label='prediction', color='orange')

plt.legend()
plt.xlabel('T(s)')
plt.ylabel('Distance')
plt.title('Position of train with time')

# Show the plot
plt.grid(color = "black")
plt.show()


