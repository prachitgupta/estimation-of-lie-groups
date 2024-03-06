import numpy as np
import matplotlib.pyplot as plt


class Integrator_numerical:
    ##spring mass xddot + xdot + x = 0
    ##setup
    def __init__(self,T,step_size):
        self.delt = step_size
        T_simulation  = T
        self.steps = int(T_simulation/self.delt)
        self.time = np.linspace(0,T_simulation,self.steps+1)
         
            
    def pendulum_dynamics(self,Xk):
        Xdot = np.transpose(np.array([Xk[1],-np.sin(Xk[0])]))
        return Xdot
    
    def system_dynamics_initalize(self,X0):
        self.X = np.zeros([2,self.steps+1])
        ##initial conditions with no control input
        self.X_initial = np.transpose(X0)
        self.X[:,0] = self.X_initial
        return self.time

    ##integrators
    def explicit_euler(self):
        
        for k in range(self.steps) :
                   
            Xdot = self.pendulum_dynamics(self.X[:,k])      
            self.X[:,k+1] = self.X[:,k] + Xdot*self.delt
            # #RESTRICT THETA
            # self.X[0,k+1]  = self.X[0,k+1] % 2*np.pi       
            
        x1 , x2 =  self.X[0,:], self.X[1,:]
        return x1, x2
    
    def implicit_euler(self):
        
        ## non linear have to used another numerical solver
        pass
    
    def sympletic_euler(self):
        for k in range(self.steps) :
            dt = self.delt
            # implicit Euler for theta
            self.X[0,k+1] = self.X[0,k] + dt *self.X[1,k] + (-1*dt**2)*np.sin(self.X[0,k])
            # explicit Euler for omega
            self.X[1,k+1] = self.X[1,k] - dt*(np.sin(self.X[0,k]))
        x1 , x2 =  self.X[0,:], self.X[1,:]
        return x1, x2
   
    ### check the energy variation and deviation from expected value as an performance metric of integratorz

    ##error at each step
    def plot_results(self,state1,state2):
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(self.time,state1,color = "black")
        plt.xlabel('time(s)')
        plt.ylabel('theta')
        plt.subplot(2, 1, 2)
        plt.plot(self.time,state2,color = "red")
        plt.xlabel('time')
        plt.ylabel('omega')
        # plt.subplot(3, 1, 3)
        # plt.plot(self.time,error, label='Euler',color = "red")
        # plt.xlabel('Time (s)')
        # plt.ylabel('error')
        plt.show()


##part 2 pendulum thethaddot + sin(thetha) = 0

##initial state (position and velocity)
theta0 = np.array([np.pi/2,0])

##system
Pendulum = Integrator_numerical(100,0.1) ##(SIMULATION TIME , DELTA t) 
t = Pendulum.system_dynamics_initalize(theta0)    

##explicit euler
thetae,omegae = Pendulum.explicit_euler()
Pendulum.plot_results(thetae,omegae)

##sympletic euler
thetas,omegas = Pendulum.sympletic_euler()
Pendulum.plot_results(thetas,omegas)
