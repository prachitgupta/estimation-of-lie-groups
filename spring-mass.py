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
         
    def spring_mass_dynamics(self,Xk):
        ##state transition matrix
        A =  (np.array([[0,1], [-1,-1]]))
        return A@Xk
            
    
    def system_dynamics_initalize(self,X0):
        self.X = np.zeros([2,self.steps+1])
        ##initial conditions with no control input
        self.X_initial = np.transpose(X0)
        self.X[:,0] = self.X_initial
        return self.time

    ##integrators
    def explicit_euler(self):
        
        for k in range(self.steps) :
            Xdot = self.spring_mass_dynamics(self.X[:,k])
            self.X[:,k+1] = self.X[:,k] + Xdot*self.delt              
        x1 , x2 =  self.X[0,:], self.X[1,:]
        return x1, x2
    
    def implicit_euler(self,dt):
        ## matrix multiplication fuction
        G = np.array([[1,(-dt)],[dt,(1+dt)]])
        ##inverse 
        Ginv = np.linalg.inv(G)
        for k in range(self.steps) :
            self.X[:,k+1] = Ginv@self.X[:,k]    
        x1 , x2 =  self.X[0,:], self.X[1,:]
        return x1, x2
    def symplectic_euler(self,dt):
        ## matrix multiplication fuction
        G = np.array([[1 - ((-1*(dt**2))/(dt+1)),dt/(dt+1)],[-dt/(dt+1),1/(dt+1)]])
        for k in range(self.steps) :
            self.X[:,k+1] = G@self.X[:,k]    
        x1 , x2 =  self.X[0,:], self.X[1,:]
        return x1, x2
        
   
    ### true value obtained analytically
    def error(self,analytical_sol,numerical_sol):
        error = analytical_sol - numerical_sol
        return error

    ##error at each step
    def plot_results(self,state1,state2,error):
        # plt.figure(figsize=(10, 6))
        # plt.subplot(3, 1, 1)
        # plt.plot(self.time,state1,color = "")
        # plt.xlabel('Time (s)')
        # plt.ylabel('state1')
        # plt.subplot(3, 1, 2)
        # plt.plot(self.time,state2,color = "green")
        # plt.xlabel('Time (s)')
        # plt.ylabel('state2')
        # plt.subplot(3, 1, 3)
        # plt.plot(self.time,error, label='Euler',color = "red")
        # plt.xlabel('Time (s)')
        # plt.ylabel('error')
        # plt.show()
        # Define custom colors and marker styles
        colors = ['blue', 'orange', 'green']  # Custom colors for each subplot
        marker_styles = ['o', '.', '*']  # Custom marker styles for each subplot

        # Create the figure and subplots
        plt.figure(figsize=(10, 6))

        # Subplot 1
        plt.subplot(3, 1, 1)
        plt.plot(self.time, state1, color=colors[0], linestyle='-', label='position')
        plt.xlabel('time')
        plt.ylabel('position')
        plt.legend()

        # Subplot 2
        plt.subplot(3, 1, 2)
        plt.plot(self.time, state2, color=colors[1], linestyle='--', label='velocity')
        plt.xlabel('time')
        plt.ylabel('velocity')
        plt.legend()

        # Subplot 3
        plt.subplot(3, 1, 3)
        plt.plot(self.time, error, color=colors[2], linestyle='-', label='error')
        plt.xlabel('time')
        plt.ylabel('Error')
        plt.legend()

        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()

##part 1 spring mass damper xddot + xdot +x = 0
##initial state (position and velocity)
X0 = np.array([10,0])

##system
SpringMass = Integrator_numerical(10,0.01) ##(SIMULATION TIME , DELTA t) 
t = SpringMass.system_dynamics_initalize(X0)     
analytical_sol = 5*(np.e**(t/2)) * (np.cos((np.sqrt(3))/2))

##explicit euler
sf,vf = SpringMass.explicit_euler()
errorf = SpringMass.error(analytical_sol, sf)
SpringMass.plot_results(sf, vf, errorf)

##implicit euler
dt = SpringMass.delt
sb,vb = SpringMass.implicit_euler(dt)
errorb = SpringMass.error(analytical_sol, sb)
SpringMass.plot_results(sb, vb, errorb)

##symplectic Euler
##use explicit Euler to advance position and use implicit Euler on the velocity
dt = SpringMass.delt
ss,vs = SpringMass.symplectic_euler(dt)
errors = SpringMass.error(analytical_sol, ss)
SpringMass.plot_results(ss, vs, errors)