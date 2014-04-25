import numpy as np
import scipy.integrate
import scipy.interpolate

### General DDE adaptation for scipy.ode.integrate
class DelayFunc:
    def __init__(self,f,t0=0):
        self.f = f
        self.t0= t0
        self.itpr = scipy.interpolate.interp1d(np.array([t0-1,t0]), np.array([self.f(t0-1),self.f(t0)]).T, kind='linear', bounds_error=False, fill_value = self.f(t0))
 
    def update(self,t,Y):
        newx = np.append(self.itpr.x, t)
        if Y.shape[0] != self.itpr.y.shape[0]:
            Y = np.array(Y).T
        elif len(Y.shape) ==1:
            Y = np.array(Y).reshape(self.itpr.y.shape[0],1)
        else:
            print "Incorrect dimension of Y"
        newy = np.hstack([self.itpr.y, Y])
        self.itpr = scipy.interpolate.interp1d( newx, newy, bounds_error = False, fill_value = self.f(self.t0) )
 
    def __call__(self,t=0):
        return np.array((self.f(t)) if (t<=self.t0) else np.array(self.itpr(t)))
 
class dde(scipy.integrate.ode):
    def __init__(self,f,jac=None):
        def f2(t,y,args):
            return f(self.Y,t,*args)
        scipy.integrate.ode.__init__(self,f2,jac)
        self.set_f_params(None)
 
    def integrate(self, t, step=0, relax=0):
        scipy.integrate.ode.integrate(self,t,step,relax)
        self.Y.update(self.t,self.y)
        return self.y
 
    def set_initial_value(self,Y):
        self.Y = Y
        scipy.integrate.ode.set_initial_value(self, Y(Y.t0), Y.t0)
 
def ddeint(deriv,deriv_hist,ts,fargs=[]):
    dfun = dde(deriv)
    dfun.set_initial_value(DelayFunc(deriv_hist,ts[0]))
    dfun.set_f_params(fargs)
    return [dfun.integrate(dfun.t+dt) for dt in np.diff(ts)]


###EXAMPLE###
# This solves the delay differential equation where:
# dx/dt = 0.5x(t) - x(t-d) + 2*y(t-d)
# dy/dt = 0.5y(t) - x(t-d) -y(t-d)

# g1 defines this derivative.
# g2 defines the "history" of the derivative (for t<d)
# Dmat is a helper for the derivative function
# d is the delay
# tt is the time points at which the solution should be produced
#############
if __name__ =="__main__":
    Dmat = 0.2*np.array([0.5,0,-1,2,0,0.5,-1,-1]).reshape(2,4)
    def g1(X,t,d):
        x = np.append(X(t),X(t-d))
        d1 = Dmat.dot(x)
        return 

    g2 = lambda t : np.array([np.sin(t),10-t])
    tt = np.linspace(0,30,20000)
    d = 0.5
    yy = ddeint(model,g,tt,fargs=(d,))

