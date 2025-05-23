# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:08:02 2015

@author: gabeo

define activation function for Poisson model
"""

''' here, exponential nonlinearity '''
#def phi(g,gain):

#    import numpy as np
#
#    g_calc = np.exp(g*1)

#    r_out = g_calc

#    return r_out

# def phi_prime(g,gain):
#
#     import numpy as np
#
#     phi_pr = np.exp(g*1)
#
#     return phi_pr

# def phi_prime2(g,gain):
#
#     '''
#     second derivative of phi wrt input
#     '''
#
#     import numpy as np
#
#     phi_pr2 = np.exp(g*1)
#
#     return phi_pr2

# def phi(g,gain):
    
#     import numpy as np

#     g_calc = np.maximum(0,10*(g-1))

#     r_out = gain*g_calc

#     return r_out

''' here, modified sigmoid '''

def sigmoid(g, gain):
    
    import numpy as np

    g_calc = 1 / (1 + np.exp(-gain * (g * 1 - 0.5)))

    r_out = 1 * g_calc

    return r_out


''' Sigmoid for the MF Equations'''
def sigmoid_MF(g, gS, gain):

    import numpy as np

    g_calc = 1 / (1 + np.exp(-gain * (np.sqrt(8/np.pi)) * ((g - 0.5) / np.sqrt(gS))))

    r_out = 1 * g_calc

    return r_out

def threshold_power_law(g, gain, power):

    import numpy as np

    g_calc = np.maximum(g-1,0)**power

    return g_calc

def exponential(g, gain, thres):

    import numpy as np

    g_calc = np.exp(gain*(g-thres))

    return g_calc


# def phi(g,gain):

#     import numpy as np

#     g_calc = np.maximum(0,np.tanh(10*(g-1)))

#     r_out = gain*g_calc

#     return r_out

# def phi(g,gain):

#     import numpy as np

#     g_calc = np.maximum(0,4*(g-1)*(2-g))

#     r_out = gain*g_calc

#     return r_out

''' here, original sigmoid '''

# def phi(g,gain):
    
#     import numpy as np

#     g_calc = 1/(1+np.exp(-g*1))

#     r_out = gain*g_calc

#     return r_out

# def phi1(g,gain):
#
#     import numpy as np
#
#     idx = np.where(g < 0)
#     g[idx] = 0
#     g_calc = np.power(g,2)
#
#     r_out = gain*g_calc
#
#     return r_out


# def phi(g,gain):

#     import numpy as np

#     g_calc = 0.5*(g + np.sqrt(g*g + 4))

#     r_out = gain*g_calc

#     return r_out

''' here, modified sigmoid for a single population'''

#def phi(g,gain):
    
#    import numpy as np

#    g_calc = 1/(1+np.exp(-2.5*(g*1 - 1)))

#    r_out = gain*g_calc

#    return r_out


''' here, tanh '''

#def phi(g,gain):
    
#    import numpy as np

#    g_calc = (1 + np.tanh(g - 5.0))/2

#    r_out = g_calc

#    return r_out

# # # #
# ''' here, half-wave rectified linear '''
#
# def phi(g, gain):
#    '''
#    voltage-rate transfer
#    '''
#
#    import numpy as np
#
#    g_calc = g*1
#
#    thresh = 0
#    ind = np.where(g_calc<thresh)[0]
#    g_calc[ind] = 0
#
#    r_out = gain*g_calc
#
#    return r_out
#
#
# def phi_prime(g,gain):
#
#    import numpy as np
#
#    g_calc = g*1
#    ind = np.where(g_calc<0)[0]
#
#    phi_pr = gain*np.ones(np.shape(g))
#    phi_pr[ind] = 0
#
#    return phi_pr
#
#
# def phi_prime2(g,gain):
#
#    '''
#    second derivative of phi wrt input
#    '''
#
#    import numpy as np
#
#    g_calc = g*1
#    thresh = 0.
#    ind = np.where(g_calc == thresh)
#    phi_pr2 = np.zeros(g.shape)
#    # phi_pr2[ind] = 1.
#
#    return phi_pr2
#
#
# ''' here, concave down'''
#
# def phi(g, gain):
#
#     import numpy as np
#
#     # g_calc = g*1.
#
#     thresh = 0.
#     # ind = np.where(g_calc<thresh)
#     # g_calc[ind[0]] = 0.
#     #
#     # r_out = gain/2.*(1 + np.tanh(g/gain))
#     r_out = gain * (np.tanh(g/gain))
#     r_out[r_out <= thresh] = 0.
#
#
#     # r_out = gain*(1 + np.exp(-g_calc/gain))**-1
#     #
#     # g_calc[g_calc < thresh] = thresh
#     # r_out = gain*np.sqrt(g_calc)
#
#     return r_out
#
#
# def phi_prime(g, gain):
#
#     import numpy as np
#
#     g_calc = g*1.
#     thresh = 0.
#     # ind = np.where(g_calc < thresh)[0]
#     # g_calc[ind] = 0.
#
#     # phi_pr = np.exp(g_calc/gain) * (1+np.exp(g_calc/gain))**-2
#
#     r = phi(g, gain)
#     phi_pr = gain*np.cosh(g/gain)**(-2)
#     phi_pr[r <= thresh] = 0.
#
#     # g_calc[g_calc < thresh] = thresh
#     # phi_pr = gain * .5 * g_calc**(-.5)
#
#     return phi_pr
#
#
# def phi_prime2(g, gain):
#
#     '''
#     second derivative of phi wrt input
#     '''
#
#     import numpy as np
#
#     g_calc = g*1
#     thresh = 0.
#     # ind = np.where(g_calc < thresh)
#     # g_calc[ind[0]] = 0.
#     #
#
#     # phi_pr2 = -(np.exp(g_calc/gain)-1)*np.exp(g_calc/gain) / (gain*(np.exp(g_calc/gain)+1)**3)
#
#     r = phi(g, gain)
#     phi_pr2 = -2. * np.cosh(g_calc/gain)**(-3) * np.sinh(g_calc/gain)
#     phi_pr2[r <= thresh] = 0.
#
#     # g_calc[g_calc < thresh] = thresh
#     # phi_pr2 = gain * .5 * -.5 * g_calc**(-1.5)
#
#     return phi_pr2
#
#
# ''' here, linear for E and quadratic for I '''
#
# def phi(g,gain):
#
#     import numpy as np
#     import params
#     reload(params)
#     par = params.params()
#
#     g_calc = g*1.
#
#     thresh = 0.
#     ind = np.where(g_calc < thresh)
#     g_calc[ind[0]] = 0.
#
#     r_out = np.zeros(g_calc.shape)
#     r_out[:par.Ne] = gain*g_calc[:par.Ne]**1
#     r_out[par.Ne:] = gain*g_calc[par.Ne:]**2
#     # r_out = gain*(g_calc**2)
#
#     return r_out
#
# def phi_prime(g,gain):
#
#     import numpy as np
#     import params
#     reload(params)
#     par = params.params()
#
#     g_calc = g*1.
#     thresh = 0.
#     ind = np.where(g_calc < thresh)
#     g_calc[ind[0]] = 0.
#
#     phi_pr = np.zeros(g_calc.shape)
#     phi_pr[:par.Ne] = gain*np.ones(par.Ne)
#     phi_pr[par.Ne:] = gain*2.*g_calc[par.Ne:]
#     phi_pr[ind[0]] = 0.
#
#     return phi_pr
#
# def phi_prime2(g,gain):
#
#     '''
#     second derivative of phi wrt input
#     '''
#
#     import numpy as np
#     import params
#     reload(params)
#     par = params.params()
#
#     g_calc = g*1
#     thresh = 0.
#     ind = np.where(g_calc < thresh)
#     g_calc[ind[0]] = 0.
#
#     phi_pr2 = np.zeros(g_calc.shape)
#     phi_pr2[:par.Ne] = 0.
#     phi_pr2[par.Ne:] = gain*2.*np.ones(par.Ni)
#     phi_pr2[ind[0]] = 0.
#
#     return phi_pr2
#
#
# def phi_pop(g,gain):
#
#     '''
#
#     :param g: 2d (E, I)
#     :param gain: 2d(E, I)
#     :return:
#     '''
#
#     import numpy as np
#     import params
#     reload(params)
#     par = params.params()
#
#     g_calc = g*1.
#
#     thresh = 0.
#     ind = np.where(g_calc < thresh)
#     g_calc[ind[0]] = 0.
#
#     r_out = np.zeros(g_calc.shape)
#     r_out[0] = gain*g_calc[0]**1
#     r_out[1] = gain*g_calc[1]**2
#     # r_out = gain*(g_calc**2)
#
#     return r_out

# #
# ''' here, quadratic for E and linear for I '''
#
# def phi(g,gain):
#
#     import numpy as np
#     import params
#     reload(params)
#     par = params.params()
#
#     g_calc = g*1.
#
#     thresh = 0.
#     ind = np.where(g_calc < thresh)
#     g_calc[ind[0]] = 0.
#
#     r_out = np.zeros(g_calc.shape)
#     r_out[:par.Ne] = gain*g_calc[:par.Ne]**2
#     r_out[par.Ne:] = gain*g_calc[par.Ne:]**1
#     # r_out = gain*(g_calc**2)
#
#     return r_out
#
# def phi_prime(g,gain):
#
#     import numpy as np
#     import params
#     reload(params)
#     par = params.params()
#
#     g_calc = g*1.
#     thresh = 0.
#     ind = np.where(g_calc < thresh)
#     g_calc[ind[0]] = 0.
#
#     phi_pr = np.zeros(g_calc.shape)
#     phi_pr[:par.Ne] = gain*2.*g_calc[:par.Ne]
#     phi_pr[par.Ne:] = gain*np.ones(par.Ni)
#     phi_pr[ind[0]] = 0.
#
#     return phi_pr
#
# def phi_prime2(g,gain):
#
#     '''
#     second derivative of phi wrt input
#     '''
#
#     import numpy as np
#     import params
#     reload(params)
#     par = params.params()
#
#     g_calc = g*1
#     thresh = 0.
#     ind = np.where(g_calc < thresh)
#     g_calc[ind[0]] = 0.
#
#     phi_pr2 = np.zeros(g_calc.shape)
#     phi_pr2[:par.Ne] = gain*2.*np.ones(par.Ne)
#     phi_pr2[par.Ne:] = 0.
#     phi_pr2[ind[0]] = 0.
#
#     return phi_pr2
#
#
# def phi_pop(g,gain):
#
#     '''
#
#     :param g: 2d (E, I)
#     :param gain: 2d(E, I)
#     :return:
#     '''
#
#     import numpy as np
#     import params
#     reload(params)
#     par = params.params()
#
#     g_calc = g*1.
#
#     thresh = 0.from theory import rates_ss
#     ind = np.where(g_calc < thresh)
#     g_calc[ind[0]] = 0.
#
#     r_out = np.zeros(g_calc.shape)
#     r_out[0] = gain*g_calc[0]**2
#     r_out[1] = gain*g_calc[1]**1
#     # r_out = gain*(g_calc**2)
#
#     return r_out
