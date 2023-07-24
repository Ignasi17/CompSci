import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import time


class Dataframe():
    """
    Class to manage the sets of data used. Each instance of the class works as a one independent set of data.
    If two datasets are introduced, x and y, the class saves the data as an array [x,y].

    Atributtes:

    n -> number of dimensions of the input data
    data -> input data. If the input are two data arrays, they will be merged in one array with form [x, y]

    Methods:

    -init: If the input is only 1D data, saves it as self.data

           If the input is 2D data (x and y):
                If grid = True: it computes and save the grid with (x,y) as self.grid. Also saves as data the x and y of the grid flattened.
                If grid = False: it merges x and y in an array [x, y] and save it in self.data 

    -grid: only available if the data is 2-dimensional. Returns a grid using as axes x and y. 

    -split: nethod that splits the data in train and test data and, also, an input y as predicted and
            real data. It uses the scikit function sklearn.model_selection.train_test_split

    -feature_matrix: Returns a feature matrix computed with Scikit: sklearn.preprocessing.PolynomialFeatures
                    Input: desired degree of the feature matrix. 1 by default.

    -scaling: calling this function, we standard scale the data. It is made using a Scikit function: 
            sklearn.preprocessing.StandardScaler

    """
    def __init__(self, x, y=None, grid=False):
    

        if type(y)==type(None):
            self.data = x
            self.n = self.data.shape

        elif type(y)!= type(None) and grid==False:

            self.data = np.column_stack((x, y))
            self.n = (x.shape[0], y.shape[0])

        elif type(y)!= type(None) and grid==True:

            self.grid = np.array(np.meshgrid(x, y))
            self.data = np.column_stack((self.grid[0].flatten(), self.grid[1].flatten()))
            self.n = (x.shape[0], y.shape[0])
        
        
            

    def split(self,y, train_size=0.7):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data, y,
        train_size=train_size)


    def grid(self):
        if len(self.n)==1:
            return 0
        else:
            return np.meshgrid(self.data[:,0], self.data[:,1])

    def feature_matrix(self, n=2, data=None, scaling = False, fit_intercept=False):
   
        poly = PolynomialFeatures(n)
        
        if type(data)==type(None):
            if self.data.ndim ==1:
                self.data = self.data.reshape(-1,1)
            matrix = poly.fit_transform(self.data)
        else:
            if data.ndim ==1:
                data = data.reshape(-1,1)
            matrix = poly.fit_transform(data)
             
        if scaling == True:

            scaler = StandardScaler()

            if fit_intercept:
                matrix = scaler.fit_transform(matrix)
            elif fit_intercept==False:
                matrix = scaler.fit_transform(matrix)[:,1:]

        return matrix

    def scale(self, data):
        return  StandardScaler().fit_transform(data)



class OLS_SVD():

    """
    Ordinary least square model using Singular Value Decomposition.
    This class computes the OLS model. To compute the inverse required, it uses SVD
    """
    #def __init__(self, SVD=True):  
    def fit(self, y):
        self.y = y
    
    def predict(self, X):

        
        u,_,_=np.linalg.svd(X, full_matrices=False)
        return u@u.T@self.y
       


class Ridge():
    """
    Ridge regression

    """
    def fit(self, feature_matrix, y,l=0.01):

        if feature_matrix.shape[0] != y.shape[0]:

                print("Feature matrix first dimension (",feature_matrix.shape[0],") and y dimension: ",y.shape[0]," don't match.")
                return 0

        elif feature_matrix.shape[0] == y.shape[0]:

            inverted = np.linalg.inv(feature_matrix.T@feature_matrix + l)
            optimal_beta = inverted@feature_matrix.T@y
            self.beta = optimal_beta
    
    def predict(self, feature_matrix):

        return feature_matrix@self.beta

class OLS():
    """
    Ordinary last square model. 
    It contains the following methods to compute the parameters:
        - OLS: inverse of the feature matrix using numpy.linalg.pinv
        - GD:  plain gradient descent method with and without momentum.
        - GD_MB: stochastic gradient descent method with and without momentum.
    It also allows to set the following learning rates schedulers:
        -ADAM
        -AdaGrad
        -RMSprop
        -Exponential
        -Linear
        -Inverse
    
    Atributes:
        -method: method to solve OLS
        -max_iter: maximum number of iterations for GD
        -epsilon: stop criterion
        -gamma: initial learning rate
        -sigma: momentum
        -n_epochs: number of epochs
        -mb: number of mb

    Methods:
    User only have to work with 'fit' function. With 'method' it chooses the technique used (direct OLS, GD, SGD) and
    with 'mode' the type of learning rate scheduler. 
    For the user:
        -predict: predict result with a given feature matrix. 
        -fit: function to compute the parameters of the regression.
            Methods:OLS, GD, GD_MB
            Modes:lr_linear, lr_inv, ADAgrad, RMSprop, ADAM

    For internal operation. This methods are used  for the fit method to work:
        -nabla: function for intern use. It defines the gradient used for the gradient descent techniques.
        -GD_MB: function to compute stochastic gradient descent with a given number of minibatches
        -lr_tun: function used to coordinate the learning rate schedulers.
        -ADAgrad: adagrad method implementation
        -ADAM: adam method implementation
        -RMSprop: rmsprop method implementation
        -GD: function to compute plain gradient descent with a given number of minibatches

        
    """


   

    def __init__(self, method = "OLS", beta0=None):
         
        self.method = method
        self.max_iter= 100000
        self.epsilon = 0.000001
        self.gamma=0.01
        self.sigma=0
        self.n_epochs = 1
        self.mb = 1
    

    #Predict function
    def predict(self, feature_matrix):

        return feature_matrix@self.beta


    def fit(self, feature_matrix, y, beta0=None, gamma=None,n_epochs=None, mb=None, max_iters=None, sigma=None, epsilon=None,
            gammatau=None, alpha=1, rho1=None, rho2=None,mode="fix"):

        if type(gamma) == type(None):
            gamma = self.gamma
        if type(gammatau) == type(None):
            gammatau = self.gamma/100
        if type(max_iters) == type(None):
            max_iters = self.max_iter
        if type(epsilon) == type(None):
            epsilon = self.epsilon
        if type(n_epochs)==type(None):
            n_epochs = self.n_epochs
        if type(mb)==type(None):
            mb = self.mb
        if type(sigma)==type(None):
            sigma = self.sigma
        if type(beta0)==type(None):
            beta0 = np.ones(feature_matrix.shape[1])*0.1
        if type(rho1)==type(None):
            rho1 = 0.9
        if type(rho2)==type(None):
            rho2 = 0.99
           

        if self.method=="OLS":
            if feature_matrix.shape[0] != y.shape[0]:

                    print("Feature matrix first dimension (",feature_matrix.shape[0],") and y dimension: ",y.shape[0]," don't match.")
                    return 0

            elif feature_matrix.shape[0] == y.shape[0]:

                inverted = np.linalg.pinv(feature_matrix.T@feature_matrix)
                optimal_beta = inverted@feature_matrix.T@y
                self.beta = optimal_beta
        

        elif self.method=="GD": #Gradient descent method with momentum

  
            #check that the dimensions are right
            if feature_matrix.shape[0] != y.shape[0]:
                print("Feature matrix first dimension (",feature_matrix.shape[0],") and y dimension: ",y.shape[0]," don't match.")
                return 0
            #if everything is right, starts the fit
            elif feature_matrix.shape[0] == y.shape[0]:

                #the following two parameters are used only to define the gradient. Each time we fit
                #with new data, the values of the gradient change.
                self._feat_matrix_gd = feature_matrix
                self._y_gd = y
                #calling of the gradient function
                self.beta = self.GD(x=beta0, gamma=gamma, epsilon=epsilon, max_iters=max_iters, sigma=sigma,
                gammatau=gammatau,mode=mode, alpha=alpha )

        elif self.method=="GD_MB": #Gradient descent with minibatches


            self.beta = self.GD_mb(x=beta0,feat=feature_matrix, y=y,gamma = gamma, mb=mb,n_epochs=n_epochs, sigma=sigma,
            gammatau=gammatau,mode=mode, alpha=alpha,rho1=rho1, rho2=rho2)
                
                #Starts the plain gradient descent

    #Definition of the gradient of the OLS
    #Function to compute the gradient. This function is only used by the method fit.
    def nabla(self, beta):

        feature_matrix = self._feat_matrix_gd
        y = self._y_gd
  

        return feature_matrix.T@(feature_matrix@beta - y)


    #Mini batches gradient descent
    def GD_mb(self, x,feat, y, mb, n_epochs, gamma, sigma ,gammatau,mode, alpha, 
            rho1, rho2):

        

        #here we made the reshuffle and minibatches

        data = np.column_stack((feat,y))
        data = shuffle(data)
        data = data.reshape((mb, int(feat.shape[0]/mb), data.shape[1])).transpose(1,2,0)
        gamma0 = gamma

    
        xprev = x

        M = int(feat.shape[0]/mb)#data size in each minibatches

        self.increment = np.zeros((feat.shape[1], n_epochs))
        start_time = time.time()
        for epoch in range(1, n_epochs + 1):
            #learning rate tuning
            
   
            for i in range(mb):
                
                
                #print("X0 = ", x)

                random_index = M*np.random.randint(mb)
                #print("Random index ", random_index)
                #print("M ", M)

                #We set, for each iteration and new minibatche, a new feat_matrix to compute the gradient
                self._feat_matrix_gd = feat[random_index:random_index+M,:]
                self._y_gd = y[random_index:random_index +M]

              

                #we redefine nabla for each minibatche


                if mode != "fix":

                    gamma = self.lr_tun(k=epoch, iter=i,gamma0=gamma0, gammatau=gammatau, alpha=alpha,mode=mode,x=x,
                    rho1=rho1, rho2=rho2,mb=mb)
                   
                
                #THE ALGORITHM
               
                #For ADAM the procedure is a bit different 
                if mode=="ADAM":
                    
                    xnext = x - gamma + sigma*(x - xprev)
                else:
                    
                    xnext = x - gamma*self.nabla(x) + sigma*(x - xprev)
                xprev = x
                x = xnext
                
                
                #print("XNEXT = " ,xnext)
               
                if np.isnan(np.sum(x)):
                    print(f"NaN value in epoch {epoch}")
                    return x
            self.increment[:, epoch-1] = x
            self.iterGD_md = self.nabla(x)
            self.epoch_num = epoch
            #learning rate tuning
     
        stop_time = time.time()
        print("Time {} seconds".format((stop_time - start_time)))
        print(f"{epoch} epochs done with grad. {self.nabla(x)}.")
        self.time_GDMB = stop_time - start_time
        return x



        

    #Function to tune the learning rate. With mode, the tuning method is choosed
    def lr_tun(self, k, gamma0, gammatau,mode, iter,alpha, x, rho1,rho2,mb,delta = 1e-8):
        if mode == "lr_exp":
            k -=1
            return gamma0*np.exp(-k*gammatau)
        elif mode =="lr_linear":
            k-=1
            alpha = k/100
            return (1 - alpha)*gamma0 + alpha*gammatau
        elif mode =="lr_inv":
            k-=1
            return gamma0/(k*2+iter + gamma0*5)
        elif mode =="ADAgrad":

            return self.ADAgrad(x=x, iter=iter, delta=delta, gamma0=gamma0)
        elif mode=="RMSprop":
            return self.RMSprop(x=x, iter=iter, delta=delta, gamma0=gamma0, rho1=rho1 )
        elif mode=="ADAM":
            return self.ADAM(x=x, iter=iter, delta=delta, gamma0=gamma0, rho1=rho1, rho2=rho2,
                epoch = k, mb=mb )
            
    ## ADAgrad method
    def ADAgrad(self, x, iter, gamma0, delta ):

        
        g = self.nabla(x)
        
        
        if iter == 0:
            self.__getir = g*g
           
        else:
            self.__getir += g*g

        
        den = delta + np.sqrt(self.__getir)
        return gamma0*1/den

    ## RMSprop method
    def RMSprop(self, x, iter, gamma0, delta, rho1 ):

        
        g = self.nabla(x)
        
        
        if iter == 0:
            self.__getir = (1-rho1)*g*g
        else:
            self.__getir +=  rho1*self.__getir + (1 - rho1)*g*g

        
        den = delta + np.sqrt(self.__getir)
        return gamma0*1/den

    ## ADAM method
    def ADAM(self, x, iter, gamma0, rho1, rho2, delta, epoch, mb):

        g = self.nabla(x)

        
        epoch -=1
        iter += 1
        
        t = epoch*mb + iter

      
        if iter==1:
            #print((1 - rho1**(epoch*mb + iter)))
            self.__getir = (1 - rho1)*g
            #self.__getir = (1-rho1)*g/(1 - rho1**(epoch*mb + iter))
            self.__getir2 = (1 - rho2)*(g*g)
            #self.__getir2 = (1 - rho2)*g*g/(1 - rho2**(epoch*mb + iter))
            #print(self.__getir2)
        else:
            self.__getir = rho1*self.__getir + (1 - rho1)*g
            #self.__getir = (rho1*self.__getir + (1 - rho1)*g)/(1 - rho1**(epoch*mb + iter))
            self.__getir2=rho2*self.__getir2 + (1 - rho2)*(g*g)
            #self.__getir2 = (rho2*self.__getir2 + (1 - rho2)*g*g)/(1 - rho2**(epoch*mb + iter))
        #bias correction
        self.__getir = self.__getir/(1 - rho1**t)
        self.__getir2 = self.__getir2/(1 - rho2**t)

        return gamma0*(delta + self.__getir/np.sqrt(self.__getir2))
        #den = delta + np.sqrt(self.__getir2)
        #return gamma0*self.__getir*1/den
            

       
              
    #Plain gradient descent and Plain gradient with momentum. If sigma=0, plain gradient. If sigma!=0, plain grad. with momentum.
    def GD(self, x, gamma, sigma,epsilon, max_iters, gammatau, mode, alpha):

        if np.linalg.norm(self.nabla(x)) > epsilon:
            stop = False
        elif np.linalg.norm(self.nabla(x)) <= epsilon:
            print(f"Grad. already smaller than epsilon {self.nabla(x)} <= {epsilon}.\nSet epsilon param. smaller.")
            return 

        #Starts the plain gradient descent
        gamma0=gamma
        iter = 0
        xprev = x
        start_time = time.time()
        while stop == False and iter < max_iters:
            
            #print("X0 = ", x)
            #print(self._feat_matrix_gd)
            #print(gamma)
            #print(self.nabla(x))
            xnext = x - gamma*self.nabla(x) + sigma*(x-xprev)
            #print("XNEXT = " ,xnext)
            xprev = x
            x = xnext
            iter +=1


            if mode!="fix":
                gamma = self.lr_tun(k=iter, gamma0=gamma0, gammatau=gammatau, alpha=alpha,mode=mode)
            
            if np.linalg.norm(self.nabla(x)) <= epsilon:
                stop = True

            #Checks if the are NaN values to stop the loop
            if np.isnan(np.sum(x)):
                stop = True
                print(f"NaN value in iter {iter}")
                self.timeGD = np.NaN
                self.iterGD = iter
                return x
        stop_time = time.time()
        print("Time {} seconds".format((stop_time - start_time)))
    
        if stop:
            print(f"{iter} iterations done with grad. {self.nabla(x)}. ")
            self.iterGD = iter
            self.timeGD = stop_time - start_time
            return x
        elif stop == False and iter==max_iters:
            self.timeGD = 0
            self.iterGD = iter
            print(f"Max number of iterations achieved: {iter} with grad. {self.nabla(x)}")
            return x 

  

  





class LogisticRegr():
    """
    Logistic Regression model.
    Class to perform a Logistic regression.
    It contains the following methods to compute the parameters:
        - GD:  plain gradient descent method with and without momentum.
        - GD_MB: stochastic gradient descent method with and without momentum.
    It also allows to set the following learning rates schedulers:
        -ADAM
        -AdaGrad
        -RMSprop
        -Exponential
        -Linear
        -Inverse
    
    Atributes:
        -method: method to solve OLS
        -max_iter: maximum number of iterations for GD
        -epsilon: stop criterion
        -gamma: initial learning rate
        -sigma: momentum
        -n_epochs: number of epochs
        -mb: number of mb

    Methods:
    User only have to work with 'fit' function. With 'method' it chooses the technique used (direct OLS, GD, SGD) and
    with 'mode' the type of learning rate scheduler. 
    For the user:
        -predict: predict result with a given feature matrix. 
        -fit: function to compute the parameters of the Logistic classification.
            Methods:OLS, GD, GD_MB
            Modes:lr_linear, lr_inv, ADAgrad, RMSprop, ADAM

    For internal operation. This methods are used  for the fit method to work:
        -nabla: function for intern use. It defines the gradient used for the gradient descent techniques.
        -GD_MB: function to compute stochastic gradient descent with a given number of minibatches
        -lr_tun: function used to coordinate the learning rate schedulers.
        -ADAgrad: adagrad method implementation
        -ADAM: adam method implementation
        -RMSprop: rmsprop method implementation
        -GD: function to compute plain gradient descent with a given number of minibatches

        
    """


   

    def __init__(self, method = "GD", beta0=None):
         
        self.method = method
        self.max_iter= 100000
        self.epsilon = 0.000001
        self.gamma=0.01
        self.sigma=0
        self.n_epochs = 1
        self.mb = 1
    

    #Predict function
    def predict(self, feature_matrix):

        p = np.exp(feature_matrix@self.beta)/(1 + np.exp(feature_matrix@self.beta))

        return p


    def fit(self, feature_matrix, y, beta0=None, gamma=None,n_epochs=None, mb=None, max_iters=None, sigma=None, epsilon=None,
            gammatau=None, alpha=1, rho1=None, rho2=None,mode="fix"):

        if type(gamma) == type(None):
            gamma = self.gamma
        if type(gammatau) == type(None):
            gammatau = self.gamma/100
        if type(max_iters) == type(None):
            max_iters = self.max_iter
        if type(epsilon) == type(None):
            epsilon = self.epsilon
        if type(n_epochs)==type(None):
            n_epochs = self.n_epochs
        if type(mb)==type(None):
            mb = self.mb
        if type(sigma)==type(None):
            sigma = self.sigma
        if type(beta0)==type(None):
            beta0 = np.ones(feature_matrix.shape[1])*0.1
        if type(rho1)==type(None):
            rho1 = 0.9
        if type(rho2)==type(None):
            rho2 = 0.99
           

        if self.method=="GD": #Gradient descent method with momentum

            
            #check that the dimensions are right
            if feature_matrix.shape[0] != y.shape[0]:
                print("Feature matrix first dimension (",feature_matrix.shape[0],") and y dimension: ",y.shape[0]," don't match.")
                return 0
            #if everything is right, starts the fit
            elif feature_matrix.shape[0] == y.shape[0]:

                #the following two parameters are used only to define the gradient. Each time we fit
                #with new data, the values of the gradient change.
                self._feat_matrix_gd = feature_matrix
                self._y_gd = y
                #calling of the gradient function
                self.beta = self.GD(x=beta0, gamma=gamma, epsilon=epsilon, max_iters=max_iters, sigma=sigma,
                gammatau=gammatau,mode=mode, alpha=alpha )

        elif self.method=="GD_MB": #Gradient descent with minibatches


            self.beta = self.GD_mb(x=beta0,feat=feature_matrix, y=y,gamma = gamma, mb=mb,n_epochs=n_epochs, sigma=sigma,
            gammatau=gammatau,mode=mode, alpha=alpha,rho1=rho1, rho2=rho2)
                
                #Starts the plain gradient descent

    #Gradient of the Sigmoid function
    def nabla(self, beta):

        feature_matrix = self._feat_matrix_gd
        y = self._y_gd

        p = np.exp(feature_matrix@beta)/(1 + np.exp(feature_matrix@beta))

        return -feature_matrix.T@(y - p)



    def GD_mb(self, x,feat, y, mb, n_epochs, gamma, sigma ,gammatau,mode, alpha, 
            rho1, rho2):

        

        #here we made the reshuffle and minibatches

        data = np.column_stack((feat,y))
        data = shuffle(data)
        data = data.reshape((mb, int(feat.shape[0]/mb), data.shape[1])).transpose(1,2,0)
        gamma0 = gamma

    
        xprev = x

        M = int(feat.shape[0]/mb)#data size in each minibatches

        self.increment = np.zeros((feat.shape[1], n_epochs))
        start_time = time.time()
        for epoch in range(1, n_epochs + 1):
            #learning rate tuning
            
   
            for i in range(mb):
                
                
                #print("X0 = ", x)

                random_index = M*np.random.randint(mb)
                #print("Random index ", random_index)
                #print("M ", M)

                self._feat_matrix_gd = feat[random_index:random_index+M,:]
                self._y_gd = y[random_index:random_index +M]

              

                #we redefine nabla for each minibatche


                if mode != "fix":

                    gamma = self.lr_tun(k=epoch, iter=i,gamma0=gamma0, gammatau=gammatau, alpha=alpha,mode=mode,x=x,
                    rho1=rho1, rho2=rho2,mb=mb)
                   
                
                #THE ALGORITHM
               
                #For ADAM the procedure is a bit different 
                if mode=="ADAM":
                    
                    xnext = x - gamma
                else:
                    
                    xnext = x - gamma*self.nabla(x) + sigma*(x - xprev)
                xprev = x
                x = xnext
                
                
                #print("XNEXT = " ,xnext)
               
                if np.isnan(np.sum(x)):
                    print(f"NaN value in epoch {epoch}")
                    return x
            self.increment[:, epoch-1] = x
            self.iterGD_md = self.nabla(x)
            self.epoch_num = epoch
            #learning rate tuning
     
        stop_time = time.time()
        print("Time {} seconds".format((stop_time - start_time)))
        print(f"{epoch} epochs done with grad. {self.nabla(x)}.")
        self.time_GDMB = stop_time - start_time
        return x



        

    #Function to tune the learning rate. With mode, the tuning method is choosed
    def lr_tun(self, k, gamma0, gammatau,mode, iter,alpha, x, rho1,rho2,mb,delta = 1e-8):
        if mode == "lr_exp":
            k -=1
            return gamma0*np.exp(-k*gammatau)
        elif mode =="lr_linear":
            k-=1
            alpha = k/100
            return (1 - alpha)*gamma0 + alpha*gammatau
        elif mode =="lr_inv":
            k-=1
            return gamma0/(k*2+iter + gamma0*5)
        elif mode =="ADAgrad":

            return self.ADAgrad(x=x, iter=iter, delta=delta, gamma0=gamma0)
        elif mode=="RMSprop":
            return self.RMSprop(x=x, iter=iter, delta=delta, gamma0=gamma0, rho1=rho1 )
        elif mode=="ADAM":
            return self.ADAM(x=x, iter=iter, delta=delta, gamma0=gamma0, rho1=rho1, rho2=rho2,
                epoch = k, mb=mb )
            
    ## ADAgrad method
    def ADAgrad(self, x, iter, gamma0, delta ):

        
        g = self.nabla(x)
        
        
        if iter == 0:
            self.__getir = g*g
           
        else:
            self.__getir += g*g

        
        den = delta + np.sqrt(self.__getir)
        return gamma0*1/den

    ## RMSprop method
    def RMSprop(self, x, iter, gamma0, delta, rho1 ):

        
        g = self.nabla(x)
        
        
        if iter == 0:
            self.__getir = (1-rho1)*g*g
        else:
            self.__getir +=  rho1*self.__getir + (1 - rho1)*g*g

        
        den = delta + np.sqrt(self.__getir)
        return gamma0*1/den

    ## ADAM method
    def ADAM(self, x, iter, gamma0, rho1, rho2, delta, epoch, mb):

        g = self.nabla(x)

        
        epoch -=1
        iter += 1
        
        t = epoch*mb + iter

      
        if iter==1:
            #print((1 - rho1**(epoch*mb + iter)))
            self.__getir = (1 - rho1)*g
            #self.__getir = (1-rho1)*g/(1 - rho1**(epoch*mb + iter))
            self.__getir2 = (1 - rho2)*(g*g)
            #self.__getir2 = (1 - rho2)*g*g/(1 - rho2**(epoch*mb + iter))
            #print(self.__getir2)
        else:
            self.__getir = rho1*self.__getir + (1 - rho1)*g
            #self.__getir = (rho1*self.__getir + (1 - rho1)*g)/(1 - rho1**(epoch*mb + iter))
            self.__getir2=rho2*self.__getir2 + (1 - rho2)*(g*g)
            #self.__getir2 = (rho2*self.__getir2 + (1 - rho2)*g*g)/(1 - rho2**(epoch*mb + iter))
        #bias correction
        self.__getir = self.__getir/(1 - rho1**t)
        self.__getir2 = self.__getir2/(1 - rho2**t)

        return gamma0*(delta + self.__getir/np.sqrt(self.__getir2))
        #den = delta + np.sqrt(self.__getir2)
        #return gamma0*self.__getir*1/den

        #Plain gradient descent and Plain gradient with momentum. If sigma=0, plain gradient. If sigma!=0, plain grad. with momentum.
    def GD(self, x, gamma, sigma,epsilon, max_iters, gammatau, mode, alpha):

        if np.linalg.norm(self.nabla(x)) > epsilon:
            stop = False
        elif np.linalg.norm(self.nabla(x)) <= epsilon:
            print(f"Grad. already smaller than epsilon {self.nabla(x)} <= {epsilon}.\nSet epsilon param. smaller.")
            return 
        else:
            stop = True

        #Starts the plain gradient descent
        gamma0=gamma
        iter = 0
        xprev = x
        start_time = time.time()

        while stop == False and iter < max_iters:
            
            #print("X0 = ", x)
            #print(self._feat_matrix_gd)
            #print(gamma)
            #print(self.nabla(x))
            xnext = x - gamma*self.nabla(x) + sigma*(x-xprev)
            #print("XNEXT = " ,xnext)
            xprev = x
            x = xnext
            iter +=1


            if mode!="fix":
                gamma = self.lr_tun(k=iter, gamma0=gamma0, gammatau=gammatau, alpha=alpha,mode=mode)
            
            if np.linalg.norm(self.nabla(x)) <= epsilon:
                stop = True

            #Checks if the are NaN values to stop the loop
            if np.isnan(np.sum(x)):
                stop = True
                print(f"NaN value in iter {iter}")
                self.timeGD = np.NaN
                self.iterGD = iter
                return x
        stop_time = time.time()
        print("Time {} seconds".format((stop_time - start_time)))
    
        if stop:
            print(f"{iter} iterations done with grad. {self.nabla(x)}. ")
            self.iterGD = iter
            self.timeGD = stop_time - start_time
            return x
        elif stop == False and iter==max_iters:
            self.timeGD = 0
            self.iterGD = iter
            print(f"Max number of iterations achieved: {iter} with grad. {self.nabla(x)}")
            return x 