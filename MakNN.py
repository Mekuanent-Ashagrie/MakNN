#!/usr/bin/env python
# coding: utf-8
# version 1.0
# Developed by Mekuanent Birara

import numpy as np
import pickle


class MakNN:

    
    def __init__(self):

        #initialize global variables 
        self.loss_function, self.optimizer, self.check_point = '', '' , ''
        self.beta_opt_momentum, self.beta_opt_rms = 0.9, 0.999  #most common values but can be changed by the user 
        self.iteration, self.batch_size, self.decay_rate = 1, 1, 0
        self.l2_lambda, self.learning_rate, self.learning_rate_init = 0, 0, 0
        self.epselon = pow(10,-8) 
        self.weight, self.bias, self.A, self.Z, self.dW, self.dB, self.dZ, self.dA, self.activation_functions, self.layers, self.drop_outs, self.batch_norm = [],[],[],[],[],[],[],[],[],[],[],[]
        self.training_loss, self.validation_loss, self.training_accuracy, self.validation_accuracy = [], [], [], []
        self.vdW, self.vdB, self.sdW, self.sdB, self.vdW_corr, self.vdB_corr, self.sdW_corr, self.sdB_corr = [],[],[],[],[],[],[],[]
        self.train_iteration_counter = 0
        self.batch_norm_gamma, self.batch_norm_beta = 0, 1
        self.number_of_batch = 0
        self.early_stoping_param, self.early_stoping_flag = 0, 0
        

    def normalize_input(self, x, norm_type = 'mean'):   
        
        if(norm_type=='mean'):
            return x-np.mean(x, axis = 0)
        else:        
            return x-np.std(x, axis = 0)


    def encode_y(self, y):
        
        y = np.array(y)
        #take max value++1 as the number of classes in y
        enc_y = np.zeros((len(y), self.layers[-1]))        
        for j in range(0, len(y)):
            for k in range(0, self.layers[-1]):
                if(y[j] == k): enc_y[j][k] = 1

        return np.transpose(enc_y)


    def split_data(self, x, y, train = 60, validation = 20, test = 20):
        
        rand = np.arange(len(x))
        np.random.shuffle(rand)
        x = x[rand]
        y = y[rand]    
        x_train = x[:int(len(x)*(train/100)),:]    
        x_val = x[int(len(x)*train/100):int(len(x)*(train+validation)/100),:] 
        x_test = x[int(len(x)*(train+validation)/100):int(len(x)*(train+validation+test)/100),:] 
        y_train = y[:int(len(y)*(train/100))]
        y_val = y[int(len(y)*train/100):int(len(y)*(train+validation)/100)]
        y_test = y[int(len(y)*(train+validation)/100):]
        
        return x_train, y_train, x_val, y_val, x_test, y_test


    def weight_init_variance(self, activation, hidden_units):

        if(activation == 'relu'):
            return np.sqrt(2/hidden_units)
        else: return np.sqrt(1/hidden_units)   


    def init_model(self, layer, activations, input_shape, l2_reg_param = 0, drop_out_param = [], batch_norm = [], weight_init = False):   

        #input_shape contains features and batch_size)        
        self.layers = np.append(np.array([input_shape[0]]), np.array(layer)) #merge input layer with hidden layers 
        self.activation_functions = np.array(activations) 
        self.drop_outs = drop_out_param
        self.l2_lambda = l2_reg_param
        self.batch_size = input_shape[1]
        self.batch_norm = batch_norm
        
        #initialize weights here 
        for i in range(1, len(self.layers)):    
            
            bl = np.zeros((self.layers[i], 1)) #proper matrix structure, no need to transpose during the forward pass           

            if(weight_init==False):wl = np.random.randn(self.layers[i],self.layers[i-1])*0.01 #no need to transpose during the forward pass
            else: wl = np.random.randn(self.layers[i],self.layers[i-1])*self.weight_init_variance(self.activation_functions[i-1], self.layers[i-1])
            al = np.zeros((self.layers[i], self.batch_size)) 
            zl = np.zeros((self.layers[i], self.batch_size))    
            
            #variables to hold parameters 
            self.weight.append(wl)
            self.bias.append(bl)    
            self.A.append(al)
            self.Z.append(zl)
        
            #variables to hold derivatives 
            self.dW.append(wl)
            self.dB.append(bl)
            self.dZ.append(zl)
            self.dA.append(al)

            #variables to hold weighted average values
            self.vdW.append(wl*0)
            self.vdB.append(bl*0) 
            self.sdW.append(wl*0)
            self.sdB.append(bl*0)

            #correlation variables for adam optimization 
            self.vdW_corr.append(wl*0)
            self.vdB_corr.append(bl*0) 
            self.sdW_corr.append(wl*0)
            self.sdB_corr.append(bl*0)

        #print model structure 
        print("Model Structure")
        print("_______________________________________")
        print("\tLayer\t","Nodes\t", "Param[W & B]") #try to add other trainable parameters specially those in batch normalization and others 
        print("_______________________________________")
        
        param = 0
        #input layer 
        print("    ", str(0),"[input]\t",self.layers[0],"\t", str(0))
        print("_______________________________________")

        for layer in range(1, len(self.layers)):            
            layer_param = (self.weight[layer-1].shape[0]*self.weight[layer-1].shape[1]) + (self.bias[layer-1].shape[0])
            param += layer_param
            print("\t",layer ,"\t",self.layers[layer],"\t", layer_param)
            print("_______________________________________")
        print("\nTotal Trainable Parameters: ", param)
        
        return 


    def activation(self, act, z):

        if(act=='sigmoid'): return 1/(1 + np.exp(-z))
        elif(act=='softmax'): return np.exp(z)/np.sum(np.exp(z), axis=0) 
        elif(act=='tanh'): return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))
        elif(act=='relu'): 
            z[z<=0]=0 
            return z
        elif(act=='leaky_relu'): 
            z[z<=0]=0.000001*z[z<=0]      
            return z 

    
    def derivative_activation(self, z, act):

        #derivations of activation functions sigmoid, softmax, relu, tanh and leakyrelu
        if(act=='sigmoid'): return (1/(1+np.exp(-z)))*(1-(1/(1+np.exp(-z))))  
        elif(act=='softmax'): return (np.exp(z)/np.sum(np.exp(z), axis=0))*(1-(np.exp(z)/np.sum(np.exp(z), axis=0)))  
        elif(act=='tanh'): return 1-(np.tanh(z)**2) 
        elif(act=='relu'):
            z[z>0]=1
            z[z<0]=0
            z[z==0]=0.00000000000001 
            return z
        elif(act=='leaky_relu'): 
            z[z>0]=1
            z[z<0]=0.01 #can even be smaller than this number 
            z[z==0]=0.00000000000001 
            return z


    def batch_normalization(self, z):

        mean = np.mean(z) 
        std = np.std(z)
        z = self.batch_norm_beta*((z-mean)/(np.sqrt(std + self.epselon))) + self.batch_norm_gamma 
        
        return z


    def frobenius_norm(self, x):

        #used to sum the square of elements for L2 regularization 
        fn = 0
        for i in range(len(x)):  #X is a list containing variable dimention numpy arrays
            fn += np.sum(x[i]**2)

        return fn
        

    def dropout_regularization(self, al, val):

        random_index = np.random.permutation(len(al))
        zero_index = random_index[:round(len(al)*val)]
        al[zero_index] = 0
        #al = al/val #scalling al so that the expected value of Z will not be highly reduced [this is optional]

        return al

    
    def loss(self, a, Y): 

        #loss functions with their derivatives 
        da_c = []
        loss = 0
        
        if(self.loss_function == 'mean_square_loss'):
            #for all
            for i in range (0, len(a)):
                loss += (np.sum((Y[i] - a[i])**2))/len(Y[i]) 
                da_c.append((0.5)*(a[i]-Y[i]))
            #L2 norm if lambda hyperparameter is set [this result in big loss]
            if(self.l2_lambda > 0 ): 
                loss = loss + (self.l2_lambda/(2*len(a)))*self.frobenius_norm(self.weight) + (self.l2_lambda/(2*len(a)))*self.frobenius_norm(self.bias)          
            return loss, da_c
        elif(self.loss_function == 'logarithmic_loss'):
            for i in range (0, len(a)):
                loss += (np.sum(-(Y[i]*np.log(a[i] + self.epselon) + (1-Y[i])*np.log((1-a[i])+self.epselon))))/len(Y[i]) # ++epselon is to avoid log0
                da_c.append(-(Y[i]/(a[i]+self.epselon))+((1-Y[i])/((1-a[i])+self.epselon)))    
            #L2 norm if lambda hyperparameter is set
            if(self.l2_lambda > 0 ): 
                loss = loss + (self.l2_lambda/(2*len(a)))*self.frobenius_norm(self.weight) + (self.l2_lambda/(2*len(a)))*self.frobenius_norm(self.bias)
            return loss, da_c


    def accuracy(self, a, Y):

        #this function computes accuracy of a model dividing true classified examples with all the examples  

        return np.mean(np.round(a) == Y)


    def forward_prop(self, x_train):

        #forward prop including the cost [No need to transpose weight and bias]
        for i in range(0, len(self.layers)-1):        
            if(i==0): zl = self.weight[i].dot(np.transpose(x_train)) + self.bias[i] #this is where [a](layer) = X_train             
            else: zl = self.weight[i].dot(self.A[i-1]) + self.bias[i]     
            #apply batch norm here if specified by the user 
            if(i < len(self.batch_norm)):
                if(self.batch_norm[i] == True): #batch norm is defined either as true or false 
                    zl = self.batch_normalization(zl) #batch_beta, batch_gamma are set to be 1 and zero and further let it be learnable 
                
            al = self.activation(self.activation_functions[i], zl)  
            #apply dropout regularization here if specified by the user 
            if((i <= len(self.drop_outs)-1) and (self.drop_outs[i]!=0)): #ignore if dropout == 0
                al = self.dropout_regularization(al, self.drop_outs[i]) #as the keep_prob is 1-activations to drop 

            self.Z[i] = zl
            self.A[i] = al
        
        return 


    def predict(self, x):

        PA, PZ = [],[]
        for i in range(1, len(self.layers)):                 
            pal = np.zeros((self.layers[i], len(x))) 
            pzl = np.zeros((self.layers[i], len(x)))  
            PA.append(pal)
            PZ.append(pzl)

        #perform a single forward pass which is prediction 
        for i in range(0, len(self.layers)-1):        
            if(i==0): pzl = self.weight[i].dot(np.transpose(x)) + self.bias[i]              
            else: pzl = self.weight[i].dot(PA[i-1]) + self.bias[i]     
            
            #check type of normalization to apply here @@@@@@@@@@@@@@@@@
            if(i <= len(self.batch_norm)):
                pzl = self.batch_normalization(pzl) 
                
            pal = self.activation(self.activation_functions[i], pzl)  
             
            PZ[i] = pzl
            PA[i] = pal

        return PA[-1] #return the last activation layer value  


    def back_prop(self, da_c, x_train):
       
        i = len(self.layers)-2            
        self.dA[i] = np.array(da_c)        
        while(i>=0): 
            #dZ   
            self.dZ[i] = self.dA[i]*self.derivative_activation(self.Z[i], self.activation_functions[i])  

            #dW 
            #apply L2 regularization if specified by the user [adding this value creates a weight decay during update]
            if(self.l2_lambda != 0):
                if(i == 0): self.dW[i] = (1/self.batch_size)*(self.dZ[i].dot(x_train)) + ((self.l2_lambda/self.batch_size)*self.weight[i])
                else: self.dW[i] = (1/self.batch_size)*(self.dZ[i].dot(np.transpose(self.A[i-1]))) + ((self.l2_lambda/self.batch_size)*self.weight[i])
            else:
                if(i == 0): self.dW[i] = (1/self.batch_size)*(self.dZ[i].dot(x_train))
                else: self.dW[i] = (1/self.batch_size)*(self.dZ[i].dot(np.transpose(self.A[i-1])))
             
            #dB  
            # check if L2 regularization is specified [adding this value creates a bias decay during update]
            if(self.l2_lambda != 0):
                self.dB[i] = (1/self.batch_size)*(np.sum(self.dZ[i], axis = 1, keepdims = True)) +  ((self.l2_lambda/self.batch_size)*self.bias[i])
            else:
                self.dB[i] = (1/self.batch_size)*(np.sum(self.dZ[i], axis = 1, keepdims = True))

            #dA 
            self.dA[i-1] = np.transpose(self.weight[i]).dot(self.dZ[i])
                
            i-=1        
        
        return 


    def exp_weighted_average(self):

        #compute weighted average function for both momentum and rms
        for i in range(0, len(self.layers)-1): 
            self.vdW[i] = self.beta_opt_momentum*self.vdW[i] + (1-self.beta_opt_momentum)*self.dW[i]
            self.vdB[i] = self.beta_opt_momentum*self.vdB[i] + (1-self.beta_opt_momentum)*self.dB[i]
            self.sdW[i] = self.beta_opt_rms*self.sdW[i] + (1-self.beta_opt_rms)*(self.dW[i]**2)
            self.sdB[i] = self.beta_opt_rms*self.sdB[i] + (1-self.beta_opt_rms)*(self.dB[i]**2)
            
        return 


    def learning_decay(self):

        #change the learning rate depending on the decay variable provided         
        self.learning_rate = (1/(1+(self.decay_rate*self.train_iteration_counter)))*self.learning_rate_init

        return 


    def update_parameters(self):
        
        #check if there is an optimizer 
        if(self.optimizer == ''):
            for i in range(0, len(self.layers)-1):            
                self.weight[i] = self.weight[i] - (self.learning_rate*(self.dW[i]))
                self.bias[i] = self.bias[i] - (self.learning_rate*(self.dB[i]))
        elif(self.optimizer == 'momentum'):   
            for i in range(0, len(self.layers)-1):            
                self.weight[i] = self.weight[i] - (self.learning_rate*(self.vdW[i]))
                self.bias[i] = self.bias[i] - (self.learning_rate*(self.vdB[i]))
        elif(self.optimizer == 'rms'):
            for i in range(0, len(self.layers)-1):                                  
                self.weight[i] = self.weight[i] - (self.learning_rate*(self.dW[i]/(np.sqrt(self.sdW[i]) + self.epselon)))
                self.bias[i] = self.bias[i] - (self.learning_rate*(self.dB[i]/(np.sqrt(self.sdB[i]) + self.epselon)))                               
        elif(self.optimizer == 'adam'):
            for i in range(0, len(self.layers)-1):    
                self.vdW_corr[i] = self.vdW[i]/(1-(self.beta_opt_momentum**self.train_iteration_counter))  
                self.vdB_corr[i] = self.vdB[i]/(1-(self.beta_opt_momentum**self.train_iteration_counter))  
                self.sdW_corr[i] = self.sdW[i]/(1-(self.beta_opt_rms**self.train_iteration_counter))  
                self.sdB_corr[i] = self.sdB[i]/(1-(self.beta_opt_rms**self.train_iteration_counter))      

                self.weight[i] = self.weight[i] - (self.learning_rate*(self.vdW_corr[i]/np.sqrt(self.sdW_corr[i] + self.epselon)))
                self.bias[i] = self.bias[i] - (self.learning_rate*(self.vdB_corr[i]/np.sqrt(self.sdB_corr[i] + self.epselon)))

        return 


    def model_check_point(self):

        #a function to save parameters with good meric value         
        if(self.train_iteration_counter > 0):
            if(self.check_point == 'train_acc'):
                if(self.training_accuracy[-1]>max(self.training_accuracy[:len(self.training_accuracy)-1])):
                    self.save_model()
                    print("------Training accuracy improved: Model saved!------\n")
                    
            elif(self.check_point == 'train_loss'):
                if(self.training_loss[-1]<min(self.training_loss[:len(self.training_loss)-1])):
                    self.save_model()
                    print("------Training loss improved: Model saved!------\n")
                    
            elif(self.check_point == 'val_acc'):
                if(self.validation_accuracy[-1]>max(self.validation_accuracy[:len(self.validation_accuracy)-1])):
                    self.save_model()
                    print("-------Validation accuracy improved: Model saved!------\n")
                    
            elif(self.check_point == 'val_loss'):
                if(self.validation_loss[-1]<min(self.validation_loss[:len(self.validation_loss)-1])):
                    self.save_model()
                    print("------Validation loss improved: Model saved!------\n")         

        return 


    def save_model(self, file_name='saved_model.pkl'):

        #function to save a model at a given instance 
        model = []
        #structure the list here 
        model.append(self.layers) #layers
        model.append(self.batch_norm) #batch normalization 
        model.append(self.activation_functions) #activations at each layer 
        model.append(self.weight)
        model.append(self.bias)
        #save the structure 
        with open(file_name, 'wb') as f:
            pickle.dump(model, f)  

        return 


    def load_model(self, file_name='saved_model.pkl'): 

        #load model 
        model = []
        with open(file_name, 'rb') as f:
            model = pickle.load(f)
        #set variables  
        self.layers = model[0]
        self.batch_norm = model[1]
        self.activation_functions = model[2]
        self.weight = model[3]
        self.bias = model[4]

        return

    
    def model_early_stopping(self):

        #compute val loss change per iteration 
        if(self.train_iteration_counter > self.number_of_batch):            
            if(max(self.validation_loss[len(self.validation_loss)-self.number_of_batch:-1]) > max(self.validation_loss[len(self.validation_loss)-2*self.number_of_batch:len(self.validation_loss)-self.number_of_batch])):
                self.early_stoping_flag += 1
            else: self.early_stoping_flag = 0

        return


    def train_model(self, x_train, y_train, x_val, y_val, check_point = '', early_stoping_steps = 0, optimizer ='adam', loss='logarithmic_loss', learning_rate = 0.01, epoch = 1, beta_momentum = 0.9, beta_rms = 0.999, decay_rate = 0, batch_beta = 1, batch_gamma = 0, verbose = True):
        
        self.optimizer = optimizer
        self.check_point = check_point
        self.early_stoping_param = early_stoping_steps #itrations to wait until stoping 
        self.learning_rate = learning_rate #to be modified depending on the decay rate defined 
        self.learning_rate_init = learning_rate 
        self.beta_opt_momentum = beta_momentum
        self.beta_opt_rms = beta_rms 
        self.decay_rate = decay_rate
        self.loss_function = loss
        self.iteration = epoch   
        self.batch_norm_gamma = batch_gamma
        self.batch_norm_beta = batch_beta
        self.number_of_batch = int(x_train.shape[0]/self.batch_size)
        
        print("Model starts training . . . . . . . . .\n")

        #encode y_val if the last node contains multiple units 
        if(self.layers[-1] > 1):
            y_val = self.encode_y(y_val)

        for i in range(0, self.iteration):                   
            for j in range(0,  self.number_of_batch): 
                print("Training on epoch: ",i+1,"  batch: ", j+1)   
                #jump back prop when learning starts[this is to make one more forward prop after the last bak prop]
                if((i!=0) or (j != 0)):       
                    self.train_iteration_counter += 1 #required for the adam weight update operation as well as learning rate decay             
                    #perform back propagation here            
                    self.back_prop(train_da_c, batch_x)                         
                    #compute weighted average if optimizer is defined 
                    if(self.optimizer != ''): 
                        self.exp_weighted_average()                        
                    #update weights
                    self.update_parameters()      

                    #modify learning rate [if the decay rate == 0 or not set by user the learning rate stays the same] 
                    self.learning_decay() 
                                  
                #split the n-th batch [observations within n-th batch]
                batch_x = x_train[(j*self.batch_size):(j*self.batch_size + self.batch_size), :]    
                batch_y = y_train[(j*self.batch_size):(j*self.batch_size + self.batch_size)]  

                #encode batch_y if the last node contains multiple units 
                if(self.layers[-1] > 1):
                    batch_y = self.encode_y(batch_y)                    
                    
                #reshape batch_y to 2D if not in 2D
                if(batch_y.ndim == 1):
                    batch_y = np.reshape(batch_y, (-1, len(batch_y)))    
                if(y_val.ndim == 1):
                    y_val = np.reshape(y_val, (-1, len(y_val)))

                #start forward pass here 
                self.forward_prop(batch_x)   

                #calculate cost here [return the derivation of cost function as well]    
                train_loss, train_da_c = self.loss(self.A[-1], batch_y)  
                #train accuracy                
                train_acc = self.accuracy(self.A[-1], batch_y) 

                #perform prediction for the validation set and calculate the loss 
                prediction = self.predict(x_val)
                #val loss

                val_loss, val_da_c = self.loss(prediction, y_val)  #batch size is size of the validation set and 
                #val accuracy
                val_acc = self.accuracy(prediction, y_val)

                #update result lists so that further can be used to plot results 
                self.training_loss.append(train_loss)
                self.validation_loss.append(val_loss)
                self.training_accuracy.append(train_acc) 
                self.validation_accuracy.append(val_acc)
                
                #print results
                if(verbose == True): print("Train_loss:", str(round(train_loss, 4)), "\tTrain_Acc:", str(round(train_acc, 4)), "\tVal_loss:", str(round(val_loss, 4)),"\tVal_Acc:", str(round(val_acc,4)), "\n")
                #check for check points
                if(self.check_point != ''):
                    self.model_check_point()

            #perform early stoping if specified [this is done per iteration]
            if(self.early_stoping_param > 0):
                self.model_early_stopping()
                if(self.early_stoping_flag == self.early_stoping_param): break

        #report end of training 
        print("Training Ends . . . . . . . . .")
                   
        return 