#first call the data maker function
#then call the model selector (optional) model selector selects the optimal model.
#call the model.model_train() function
#use model.generate(list of parameters) to generate 

import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

batch_size = 32      #no of training batches (since it's slow to train at all the data at once)
max_iters = 5000        #max no of training iterations
n_inlayer_out = 20         #no of neruons of the first layer
n_hidden=50                 #no of neurons of hidden layer
n_hidden2=100
UPLOAD_FOLDER = '/tmp/uploads'

def set_seed(seed=42):                          #doing this so that the same random numbers are picked
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#set_seed()                            #call this function if you do not want random initilization (call it before calling data_maker and initialising model)

def data_maker(parameters_columns: list, targets_columns: list, csv_path):          #function that takes data and prepares it for the model
    data = pd.read_csv(csv_path)
    scalerx = StandardScaler()            #standard scalar object to calculate z score. So that all numeric parameters range from -1 to 1                  
    scalery = StandardScaler()
    categorical_columns = []                #to separate numerical and categorical columns
    numerical_columns = []
    encoding_dicts = []                     #to encode the categorical columns into numbers first (before using embedding table)
    targets_columns_names = []                  
    
    data = data.iloc[:, parameters_columns + targets_columns]  # make dataframe of only parameters and targets columns
    
    new_parameters_columns = list(range(len(parameters_columns)))       #since the indices of columns change when we shorten the dataframe by removing unrequired columns
    new_targets_columns = list(range(len(parameters_columns), len(targets_columns) + len(parameters_columns))) #same thing
    
    data.replace("", np.nan, inplace=True)   # removing data with any missing information 
    data.dropna(inplace=True)
    
    for idx in new_parameters_columns:          #separating the columns into numeric and categorical
        col_name = data.columns[idx]
        if data[col_name].dtype == 'object':
            categorical_columns.append(col_name)
        else:
            numerical_columns.append(col_name)
        
    for idx in new_targets_columns:                 #checking if categorical columns have strings (since this is regression and not classification)
        col_name = data.columns[idx]            
        targets_columns_names.append(col_name)
        if data[col_name].dtype == 'object':
            raise ValueError("Error: Don't put strings in targets.")
    
    for col_name in numerical_columns + targets_columns_names:      #removing outliers from the numeric columns. Any value greater than or less than 1.5 * interquartile range
        Q1 = data[col_name].quantile(0.25)                          #is an outlier. so remove those rows
        Q3 = data[col_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[col_name] >= lower_bound) & (data[col_name] <= upper_bound)]
    
    Xnum = data[numerical_columns].values                   #numeric columns after removing outliers
    Xcat = data[categorical_columns].values                 #categorical columns after removing outliers
    y = data[targets_columns_names].values                  #target columns after removing outliers
    
    if Xnum.size!=0:                                        #calculating the z score for x and y to make them range from -1 to 1. This is done so that the model trains better.
        Xnum = scalerx.fit_transform(Xnum)                  #(for each row in each column x=(x-mean)/std)
    y = scalery.fit_transform(y)
    
    for col_idx in range(Xcat.shape[1]):                    #for each categorical columns
        col_name = categorical_columns[col_idx]
        col_data = Xcat[:, col_idx]

        col_data = np.char.lower(col_data.astype(str))      #to lower for encoding

        unique_values = np.unique(col_data)                   #store all the unique categories of a column
        mapping = {val: idx for idx, val in enumerate(unique_values)}  #encode each category with a number and store it in a dict

        encoding_dicts.append(mapping)                              #append that dict to a list that contains all the dictionaries. this will be requires later for encoding further data and decoding as well 
        Xcat[:, col_idx] = np.vectorize(mapping.get)(col_data)        #encode the strings with the dict to a number
    
    return torch.tensor(Xnum.astype(float), dtype=torch.float32), torch.tensor(Xcat.astype(float), dtype=torch.float32), torch.tensor(y.astype(float), dtype=torch.float32), scalerx, scalery, encoding_dicts

    #return the values of all the colums as tensors of float type (to feed into the pytorch nueral network). Also returning scalerx, scalery, encoding_dicts because they are used later for 
    #further scaling and encoding of new data and then unscaling and decoding as well



class Swish(nn.Module):                         #swish activation (for better non linearity) x*sigmoid=swish ranges from neg to pos infinity
    def forward(self, x):
        return x * torch.sigmoid(x)


class MLP(nn.Module):
    def __init__(self, xnum, xcat, y, scalerx, scalery, encoding_dicts,strength=0):
        
        self.xnum_train = xnum[:int(0.8 * xnum.shape[0])]           #sperating the training and testing splits
        self.xcat_train = xcat[:int(0.8 * xcat.shape[0])]
        self.y_train = y[:int(0.8 * y.shape[0])]
        self.xnum_test = xnum[int(0.8 * xnum.shape[0]):]
        self.xcat_test = xcat[int(0.8 * xcat.shape[0]):]
        self.y_test = y[int(0.8 * y.shape[0]):]
        self.strength=strength                                          #strenth of the model 
        self.encoding_dicts = encoding_dicts                               
        self.scalerx = scalerx
        self.scalery = scalery
        n_inlayer_in=xnum.shape[1]+xcat.shape[1]                        #no of inputs to the first layer = no of categorical columns + no of numerical columns

        super(MLP, self).__init__()                             #call the init of the parent nn.Module
        
        self.embeddings = nn.ModuleList()                       #list of modules. (similar to python list but contains layers of neurons)
        
        for dict in encoding_dicts:                                 #for each encoding dicts (ie for each no of categorical columns)
            self.embeddings.append(nn.Embedding(len(dict), 1))      #make an embedding table of size equal to number of categories in a column
                                                                    #embedding table is learnable (its value is changed by optimizer)
                                                                    #ie the length of the dictionary
                                                                    
                                                                    
        self.fc1 = nn.Linear(n_inlayer_in, n_inlayer_out)            #first layer
        #self.batchnorm = nn.BatchNorm1d(n_inlayer_out)               #normalize the outputs of the first layer (same as standard scalar above)
        self.act = Swish()                                         #this is done to remove the exploding and vanishing gradients problem
                                                 
        self.finalact=nn.ReLU()
        
        if strength==0:                                     #using only two layers and no residual connections if strength == 0
            self.fc2 = nn.Linear(n_inlayer_out, y.shape[1])

        
        if strength==1:                                                      #using more layer with residual connectoins and dropouts
            self.proj1=nn.Linear(n_inlayer_out,n_inlayer_in)                #this is used to transform the output of hidden layer into the dimension of input to add the input again (for residual connecions)
            
            self.seq1 = nn.Sequential(nn.Linear(n_inlayer_in, n_hidden),        #sequential layers where data flows from left to right. dropout(0.4) shuts off 40% of random neurons during 
                                      #nn.BatchNorm1d(n_hidden),
                                      Swish(),                              #propagaation so the neural net trains on a subset of neruons (this helps reduce overfitting)  
                                      nn.Linear(n_hidden,n_inlayer_in),
                                      nn.Dropout(0.1))
            
            self.fc2 = nn.Linear(n_inlayer_in, y.shape[1])            #final layer
            
            
        if strength == 2:
            self.proj1 = nn.Linear(n_inlayer_in, n_inlayer_out)  # Project the input to shape of seq1
            self.proj2 = nn.Linear(n_inlayer_out, n_hidden)      
            self.proj3 = nn.Linear(n_hidden, n_hidden2)         
            
          #  self.finalproj=nn.Linear(n_inlayer_in,y.shape[1])
            self.finalproj=nn.Linear(y.shape[1],y.shape[1])
          
          
            
            self.seq1 = nn.Sequential(
                nn.Linear(n_inlayer_out, n_hidden),
                #nn.BatchNorm1d(n_hidden),
                Swish(),
                nn.Dropout(0.1)
            )
            
            self.seq2 = nn.Sequential(
                nn.Linear(n_hidden, n_hidden2),
               # nn.BatchNorm1d(n_hidden2),
                Swish(),
                nn.Dropout(0.1)
            )
            
            self.fc2 = nn.Linear(n_hidden2, y.shape[1])
    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)


            
       
        
    def forward(self, xnum, xcat):
        embedded_columns = []                       #list of columns after embedding
        for i in range(xcat.shape[1]):              #for each categorical column
            column = xcat[:, i]                     #extract all the data
            embedded_column = self.embeddings[i](column.clone().detach().int()) #cloning the column so this does not affect the backward pass (recommended by pytorch) and applying embeddings 
            embedded_columns.append(embedded_column)  #append the column 
        
        if xcat.size(1) > 0:
            xcat_emb = torch.cat(embedded_columns, dim=1) #catenate all those columns in dimension 1 
        
        x = torch.cat([xnum, xcat_emb] if xcat.size(1) > 0 else [xnum], dim=1).float()    #catenate numerical and embedded columns in dimenstion 1
        
        
        if self.strength==0:                #forward pass for simpler mode
            x1=self.fc1(x)
            #x1_norm=self.batchnorm(x1)
            x1_act=self.act(x1)
            x2=self.fc2(x1_act)
            return x2
            
            
            
        
        if self.strength==1:                #forward pass for complex mode with residual connection
            x1 = self.fc1(x)
           # x1norm = self.batchnorm(x1)
            x1_act=self.act(x1) 
            x1_res=self.proj1(x1_act)+x #residual connection 1 (this is required because when neural network is deep, the data is converted to noise so the data is 
                                        #passed again after each block
            
            x2=self.seq1(x1_res)
            x2_res=x1_res+x2          #residual connection 2
            
            x3 = self.fc2(x2_res)  
            return x3
    
        if self.strength == 2:
            x1 = self.fc1(x)
            #x1norm = self.batchnorm(x1)
            x1_act = self.act(x1)
            x1_res = x1_act + self.proj1(x)  # Residual connection with projection
            
            x2 = self.seq1(x1_res)
            x2_res = self.proj2(x1_res) + x2  # Residual connection 2
            
            x3 = self.seq2(x2_res)
            x3_res = x3 + self.proj3(x2_res)  # Residual connection 3
            
            x4 = self.fc2(x3_res)
            
            x4_res=x4 + self.finalproj(x4)
            
            return x4_res
    
    
    
    def train_model(self, epochs=max_iters, patience=100, min_delta=1e-10,lr=0.01,second_lr=0.00001):
        
        self.train()
        optimizer = optim.Adam(self.parameters(), lr,weight_decay=0.01)            #optimizer to change the parameters. lr is learning rate. weight decay=0.1 limits the parameters from 
                                                                                        #becoming too large and cause overfitting. this is a form of l2 regularization
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience // 4, factor=0.5, min_lr=second_lr)
        #this is used to decrease the learning rate if the model starts plateuing (no change in loss) 

        criterion = nn.MSELoss()  #mean squared error for loss calculation
        best_val_loss = float('inf')    #for early stopping (to not let the model overfit)
        epochs_no_improve = 0
        training_losses = []

        for epoch in range(epochs):
            self.train()                                                                    #set the model in training mode
            batch = torch.randint(0, self.xnum_train.shape[0], (batch_size,))               #generate random integers of batch size
            xnum=self.xnum_train[batch]                                                     #only select the rows from the random numbers
            xcat=self.xcat_train[batch]
            y=self.y_train[batch]

            print(xnum.shape)
            optimizer.zero_grad()                                                           #set all the gradients to zero (because the gradients sum up ie grad=grad+new grad)
            outputs = self(xnum,xcat)                                                             #outputs of the neural net
            loss = criterion(outputs, y)                                                        #caluclate loss
            loss.backward()                                                                 #backpropagate 
            optimizer.step()                                                                #change the parameters one step
            epoch_loss = loss.item()      

            
            print(f"Epoch {epoch+1}, Training Loss: {epoch_loss}")
            
            # Validation loss
            val_loss = self.validation_loss(self.xnum_test, self.xcat_test, self.y_test)            #check validatoin loss
            scheduler.step(val_loss)                                                                #if validation loss doesn't improve change learning rate

            print(f"Epoch {epoch+1}, Validation Loss: {val_loss.item()}")

            # Early stopping
            if val_loss < best_val_loss - min_delta:                                            #if validation loss doesn't improve for many iterations 
                best_val_loss = val_loss                                                        #stop training (to avoid overfitting)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            # Check for convergence
        ''' if len(training_losses) > 2 and abs(training_losses[-1] - training_losses[-2]) < min_delta:  #will correct this later (although not really required )
                print(f"Training loss converged at epoch {epoch+1}")                        
                break'''
            

    def validation_loss(self, x_num, xcat, y):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():              #don't calculate gradients since we dont have to change parameters/ do a backward pass (efficient)
            outputs = self(x_num, xcat)
            self.validation_loss_val = nn.MSELoss()(outputs, y)
        return self.validation_loss_val

    @staticmethod
    def load_model(filepath):
        model = MLP()
        model.load_state_dict(torch.load(filepath))
        model.eval()
        return model

    def generate(self, input:list):
        self.eval()
        xnum = []       
        xcat = []
        j = 0
        for i in input:      #iterate over input list
            if isinstance(i, (float, int)): #if input is numerical append on xnum
                xnum.append(float(i))
            else:
                xcat.append(self.encoding_dicts[j][i.lower()]) #else use the respective encoding dictionary to encode the value 
                j += 1                                           #j++ to move on to the next dict
        if len(xnum)!=0:      
            xnum = torch.tensor(self.scalerx.transform([xnum]))  #again standardise the input and convert it into a tensor
        else:
            xnum=torch.tensor(xnum)                     
        
        with torch.no_grad():          #no need gradients as no backward pass 
            xcat = torch.tensor(xcat).unsqueeze(0) #convert it into a tensor and add a dimension
            predicted_value = self(xnum, xcat).tolist() #pass it through the neural network and generate output
            predicted_value=self.scalery.inverse_transform(predicted_value)
            return predicted_value

# Example usage:

def modelselector(xnum,xcat,y,scalerx,scalery,encoding_dicts):  #to increase the strenght  of model if needed
    model = MLP(xnum, xcat, y, scalerx, scalery, encoding_dicts,0)  #first start at strength 0 
    model.train_model()
    final_model=model
    val_loss0=model.validation_loss_val
    if val_loss0>0.15:                                                                      #if loss is high use stronger model
        print("loss is too high. using stronger model")                             

        model1 = MLP(xnum, xcat, y, scalerx, scalery, encoding_dicts,1)  
        model1.train_model()                                        
        val_loss1=model1.validation_loss_val
        
        if val_loss1>0.15:
            model2=MLP(xnum, xcat, y, scalerx, scalery, encoding_dicts,1) 
            model2.train_model()
            val_loss2=model2.validation_loss_val
            
            if val_loss0<val_loss1 and val_loss0<val_loss2:                     #choose the model with lowest value of loss
                final_model=model
            elif val_loss1<val_loss0 and val_loss1<val_loss2:
                final_model=model1
            else:
                final_model=model2
        
        
        else:                                                                           #choose the model with lower loss
            if val_loss0<val_loss1:
                final_model=model
            else:
                final_model=model1
            
            
            
        
           
       
    print(final_model) 
       
    return final_model

# def trainmodel(filepath):
#     xnum, xcat, y, scalerx, scalery, encoding_dicts = data_maker([0,2], [1], filepath)
#     model=MLP(xnum, xcat, y, scalerx, scalery, encoding_dicts,2)
#     model.train_model()
#     Val_loss = model.validation_loss_val
#     numpy_value = Val_loss.numpy()
#     scalar_value = numpy_value.item()
#     print(scalar_value)
#     return (model_save_path)
         


# z = model.generate([5000,'no'])









