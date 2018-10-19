import torch
device=torch.device('cpu')

def better_compute2(input_mtx,filtersize = 3):
    #use with dilated convolution
    
    stride=1.5 # algorithm only works for stride value 1.5

    assert input_mtx.dim()==4,\
    "Input tensor dimension is %dD instead of 4D" %input_mtx.dim()
  
    batchsize = input_mtx.size()[0]
    channels = input_mtx.size()[1]
    input_rows = input_mtx.size()[2] #height
    input_cols = input_mtx.size()[3] #width
    
    rows = ((input_rows-filtersize)/stride)+1 #output H dimension
    columns = ((input_cols-filtersize)/stride)+1 #output W dimension
    assert rows%1 == 0 and columns%1 ==0,\
    "Invalid output HxW dimension, current output dimension for HxW is %f x %f" %(rows,columns) #safety check
  #  rows = int(rows)
   # columns = int(columns)
    new_rows = (2*input_rows)-1
    new_cols =  (2*input_cols)-1
    
    itm1 = torch.zeros(input_rows,new_cols,device=device,requires_grad=False)
    itm2 = torch.zeros(new_rows,new_cols,device=device,requires_grad=False)
    output2 = torch.zeros(batchsize,channels,new_rows,new_cols,device=device,requires_grad=False)
    
    #im gonna put rows and cols operations together here, input matrix should be a square anyway, so input_rows = input_cols
    for batch in range(batchsize):
        for chl in range(channels):
            for width in range(input_cols): #copy values into horizontally expanded matrix
                itm1[:,2*width] = input_mtx[batch][chl][:,width]
                
            for cols in range(new_cols):   #compute in-between values
                if cols%2 !=0:
                    itm1[:,cols] = (itm1[:,cols-1] + itm1[:,cols+1])/2
            
            for height in range(input_rows): #copy values into vertically expanded matrix
                itm2[2*height] = itm1[height]
             
            for rows in range(new_rows):
                if rows%2 !=0:
                    itm2[rows] = (itm2[rows-1] + itm2[rows+1])/2
            
            output2[batch][chl] = itm2

    return output2

def better_compute3(input_mtx,filtersize = 3):
    #use with dilated convolution
    
    stride=1.5 # algorithm only works for stride value 1.5

    assert input_mtx.dim()==4,\
    "Input tensor dimension is %dD instead of 4D" %input_mtx.dim()
  
    batchsize = input_mtx.size()[0]
    channels = input_mtx.size()[1]
    input_rows = input_mtx.size()[2] #height
    input_cols = input_mtx.size()[3] #width
    
    rows = ((input_rows-filtersize)/stride)+1 #output H dimension
    columns = ((input_cols-filtersize)/stride)+1 #output W dimension
    assert rows%1 == 0 and columns%1 ==0,\
    "Invalid output HxW dimension, current output dimension for HxW is %f x %f" %(rows,columns) #safety check
  #  rows = int(rows)
   # columns = int(columns)
    new_rows = (2*input_rows)-1
    new_cols =  (2*input_cols)-1
    
    itm1 = torch.zeros(input_rows,new_cols,device=device,requires_grad=False)
    itm2 = torch.zeros(new_rows,new_cols,device=device,requires_grad=False)
    output2 = torch.zeros(batchsize,channels,new_rows,new_cols,device=device,requires_grad=False)
    
    for batch in range(batchsize):
        for chl in range(channels):
            itm1[:,::2] = input_mtx[batch][chl][:,::1] #fill in alternating columns
            itm1[:,1:-1:2] = (itm1[:,0:-1:2] +itm1[:,2:new_rows:2])/2     #calculate inbetween column values
            itm2[::2,:] = itm1[::1,:] #fill in alternating rows
            itm2[1:-1:2,:] = (itm2[0:-1:2,:] + itm2[2:new_cols:2,:])/2 #calculate inbetween row values
            output2[batch][chl] = itm2
        
    return output2

	

 