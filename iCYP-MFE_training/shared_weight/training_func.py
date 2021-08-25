# Import libraries
import torch
import time

#===========================================================================================
# Train function
def train(epoch, model, criteron, optimizer, device, training_loader, time_count=False):
    #--------------------
    start = time.time()
    #--------------------
    model_ = model
    model_.train()
    train_loss = 0
    for _, data in enumerate(training_loader):
        dataX = data[0].to(device)
        list_labels = [data[i].to(device) for i in range(1,6)]
        #--------------------
        # output share_layer and list output of tasks
        Out_share, list_task = model_(dataX)
        #-------------------- 
        optimizer.zero_grad()
        #-------------------- 
        loss = 0
        for i in range(len(list_task)):
            loss += criteron(list_task[i], list_labels[i]) 
        loss = loss/len(list_task) 
        loss.backward()
        train_loss += loss.item()*len(dataX) #(loss.item is the average loss of training batch)
        optimizer.step() 
    #--------------------
    end = time.time()
    duration = end - start
    if time_count:
        print("Average time for training 1 epoch: {}".format(duration))
    #--------------------
    print('====> Epoch: {} Average Train Loss: {:.4f}'.format(epoch, train_loss / len(training_loader.dataset)))
    train_loss = (train_loss / len(training_loader.dataset) )
    #--------------------
    return train_loss

#===========================================================================================
# Test Function
def validate(epoch, model, criteron, optimizer, device, test_loader, time_count=False):
    #--------------------
    start = time.time()
    #--------------------
    model_ = model
    model_.eval()
    test_loss = 0
    with torch.no_grad():
        for _, data in enumerate(test_loader): 
            dataX = data[0].to(device)
            list_labels = [data[i].to(device) for i in range(1,6)]
            #--------------------
            # Output share_layer and list output of tasks
            out_share, list_task = model_(dataX)
            # loss
            loss = 0
            for i in range(len(list_task)):
                loss += criteron(list_task[i], list_labels[i])
            loss = loss/len(list_task)
            test_loss += loss.item()*len(dataX)
    #--------------------
    end = time.time()
    duration = end - start
    if time_count:
        print("Average time for validating 1 epoch: {}".format(duration))
    #--------------------
    test_loss  = (test_loss / len(test_loader.dataset) )
    print('====> Average Test Loss: {:.4f}'.format(test_loss))
    #--------------------
    return test_loss
