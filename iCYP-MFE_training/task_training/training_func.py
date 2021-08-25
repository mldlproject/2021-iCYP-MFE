# Import libraries
import torch
import time

#===========================================================================================
# Training Function
def train(epoch, model, criteron, optimizer, device, training_loader, time_count=False):
    model_ = model
    #--------------------
    start = time.time()
    #--------------------
    model_.train()
    train_loss = 0
    for _, (data, labels) in enumerate(training_loader):
        data = data.to(device)
        labels = labels.to(device)
        outputs = model_(data).view_as(labels)
        #-------------------- 
        optimizer.zero_grad()
        #-------------------- 
        loss = criteron(outputs, labels)  
        loss.backward()
        train_loss += loss.item()*len(data) #(loss.item is the average loss of training batch)
        optimizer.step() 
    #--------------------
    end = time.time()
    duration = end - start
    if time_count:
        print("Average time for training 1 epoch: {}".format(duration))
    #--------------------
    print('====> Epoch: {} Average Train Loss: {:.4f}'.format(epoch, train_loss / len(training_loader.dataset)))
    train_loss = (train_loss / len(training_loader.dataset) )
    return train_loss

#===========================================================================================
# Validation Function
def validate(epoch, model, criteron, device, validation_loader, time_count=False):
    #--------------------
    start = time.time()
    #--------------------
    model_ = model
    model_.eval()
    validation_loss = 0
    pred_prob = []
    with torch.no_grad():
        for _, (data, labels) in enumerate(validation_loader): 
            data = data.to(device)
            labels = labels.to(device)  
            outputs = model_(data).view_as(labels)
            loss = criteron(outputs, labels)
            validation_loss += loss.item()*len(data)
            pred_prob.append(outputs)
    #--------------------
    end = time.time()
    duration = end - start
    if time_count:
        print("Average time for validating 1 epoch: {}".format(duration))
    #--------------------
    validation_loss = (validation_loss / len(validation_loader.dataset) )
    print('====> Average Validation Loss: {:.4f}'.format(validation_loss))
    #--------------------
    return validation_loss, pred_prob
    
#===========================================================================================
# Test Function
def test(epoch, model, criteron, device, test_loader, time_count=False):
    #--------------------
    start = time.time()
    #--------------------
    model_ = model
    model_.eval()
    test_loss  = 0
    pred_prob  = []
    with torch.no_grad():
        for _, (data, labels) in enumerate(test_loader): 
            data = data.to(device)
            labels = labels.to(device)
            #--------------------
            outputs = model_(data).view_as(labels)
            #--------------------
            loss = criteron(outputs, labels)
            test_loss += loss.item()*len(data)       
            pred_prob.append(outputs)
    #--------------------
    end = time.time()
    duration = end - start
    if time_count:
        print("Average time for testing 1 epoch: {}".format(duration))
    #--------------------
    test_loss = (test_loss / len(test_loader.dataset) )
    print('====> Average Test Loss: {:.4f}'.format(test_loss))
    return test_loss, pred_prob
