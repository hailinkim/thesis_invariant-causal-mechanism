import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def train(model, dataloader, optimizer, criterion):
    device = next(model.parameters()).device
    # model.train()
    total_loss = 0

    for input_sequence, target_sequence in dataloader: #input_sequence: 1 x (N+1) x (S+1), target_sequence: 1 x N^2
        optimizer.zero_grad()
        #pre-process input, target sequence
        src = input_sequence.float().to(device) #(N+1) x (S+1)
        #right-shift and pad the target sequence
        tgt = torch.cat((torch.tensor([[2]], device = device), target_sequence[:,:-1].to(device)), dim = 1)
        output, parent, children = model(src, tgt) #output: decoder output; parent, children are logits
        # print("train decoder output : ", output)
        #transform decoder output to logits
        output_logits = model.generator(output)
        output_logits = output_logits.squeeze(2) # 1x N^2
        # print("logits: ", output_logits)
        # output_prob = model.sigmoid(output_logits)
        # print("prob: ", output_prob)
        # binary prediction
        # pred = (output_prob >= 0.3).int()
        # print("pred: ", pred)
        # Compute main loss and auxiliary loss
        ground_truth = target_sequence.float()
        # print("gt: ", ground_truth)
        autoregressive_loss = criterion(output_logits, ground_truth)
        # print(autoregressive_loss)
        auxiliary_loss = criterion(parent, ground_truth)  # Adjust based on actual targets
        auxiliary_loss2 = criterion(children, ground_truth)
        loss = autoregressive_loss + auxiliary_loss + auxiliary_loss2

        # Backward and optimize
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # return average loss
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion):
    device = next(model.parameters()).device
    # model.eval()  # Set the model to evaluation mode
    total_loss = 0

    with torch.no_grad():  # Disable gradient calculation
        for input_sequence, target_sequence in dataloader:
            src = input_sequence.float().to(device)
            tgt = torch.cat((torch.tensor([[2]], device=device), target_sequence[:,:-1].to(device)), dim=1)

            # Forward pass
            output, parent, children = model(src, tgt)
            output_logits = model.generator(output)
            output_logits = output_logits.squeeze(2)

            # Compute main loss and auxiliary loss
            ground_truth = target_sequence.float()
            autoregressive_loss = criterion(output_logits, ground_truth)
            auxiliary_loss = criterion(parent, ground_truth)  # Adjust based on actual targets
            auxiliary_loss2 = criterion(children, ground_truth)
            loss = autoregressive_loss + auxiliary_loss + auxiliary_loss2

            total_loss += loss.item()

    # Return average loss
    return total_loss / len(dataloader)


def test(model, dataloader):
    device = next(model.parameters()).device
    # model.eval()
    
    fpr_list = []
    fdr_list = []
    tpr_list = []
    hd_list = []
    with torch.no_grad():
        for input_sequence, target_sequence in dataloader: #input_sequence: 1 x (N+1) x (S+1)
            src = input_sequence.float().to(device) #(N+1) x (S+1)
            tgt = torch.ones(1, 1, device=device).type_as(target_sequence).fill_(2) #torch.tensor([[2]], device = device)
            max_len = (src.size(1)-1)**2 # N^2
            #autoregressive prediction
            for _ in range(max_len):
                summary = model.encode(src)
                summary.to(device)
                output, _, _ = model.decode(tgt, summary)
                # output, _, _ = model(src, tgt)
                # print("test output size: ", output.size())
                # print("test output: ", output)
                # print("test output reshaped: ", output[:,-1,:])
                #compute logit for the next element
                output_logits = model.generator(output[:,-1,:])
                # print("output: ", output[:,-1,:])
                output_prob = model.sigmoid(output_logits)
                # print("prob: ", output_prob)
                # binary prediction
                pred = (output_prob >= 0.3).int()
                tgt = torch.cat([tgt, pred], dim=1)
                
            #exclude BOS token(2) from the prediction
            tgt = tgt[:,1:].float().to(device)
            print("pred: ", tgt)
            print("tgt: ", target_sequence)
            assert tgt.size() == target_sequence.size(), "wrong output dimension"
            #compute evaluation metrics between prediction and ground-truth adj mat
            tpr, fpr, fdr, hd = compute_metrics(target_sequence.to(device), tgt)
            tpr_list.append(tpr.item())
            fpr_list.append(fpr.item())
            fdr_list.append(fdr.item())
            hd_list.append(hd.item())
    return np.mean(tpr_list), np.mean(fpr_list), np.mean(fdr_list), np.mean(hd_list)


def compute_metrics(ground_truth, predicted):
    # True Positives (TP)
    TP = torch.sum((ground_truth == 1) & (predicted == 1))

    # True Negatives (TN)
    TN = torch.sum((ground_truth == 0) & (predicted == 0))

    # False Positives (FP)
    FP = torch.sum((ground_truth == 0) & (predicted == 1))

    # False Negatives (FN)
    FN = torch.sum((ground_truth == 1) & (predicted == 0))

    # Computing TPR, FPR, FDR
    TPR = TP.float() / (TP + FN) if (TP + FN) != 0 else torch.tensor(0.0) #TP out of actual true's
    FPR = FP.float() / (FP + TN) if (FP + TN) != 0 else torch.tensor(0.0)
    FDR = FP.float() / (FP + TP) if (FP + TP) != 0 else torch.tensor(0.0) #FP out of predicted true's

    # Computing Hamming Distance
    hamming_distance = torch.sum(ground_truth != predicted)

    return TPR, FPR, FDR, hamming_distance
