# -*- coding: utf-8 -*-
import math
import os
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import DataHelper as DH
import emetrics as EM

# Transformer Parameters
d_model = 128 # Embedding Size
d_ff = 512 # FeedForward dimension
d_k = d_v = 32 # dimension of K(=Q), V
n_layers = 1 # number of Encoder
n_heads = 4 # number of heads in Multi-Head Attention

def seed_torch(seed=2):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

class DatasetIterater(Data.Dataset):
    def __init__(self, texta, textb, label):
        self.texta = texta
        self.textb = textb
        self.label = label

    def __getitem__(self, item):
        return self.texta[item], self.textb[item], self.label[item]

    def __len__(self):
        return len(self.texta)

def BatchPad(batch_data, pad=0):
    texta, textb, label = list(zip(*batch_data))
    max_len_a = max([len(seq_a) for seq_a in texta])
    max_len_b = max([len(seq_b) for seq_b in textb])
    texta = [seq+[pad]*(max_len_a-len(seq)) for seq in texta]
    textb = [seq+[pad]*(max_len_b-len(seq)) for seq in textb]
    texta = torch.LongTensor(texta)
    textb = torch.LongTensor(textb)
    label = torch.FloatTensor(label)
    return (texta, textb, label)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def get_attn_pad_mask(seq_q, seq_k):
    # seq_q=seq_k: [batch_size, seq_len]

    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k) # [batch_size, len_q, len_k]

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # Q: [batch_size, n_heads, len_q, d_k]
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.fc0 = nn.Linear(d_model, d_model, bias=False)
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        # input_Q: [batch_size, len_q, d_model]
        # input_K: [batch_size, len_k, d_model]
        # input_V: [batch_size, len_v(=len_k), d_model]
        # attn_mask: [batch_size, seq_len, seq_len]
        
        ##residual, batch_size = input_Q, input_Q.size(0)
        batch_size, seq_len, model_len = input_Q.size()
        residual_2D = input_Q.view(batch_size*seq_len, model_len)
        residual = self.fc0(residual_2D).view(batch_size, seq_len, model_len)

        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2) # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2) # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                      2) # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                               1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v]
        # attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).cuda()(output+residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        # inputs: [batch_size, seq_len, d_model]
        
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output+residual) # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        # enc_inputs: [batch_size, src_len, d_model]
        # enc_self_attn_mask: [batch_size, src_len, src_len]

        # enc_outputs: [batch_size, src_len, d_model]
        # attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.stream0 = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.stream1 = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.stream2 = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
    def forward(self, enc_inputs):
        #enc_inputs: [batch_size, src_len]
        
        enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]
        stream0 = enc_outputs

        enc_self_attns0, enc_self_attns1, enc_self_attns2 = [], [], []
        for layer in self.stream0:
            # enc_outputs: [batch_size, src_len, d_model]
            # enc_self_attn: [batch_size, n_heads, src_len, src_len]
            stream0, enc_self_attn0 = layer(stream0, enc_self_attn_mask)
            enc_self_attns0.append(enc_self_attn0)

        #skip connect -> stream0
        stream1 = stream0 + enc_outputs
        stream2 = stream0 + enc_outputs
        for layer in self.stream1:
            stream1, enc_self_attn1 = layer(stream1, enc_self_attn_mask)
            enc_self_attns1.append(enc_self_attn1)

        for layer in self.stream2:
            stream2, enc_self_attn2 = layer(stream2, enc_self_attn_mask)
            enc_self_attns2.append(enc_self_attn2)

        return torch.cat((stream1, stream2), 2), enc_self_attns0, enc_self_attns1, enc_self_attns2

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()

        self.encoderD = Encoder(DH.drugSeq_vocabSize)
        self.encoderT = Encoder(DH.targetSeq_vocabSize)
        self.fc0 = nn.Sequential(
            nn.Linear(4*d_model, 16*d_model, bias=False),
            nn.LayerNorm(16*d_model),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16*d_model, 4*d_model, bias=False),
            nn.LayerNorm(4*d_model),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Linear(4*d_model, 1, bias=False)

    def forward(self, input_Drugs, input_Tars):
        # input: [batch_size, src_len]

        # enc_outputs: [batch_size, src_len, d_model]
        # enc_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_Drugs, enc_attnsD0, enc_attnsD1, enc_attnsD2 = self.encoderD(input_Drugs)
        enc_Tars, enc_attnsT0, enc_attnsT1, enc_attnsT2 = self.encoderT(input_Tars)

        enc_Drugs_2D0 = torch.sum(enc_Drugs, dim=1)
        enc_Drugs_2D1 = enc_Drugs_2D0.squeeze()
        enc_Tars_2D0 = torch.sum(enc_Tars, dim=1)
        enc_Tars_2D1 = enc_Tars_2D0.squeeze()
        #fc = enc_Drugs_2D1 + enc_Tars_2D1
        fc = torch.cat((enc_Drugs_2D1, enc_Tars_2D1), 1)

        fc0 = self.fc0(fc)
        fc1 = self.fc1(fc0)
        fc2 = self.fc2(fc1)
        affi = fc2.squeeze()

        return affi, enc_attnsD0, enc_attnsT0, enc_attnsD1, enc_attnsT1, enc_attnsD2, enc_attnsT2

if __name__ == '__main__':
    '''
    smile_maxlenDB, proSeq_maxlenDB = 100, 1000
    trainKB_num, testKB_num = 263583, 113168
    
    # bindingDB dataset from DeepAffinity, comments='!' : avoid ‘#’
    fpath_DBTrian_ic50 = 'D:/zl/transDTA-version4/data_bindingDB/0 train_ic50'
    fpath_DBTrain_drugSeqs = 'D:/zl/transDTA-version4/data_bindingDB/0 train_smile'
    fpath_DBTrain_targetSeqs = 'D:/zl/transDTA-version4/data_bindingDB/0 train_sps'

    fpath_DBTest_ic50 = 'D:/zl/transDTA-version4/data_bindingDB/4 kinase_ic50'
    fpath_DBTest_drugSeqs = 'D:/zl/transDTA-version4/data_bindingDB/4 kinase_smile'
    fpath_DBTest_targetSeqs = 'D:/zl/transDTA-version4/data_bindingDB/4 kinase_sps'

    pIc50_DBTrain = np.loadtxt(fpath_DBTrian_ic50, dtype=float).tolist()
    targetSeqs_DBTrain = np.loadtxt(fpath_DBTrain_targetSeqs, dtype=str).tolist()
    drugSeqs_DBTrain = np.loadtxt(fpath_DBTrain_drugSeqs, dtype=str, comments='!').tolist()

    pIc50_DBTest = np.loadtxt(fpath_DBTest_ic50, dtype=float).tolist()
    targetSeqs_DBTest = np.loadtxt(fpath_DBTest_targetSeqs, dtype=str).tolist()
    drugSeqs_DBTest = np.loadtxt(fpath_DBTest_drugSeqs, dtype=str, comments='!').tolist()

    labeledTrain_drugs, labelTrain_targets = DH.LabelDT(drugSeqs_DBTrain, targetSeqs_DBTrain,
                                                             smile_maxlenDB, proSeq_maxlenDB, sps=True)

    labeledTest_drugs, labelTest_targets = DH.LabelDT(drugSeqs_DBTest, targetSeqs_DBTest,
                                                             smile_maxlenDB, proSeq_maxlenDB, sps=True)

    labeledDrugs_shuttle, labeledTargets_shuttle, affiMatrix_shuttle \
                                    = DH.shuttle(labeledTrain_drugs, labelTrain_targets, pIc50_DBTrain)
    '''
    # davis dataset from DeepDTA
    '''smile_maxlenDA, proSeq_maxlenDA = 85, 1200
    trainDV_num, testDV_num = 25046, 5010
    #valDV_num = 2010
    fpath_davis = 'D:/zl/transDTA-version4/data_davis/'
    '''
    # kiba dataset from DeepDTA
    smile_maxlenKB, proSeq_maxlenKB = 100, 1000
    trainKB_num, testKB_num = 98545, 19709
    fpath_kiba = 'D:/zl/transDTA-version4/data_kiba/'

    # davis -> logspance_trans=True. kiba -> logspance_trans=False
    # davis: affinity=pKd. kiba: affinity=kiba score
    drug, target, affinity = DH.LoadData(fpath_kiba, logspance_trans=False)
    drug_seqs, target_seqs, affiMatrix = DH.GetSamples('kiba', drug, target, affinity)
    labeled_drugs, labeled_targets = DH.LabelDT(drug_seqs, target_seqs,
                                                      smile_maxlenKB, proSeq_maxlenKB)
    # shuttle
    labeledDrugs_shuttle, labeledTargets_shuttle, affiMatrix_shuttle \
                                   = DH.shuttle(labeled_drugs, labeled_targets, affiMatrix)

    #train data 5-fold
    '''Drugs_fold1 = labeledDrugs_shuttle[0:5010]
    Targets_fold1 = labeledTargets_shuttle[0:5010]
    affiMatrix_fold1 = affiMatrix_shuttle[0:5010]

    Drugs_fold2 = labeledDrugs_shuttle[5010:10020]
    Targets_fold2 = labeledTargets_shuttle[5010:10020]
    affiMatrix_fold2 =affiMatrix_shuttle[5010:10020]

    Drugs_fold3 = labeledDrugs_shuttle[10020:15030]
    Targets_fold3 = labeledTargets_shuttle[10020:15030]
    affiMatrix_fold3 =affiMatrix_shuttle[10020:15030]

    Drugs_fold4 = labeledDrugs_shuttle[15030:20040]
    Targets_fold4 = labeledTargets_shuttle[15030:20040]
    affiMatrix_fold4 =affiMatrix_shuttle[15030:20040]

    Drugs_fold5 = labeledDrugs_shuttle[20040:25046]
    Targets_fold5 = labeledTargets_shuttle[20040:25046]
    affiMatrix_fold5 =affiMatrix_shuttle[20040:25046]

    Drugs_fold6 = labeledDrugs_shuttle[25046:30056]
    Targets_fold6 = labeledTargets_shuttle[25046:30056]
    affiMatrix_fold6 = affiMatrix_shuttle[25046:30056]'''

    Drugs_fold1 = labeledDrugs_shuttle[0:19709]
    Targets_fold1 = labeledTargets_shuttle[0:19709]
    affiMatrix_fold1 = affiMatrix_shuttle[0:19709]

    Drugs_fold2 = labeledDrugs_shuttle[19709:39418]
    Targets_fold2 = labeledTargets_shuttle[19709:39418]
    affiMatrix_fold2 =affiMatrix_shuttle[19709:39418]

    Drugs_fold3 = labeledDrugs_shuttle[39418:59127]
    Targets_fold3 = labeledTargets_shuttle[39418:59127]
    affiMatrix_fold3 =affiMatrix_shuttle[39418:59127]

    Drugs_fold4 = labeledDrugs_shuttle[59127:78836]
    Targets_fold4 = labeledTargets_shuttle[59127:78836]
    affiMatrix_fold4 =affiMatrix_shuttle[59127:78836]

    Drugs_fold5 = labeledDrugs_shuttle[78836:98545]
    Targets_fold5 = labeledTargets_shuttle[78836:98545]
    affiMatrix_fold5 =affiMatrix_shuttle[78836:98545]

    Drugs_fold6 = labeledDrugs_shuttle[98545:118254]
    Targets_fold6 = labeledTargets_shuttle[98545:118254]
    affiMatrix_fold6 = affiMatrix_shuttle[98545:118254]

    #98545
    train1_drugs = np.hstack((Drugs_fold1, Drugs_fold2, Drugs_fold3, Drugs_fold4, Drugs_fold5))
    train1_targets = np.hstack((Targets_fold1, Targets_fold2, Targets_fold3, Targets_fold4, Targets_fold5))
    train1_affinity = np.hstack((affiMatrix_fold1, affiMatrix_fold2, affiMatrix_fold3, affiMatrix_fold4, affiMatrix_fold5))
    #98545
    train2_drugs = np.hstack((Drugs_fold2, Drugs_fold3, Drugs_fold4, Drugs_fold5, Drugs_fold1))
    train2_targets = np.hstack((Targets_fold2, Targets_fold3, Targets_fold4, Targets_fold5, Targets_fold1))
    train2_affinity = np.hstack((affiMatrix_fold2, affiMatrix_fold3, affiMatrix_fold4, affiMatrix_fold5, affiMatrix_fold1))
    #98545
    train3_drugs = np.hstack((Drugs_fold3, Drugs_fold4, Drugs_fold5, Drugs_fold1, Drugs_fold2))
    train3_targets = np.hstack((Targets_fold3, Targets_fold4, Targets_fold5, Targets_fold1, Targets_fold2))
    train3_affinity = np.hstack((affiMatrix_fold3, affiMatrix_fold4, affiMatrix_fold5, affiMatrix_fold1, affiMatrix_fold2))
    #98545
    train4_drugs = np.hstack((Drugs_fold4, Drugs_fold5, Drugs_fold1, Drugs_fold2, Drugs_fold3))
    train4_targets = np.hstack((Targets_fold4, Targets_fold5, Targets_fold1, Targets_fold2, Targets_fold3))
    train4_affinity = np.hstack((affiMatrix_fold4, affiMatrix_fold5, affiMatrix_fold1, affiMatrix_fold2, affiMatrix_fold3))
    #98545
    train5_drugs = np.hstack((Drugs_fold5, Drugs_fold1, Drugs_fold2, Drugs_fold3, Drugs_fold4))
    train5_targets = np.hstack((Targets_fold5, Targets_fold1, Targets_fold2, Targets_fold3, Targets_fold4))
    train5_affinity = np.hstack((affiMatrix_fold5, affiMatrix_fold1, affiMatrix_fold2, affiMatrix_fold3, affiMatrix_fold4))

    # train_loader <- davis or kiba or bindingDB
    '''Data_iter = DatasetIterater(labeledDrugs_shuttle, labeledTargets_shuttle, affiMatrix_shuttle)
    train_iter, val_iter, test_iter = Data.random_split(Data_iter, [23036, 2010, 5010])
    train_loader = Data.DataLoader(train_iter, batch_size, False, collate_fn=BatchPad)
    val_loader = Data.DataLoader(val_iter, batch_size, False, collate_fn=BatchPad)
    test_loader = Data.DataLoader(test_iter, batch_size, False, collate_fn=BatchPad)'''
    '''
    train_iter = DatasetIterater(labeledDrugs_shuttle, labeledTargets_shuttle, affiMatrix_shuttle)
    test_iter = DatasetIterater(labeledTest_drugs, labelTest_targets, pIc50_DBTest)
    train_loader = Data.DataLoader(train_iter, batch_size, False, collate_fn=BatchPad)
    test_loader = Data.DataLoader(test_iter, batch_size, False, collate_fn=BatchPad)
    '''
    #Data_iter = DatasetIterater(labeledDrugs_shuttle, labeledTargets_shuttle, affiMatrix_shuttle)
    #train_iter, test_iter = Data.random_split(Data_iter, [trainDV_num, testDV_num])

    model_fromTrain1 = 'D:/zl/transDTA-version4/model_fromTrain1.pth'
    model_fromTrain2 = 'D:/zl/transDTA-version4/model_fromTrain2.pth'
    model_fromTrain3 = 'D:/zl/transDTA-version4/model_fromTrain3.pth'
    model_fromTrain4 = 'D:/zl/transDTA-version4/model_fromTrain4.pth'
    model_fromTrain5 = 'D:/zl/transDTA-version4/model_fromTrain5.pth'
    #model_fromVal = 'D:/zl/transDTA-version4/model_fromVal.pth'

    MSE, CI, RM2 = [], [], []
    for count in range(5):

        model = Transformer().cuda()
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        EPOCHS, batch_size, accumulation_steps = 600, 32, 32  # bs=1024 -> update loss
        trainEP_loss_list = []
        # valEP_loss_list = []
        min_train_loss = 100000  # save best model in train
        # min_val_loss = 100000 # save best model in val

        if count == 0:
            train_iter = DatasetIterater(train1_drugs, train1_targets, train1_affinity)
        elif count == 1:
            train_iter = DatasetIterater(train2_drugs, train2_targets, train2_affinity)
        elif count == 2:
            train_iter = DatasetIterater(train3_drugs, train3_targets, train3_affinity)
        elif count == 3:
            train_iter = DatasetIterater(train4_drugs, train4_targets, train4_affinity)
        else:
            train_iter = DatasetIterater(train5_drugs, train5_targets, train5_affinity)

        test_iter = DatasetIterater(Drugs_fold6, Targets_fold6, affiMatrix_fold6)
        train_loader = Data.DataLoader(train_iter, batch_size, False, collate_fn=BatchPad)
        test_loader = Data.DataLoader(test_iter, batch_size, False, collate_fn=BatchPad)

        '''
        ###############
        ##Train Process
        ###############
        '''
        seed_torch(seed=2)
        for epoch in range(EPOCHS):
            torch.cuda.synchronize()
            start = time.time()
            model.train() # -> model.eval(), start Batch Normalization and Dropout

            train_sum_loss = 0
            for train_batch_idx, (SeqDrug, SeqTar, real_affi) in enumerate(train_loader):

                SeqDrug, SeqTar, real_affi = SeqDrug.cuda(), SeqTar.cuda(), real_affi.cuda()
                pre_affi, enc_attnD0, enc_attnT0, enc_attnsD1, enc_attnsT1,enc_attnsD2, enc_attnsT2 \
                                                = model(SeqDrug, SeqTar) # pre_affi: [batch_affini]

                train_loss = criterion(pre_affi, real_affi)
                train_sum_loss += train_loss.item() # loss -> loss.item(), avoid CUDA out of memory

                train_loss.backward()
                # batch_size from 32 -> 1024
                if ((train_batch_idx+1)%accumulation_steps) == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                if ((train_batch_idx+1)%300) == 0:
                    print('Epoch:', '%04d' % (epoch+1), 'loss =', '{:.6f}'.format(train_loss))

                if (train_batch_idx+1) == (trainKB_num//batch_size+1):
                    train_epoch_loss = train_sum_loss / train_batch_idx
                    trainEP_loss_list.append(train_epoch_loss)
                    print('\n')
                    print('Epoch:', '%04d' % (epoch+1), 'train_epoch_loss = ', '{:.6f}'.format(train_epoch_loss))

                    # save best train model
                    if train_epoch_loss < min_train_loss:
                        min_train_loss = train_epoch_loss
                        if count == 0:
                            torch.save(model.state_dict(), model_fromTrain1)
                            print('Best model in train1 from', '%04d' % (epoch+1), 'Epoch', 'at', format(model_fromTrain1))
                        elif count == 1:
                            torch.save(model.state_dict(), model_fromTrain2)
                            print('Best model in train2 from', '%04d' % (epoch+1), 'Epoch', 'at', format(model_fromTrain2))
                        elif count == 2:
                            torch.save(model.state_dict(), model_fromTrain3)
                            print('Best model in train3 from', '%04d' % (epoch+1), 'Epoch', 'at', format(model_fromTrain3))
                        elif count == 3:
                            torch.save(model.state_dict(), model_fromTrain4)
                            print('Best model in train4 from', '%04d' % (epoch+1), 'Epoch', 'at', format(model_fromTrain4))
                        else:
                            torch.save(model.state_dict(), model_fromTrain5)
                            print('Best model in train5 from', '%04d' % (epoch+1), 'Epoch', 'at', format(model_fromTrain5))

            # Val Process
            '''
            val_sum_loss = 0
            val_obs, val_pred = [], []
            with torch.no_grad():
                for val_batch_idx, (SeqDrug, SeqTar, real_affi) in enumerate(val_loader):
    
                    SeqDrug, SeqTar, real_affi = SeqDrug.cuda(), SeqTar.cuda(), real_affi.cuda()
                    pre_affi, enc_attnD0, enc_attnT0, enc_attnsD1, enc_attnsT1, enc_attnsD2, enc_attnsT2 \
                                                                                  = model(SeqDrug, SeqTar)
    
                    val_loss = criterion(pre_affi, real_affi)
                    val_sum_loss += val_loss.item()  # loss -> loss.item(), avoid CUDA out of memory
    
                    val_obs.extend(real_affi.tolist())
                    val_pred.extend(pre_affi.tolist())
    
                    if (val_batch_idx+1) == (valDV_num//batch_size+1):
                        val_epoch_loss = val_sum_loss / val_batch_idx
                        valEP_loss_list.append(val_epoch_loss)
                        print('Epoch:', '%04d' % (epoch+1), 'val_epoch_loss = ', '{:.6f}'.format(val_epoch_loss))
    
                        # save best val model
                        if val_epoch_loss < min_val_loss:
                            min_val_loss = val_epoch_loss
                            torch.save(model.state_dict(), model_fromVal)
                            print('Best model in val from', '%04d' % (epoch+1), 'Epoch', 'at', format(model_fromVal))
    
            print('val_MSE:', '{:.3f}'.format(EM.get_MSE(val_obs, val_pred)))
            print('val_CI:', '{:.3f}'.format(EM.get_cindex(val_obs, val_pred)))
            print('val_rm2:', '{:.3f}'.format(EM.get_rm2(val_obs, val_pred)))'''

            # record time for 1 epoch
            torch.cuda.synchronize()
            print('Time taken for 1 epoch is {:.4f} minutes'.format((time.time()-start)/60))
            print('\n')

        np.savetxt('trainLossMean_list.csv', trainEP_loss_list, delimiter=',')
        #np.savetxt('valLossMean_list.csv', valEP_loss_list, delimiter=',')

        '''
        ###############
        ##Test Process
        ###############
        '''
        predModel = Transformer().cuda()
        if count == 0:
            predModel.load_state_dict(torch.load(model_fromTrain1))
        elif count == 1:
            predModel.load_state_dict(torch.load(model_fromTrain2))
        elif count == 2:
            predModel.load_state_dict(torch.load(model_fromTrain3))
        elif count == 3:
            predModel.load_state_dict(torch.load(model_fromTrain4))
        else:
            predModel.load_state_dict(torch.load(model_fromTrain5))
        predModel.eval()  # -> model.train(), keep Batch Normalization and avoid Dropout

        train_obs, train_pred = [], []
        test_obs, test_pred = [], []
        with torch.no_grad():
            for (DrugSeqs, TarSeqs, real_affi) in train_loader:
                DrugSeqs, TarSeqs, real_affi = DrugSeqs.cuda(), TarSeqs.cuda(), real_affi.cuda()
                pred_affi, enc_attnD0, enc_attnT0, enc_attnsD1, enc_attnsT1, enc_attnsD2, enc_attnsT2\
                                           = predModel(DrugSeqs, TarSeqs)  # pred_affi: [batch_affini]
                train_obs.extend(real_affi.tolist())
                train_pred.extend(pred_affi.tolist())

            for (DrugSeqs, TarSeqs, real_affi) in test_loader:
                DrugSeqs, TarSeqs, real_affi = DrugSeqs.cuda(), TarSeqs.cuda(), real_affi.cuda()
                pred_affi, enc_attnD0, enc_attnT0, enc_attnsD1, enc_attnsT1, enc_attnsD2, enc_attnsT2\
                                           = predModel(DrugSeqs, TarSeqs)  # pred_affi: [batch_affini]
                test_obs.extend(real_affi.tolist())
                test_pred.extend(pred_affi.tolist())

        #np.savetxt('test_obs.csv', test_obs, delimiter=',')
        #np.savetxt('test_pred.csv', test_pred, delimiter=',')

        print('train_MSE:', '{:.3f}'.format(EM.get_MSE(train_obs, train_pred)))
        print('train_CI:', '{:.3f}'.format(EM.get_cindex(train_obs, train_pred)))
        print('train_rm2:', '{:.3f}'.format(EM.get_rm2(train_obs, train_pred)))

        print('\n')
        print('test_MSE:', '{:.3f}'.format(EM.get_MSE(test_obs, test_pred)))
        print('test_CI:', '{:.3f}'.format(EM.get_cindex(test_obs, test_pred)))
        print('test_rm2:', '{:.3f}'.format(EM.get_rm2(test_obs, test_pred)))

        mse = EM.get_MSE(test_obs, test_pred)
        ci = EM.get_cindex(test_obs, test_pred)
        rm2 = EM.get_rm2(test_obs, test_pred)

        MSE.append(mse)
        CI.append(ci)
        RM2.append(rm2)

    np.savetxt('mse.csv', MSE, delimiter=',')
    np.savetxt('CI.csv', CI, delimiter=',')
    np.savetxt('RM2.csv', RM2, delimiter=',')