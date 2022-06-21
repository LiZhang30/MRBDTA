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

#Transformer Parameters
d_model = 128 #Embedding Size
d_ff = 512 #FeedForward dimension
d_k = d_v = 32 #dimension of K(=Q), V
n_layers = 1 #number of Encoder
n_heads = 4 #number of heads in Multi-Head Attention

def seed_torch(seed=1029):
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
        #x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def get_attn_pad_mask(seq_q, seq_k):
    #seq_q=seq_k: [batch_size, seq_len]
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    #eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        #Q: [batch_size, n_heads, len_q, d_k]
        #K: [batch_size, n_heads, len_k, d_k]
        #V: [batch_size, n_heads, len_v(=len_k), d_v]
        #attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
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
        #input_Q: [batch_size, len_q, d_model]
        #input_K: [batch_size, len_k, d_model]
        #input_V: [batch_size, len_v(=len_k), d_model]
        #attn_mask: [batch_size, seq_len, seq_len]
        
        ##residual, batch_size = input_Q, input_Q.size(0)
        batch_size, seq_len, model_len = input_Q.size()
        residual_2D = input_Q.view(batch_size*seq_len, model_len)
        residual = self.fc0(residual_2D).view(batch_size, seq_len, model_len)

        #(B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        #context: [batch_size, n_heads, len_q, d_v]
        #attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
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
        #inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output+residual)

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        #enc_inputs: [batch_size, src_len, d_model]
        #enc_self_attn_mask: [batch_size, src_len, src_len]

        #enc_outputs: [batch_size, src_len, d_model]
        #attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
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
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        stream0 = enc_outputs

        enc_self_attns0, enc_self_attns1, enc_self_attns2 = [], [], []
        for layer in self.stream0:
            #enc_outputs: [batch_size, src_len, d_model]
            #enc_self_attn: [batch_size, n_heads, src_len, src_len]
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
        #input: [batch_size, src_len]

        #enc_outputs: [batch_size, src_len, d_model]
        #enc_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_Drugs, enc_attnsD0, enc_attnsD1, enc_attnsD2 = self.encoderD(input_Drugs)
        enc_Tars, enc_attnsT0, enc_attnsT1, enc_attnsT2 = self.encoderT(input_Tars)

        enc_Drugs_2D0 = torch.sum(enc_Drugs, dim=1)
        enc_Drugs_2D1 = enc_Drugs_2D0.squeeze()
        enc_Tars_2D0 = torch.sum(enc_Tars, dim=1)
        enc_Tars_2D1 = enc_Tars_2D0.squeeze()
        fc = torch.cat((enc_Drugs_2D1, enc_Tars_2D1), 1)

        fc0 = self.fc0(fc)
        fc1 = self.fc1(fc0)
        fc2 = self.fc2(fc1)
        affi = fc2.squeeze()

        return affi, enc_attnsD0, enc_attnsT0, enc_attnsD1, enc_attnsT1, enc_attnsD2, enc_attnsT2

if __name__ == '__main__':
    #seed_torch()

    '''# davis dataset
    smile_maxlenDA, proSeq_maxlenDA = 85, 1200
    trainDV_num, testDV_num = 25046, 5010
    fpath_davis = 'D:/zl/transDTA-version4/data_davis/'
    '''
    #kiba dataset
    smile_maxlenKB, proSeq_maxlenKB = 100, 1000
    trainKB_num, testKB_num = 5000, 1000
    fpath_kiba = 'D:/zl/transDTA-version4/data_kiba/'
    
    #davis -> logspance_trans=True. kiba -> logspance_trans=False
    #davis: affinity=pKd. kiba: affinity=kiba score
    drug, target, affinity = DH.LoadData(fpath_kiba, logspance_trans=False)
    drug_seqs, target_seqs, affiMatrix = DH.GetSamples('kiba', drug, target, affinity)
    labeled_drugs, labeled_targets = DH.LabelDT(drug_seqs, target_seqs,
                                                      smile_maxlenKB, proSeq_maxlenKB)
    #shuttle
    labeledDrugs_shuttle, labeledTargets_shuttle, affiMatrix_shuttle \
                                   = DH.Shuttle(labeled_drugs, labeled_targets, affiMatrix)

    model = Transformer().cuda()
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS, batch_size, accumulation_steps = 10, 32, 8 #bs=32*8=256 -> update loss
    trainEP_loss_list = []
    #valEP_loss_list = []
    model_fromTrain = 'D:/zl/transDTA-version4/model_fromTrain.pth'
    #model_fromVal = 'D:/zl/transDTA-version4/model_fromVal.pth'
    min_train_loss = 100000 #save best model in train
    #min_val_loss = 100000 #save best model in val

    #train_loader <- davis or kiba
    '''Data_iter = DatasetIterater(labeledDrugs_shuttle, labeledTargets_shuttle, affiMatrix_shuttle)
    train_iter, val_iter, test_iter = Data.random_split(Data_iter, [23036, 2010, 5010])
    train_loader = Data.DataLoader(train_iter, batch_size, False, collate_fn=BatchPad)
    val_loader = Data.DataLoader(val_iter, batch_size, False, collate_fn=BatchPad)
    test_loader = Data.DataLoader(test_iter, batch_size, False, collate_fn=BatchPad)'''

    Data_iter = DatasetIterater(labeledDrugs_shuttle[0:trainKB_num+testKB_num],
                                                    labeledTargets_shuttle[0:trainKB_num+testKB_num],
                                                    affiMatrix_shuttle[0:trainKB_num+testKB_num])
    train_iter, test_iter = Data.random_split(Data_iter, [trainKB_num, testKB_num])
    train_loader = Data.DataLoader(train_iter, batch_size, False, collate_fn=BatchPad)
    test_loader = Data.DataLoader(test_iter, batch_size, False, collate_fn=BatchPad)

    '''
    ###############
    ##Train Process
    ###############
    '''
    for epoch in range(EPOCHS):
        torch.cuda.synchronize()
        start = time.time()
        model.train()

        train_sum_loss = 0
        for train_batch_idx, (SeqDrug, SeqTar, real_affi) in enumerate(train_loader):

            SeqDrug, SeqTar, real_affi = SeqDrug.cuda(), SeqTar.cuda(), real_affi.cuda()
            pre_affi, enc_attnD0, enc_attnT0, enc_attnsD1, enc_attnsT1,enc_attnsD2, enc_attnsT2 \
                                            = model(SeqDrug, SeqTar)

            train_loss = criterion(pre_affi, real_affi)
            train_sum_loss += train_loss.item()

            train_loss.backward()
            #batch_size from 32 -> 256
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
                    torch.save(model.state_dict(), model_fromTrain)
                    print('Best model in train from', '%04d' % (epoch+1), 'Epoch', 'at', format(model_fromTrain))

        #Val Process
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

        #record time for 1 epoch
        torch.cuda.synchronize()
        print('Time taken for 1 epoch is {:.4f} minutes'.format((time.time()-start)/60))
        print('\n')

    ##np.savetxt('trainLossMean_list.csv', trainEP_loss_list, delimiter=',')
    ##np.savetxt('valLossMean_list.csv', valEP_loss_list, delimiter=',')

    '''
    ###############
    ##Test Process
    ###############
    '''
    predModel = Transformer().cuda()
    predModel.load_state_dict(torch.load(model_fromTrain))
    predModel.eval()

    '''train_obs, train_pred = [], []
    val_obs, val_pred = [], []'''
    test_obs, test_pred = [], []
    DrugSeqs_buf, TarSeqs_buf =[], []

    with torch.no_grad():
        '''for (DrugSeqs, TarSeqs, real_affi) in train_loader:
            DrugSeqs, TarSeqs, real_affi = DrugSeqs.cuda(), TarSeqs.cuda(), real_affi.cuda()
            pred_affi, enc_attnD0, enc_attnT0, enc_attnsD1, enc_attnsT1, enc_attnsD2, enc_attnsT2\
                                       = predModel(DrugSeqs, TarSeqs)
            train_obs.extend(real_affi.tolist())
            train_pred.extend(pred_affi.tolist())'''

        '''for (DrugSeqs, TarSeqs, real_affi) in val_loader:
            DrugSeqs, TarSeqs, real_affi = DrugSeqs.cuda(), TarSeqs.cuda(), real_affi.cuda()
            pred_affi, enc_attnD0, enc_attnT0, enc_attnsD1, enc_attnsT1, enc_attnsD2, enc_attnsT2\
                                       = predModel(DrugSeqs, TarSeqs)  # pred_affi: [batch_affini]
            val_obs.extend(real_affi.tolist())
            val_pred.extend(pred_affi.tolist())'''

        for (DrugSeqs, TarSeqs, real_affi) in test_loader:
            DrugSeqs, TarSeqs, real_affi = DrugSeqs.cuda(), TarSeqs.cuda(), real_affi.cuda()
            pred_affi, enc_attnD0, enc_attnT0, enc_attnsD1, enc_attnsT1, enc_attnsD2, enc_attnsT2\
                                       = predModel(DrugSeqs, TarSeqs)  # pred_affi: [batch_affini]
            ##test_obs.extend(real_affi.tolist())
            test_pred.extend(pred_affi.tolist())
            DrugSeqs_buf.extend(DrugSeqs.tolist())
            TarSeqs_buf.extend(TarSeqs.tolist())

    DrugSeqs_Int = DH.ClearZeros(DrugSeqs_buf)
    TarSeqs_Int = DH.ClearZeros(TarSeqs_buf)

    DrugSeqs_Str = DH.IntToToken(DrugSeqs_Int, DH.drugSeq_vocab)
    TarSeqs_Str = DH.IntToToken(TarSeqs_Int, DH.targetSeq_vocab)

    ##np.savetxt('test_obs.csv', test_obs, delimiter=',')
    #np.savetxt('DrugSeqs_buf.csv', DrugSeqs_buf, delimiter=',')
    np.savetxt('test_pred.csv', list(zip(DrugSeqs_Str,TarSeqs_Str,test_pred)), fmt='%s', delimiter=',')

    '''print('train_MSE:', '{:.3f}'.format(EM.get_MSE(train_obs, train_pred)))
    print('train_CI:', '{:.3f}'.format(EM.get_cindex(train_obs, train_pred)))
    print('train_rm2:', '{:.3f}'.format(EM.get_rm2(train_obs, train_pred)))'''

    '''print('\n')
    print('val_MSE:', '{:.3f}'.format(EM.get_MSE(val_obs, val_pred)))
    print('val_CI:', '{:.3f}'.format(EM.get_cindex(val_obs, val_pred)))
    print('val_rm2:', '{:.3f}'.format(EM.get_rm2(val_obs, val_pred)))'''

    '''print('\n')
    print('test_MSE:', '{:.3f}'.format(EM.get_MSE(test_obs, test_pred)))
    print('test_CI:', '{:.3f}'.format(EM.get_cindex(test_obs, test_pred)))
    print('test_rm2:', '{:.3f}'.format(EM.get_rm2(test_obs, test_pred)))'''