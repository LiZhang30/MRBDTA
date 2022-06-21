import math
import json
import pickle
import numpy as np
from collections import OrderedDict

#pad:0 -> Symbol that will fill in blank sequence if current batch data size is short than time steps
#vocab -> From DeepDTA
targetSeq_vocab = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7,
                   "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, "O": 13,
                   "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, "U": 19,
                   "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}
targetSeq_vocabSize = 25

#vocab -> from DeepDTA
#Canonical SMILE
drugSeq_vocab = { "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6,
			      ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12,
			      "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18,
			      "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
			      "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30,
			      "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36,
			      "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42,
			      "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48,
			      "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54,
			      "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
			      "t": 61, "y": 62}
drugSeq_vocabSize = 62

'''
# Iso SMILE
CharIsoSmiSet = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33,
                 "/": 34, ".": 2, "1": 35, "0": 3, "3": 36, "2": 4,
                 "5": 37, "4": 5, "7": 38, "6": 6, "9": 39, "8": 7,
                 "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46,
                 "M": 47, "L": 13, "O": 48, "N": 14, "P": 15, "S": 49,
                 "R": 16, "U": 50, "T": 17, "W": 51, "V": 18, "Y": 52,
                 "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59,
                 "h": 24, "m": 60, "l": 25, "o": 61, "n": 26, "s": 62,
                 "r": 27, "u": 63, "t": 28, "y": 64}
CharIsoSmiLen = 64
'''

#transfer token -> number
def LabelDT(drug_seqs, target_seqs, drugSeq_maxlen, targetSeq_maxLen):
    label_drugSeqs, label_targetSeqs = [], []
    drugSeq_truncated, targetSeq_truncated = drugSeq_maxlen, targetSeq_maxLen

    for i in range(len(drug_seqs)):

        label_drugSeqs.append([])
        if len(drug_seqs[i]) >= drugSeq_truncated:
            for j in range(drugSeq_truncated):
                label_drug = drugSeq_vocab[drug_seqs[i][j].split()[0]]
                label_drugSeqs[i].append(label_drug)
        else:
            for j in range(len(drug_seqs[i])):
                label_drug = drugSeq_vocab[drug_seqs[i][j].split()[0]]
                label_drugSeqs[i].append(label_drug)

        label_targetSeqs.append([])
        if len(target_seqs[i]) >= targetSeq_truncated:
            for j in range(targetSeq_truncated):
                label_traget = targetSeq_vocab[target_seqs[i][j].split()[0]]
                label_targetSeqs[i].append(label_traget)
        else:
            for j in range(len(target_seqs[i])):
                label_traget = targetSeq_vocab[target_seqs[i][j].split()[0]]
                label_targetSeqs[i].append(label_traget)

    return label_drugSeqs, label_targetSeqs

#get compile + protein pairs
def GetPairs(label_drugSeqs, label_targetSeqs):

    pairs = []
    for i in range(len(label_targetSeqs)):
        drugSeq = label_drugSeqs[i]
        targetSeq = label_targetSeqs[i]
        pairs.append(drugSeq+targetSeq) # avoid ‘extend()’

    return pairs

#load davis and kiba
def LoadData(path, logspance_trans):

    print("Read %s start" % path)

    ligands = json.load(open(path + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(path + "proteins.txt"), object_pairs_hook=OrderedDict)

    Y = pickle.load(open(path + "Y", "rb"), encoding='latin1')  # TODO: read from raw

    if logspance_trans:
        Y = -(np.log10(Y / (math.pow(10, 9))))

    XD = []
    XT = []

    for d in ligands.keys():
        XD.append(ligands[d])

    for t in proteins.keys():
        XT.append(proteins[t])

    return XD, XT, Y

#create samples for davis and kiba
def GetSamples(dataSet_name, drugSeqs, targetSeqs, affi_matrix):
    drugSeqs_buff, targetSeqs_buff, affiMatrix_buff= [], [], []
    if dataSet_name == 'davis':
        for i in range(len(drugSeqs)):
            for j in range(len(targetSeqs)):
                drugSeqs_buff.append(drugSeqs[i])
                targetSeqs_buff.append(targetSeqs[j])
                affiMatrix_buff.append(affi_matrix[i, j])

    if dataSet_name == 'kiba':
        for a in range(len(drugSeqs)):
            for b in range(len(targetSeqs)):
                if  ~(np.isnan(affi_matrix[a, b])):
                    drugSeqs_buff.append(drugSeqs[a])
                    targetSeqs_buff.append(targetSeqs[b])
                    affiMatrix_buff.append(affi_matrix[a, b])

    return drugSeqs_buff, targetSeqs_buff, affiMatrix_buff

#shuttle
def Shuttle(drug, target, affini):

    drug = np.array(drug)
    target = np.array(target)
    affini = np.array(affini)
    index = [i for i in range(len(affini))]
    np.random.shuffle(index)

    shttle_drug = drug[index]
    shttle_target = target[index]
    shttle_affini = affini[index]

    return shttle_drug, shttle_target, shttle_affini

#clear 0 values in list
def ClearZeros(inp_list):
    list_buf = []

    for i in range(len(inp_list)):
        list_buf.append([])

        for j in inp_list[i]:
            if j != 0:
                list_buf[i].append(j)

    return list_buf

#transfer number -> token
def IntToToken(inp_list,inp_dict):
    buf = []
    seqs = []
    single_seq = []

    for i in range(len(inp_list)):
        buf.append([])

        for j in inp_list[i]:
            for key, value in inp_dict.items():
                if value == j:
                    buf[i].append(key)

        single_seq = ''.join(buf[i])
        seqs.append(single_seq)

    return seqs
