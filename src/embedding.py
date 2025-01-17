#!/usr/bin/env python
# _*_coding:utf-8_*_

import re, sys, os, platform
import math
import argparse
import numpy as np
from collections import Counter
def CalculateKSCTriad(sequence, gap, features, AADict):
    res = []
    for g in range(gap + 1):
        myDict = {}
        for f in features:
            myDict[f] = 0

        for i in range(len(sequence)):
            if i + g + 1 < len(sequence) and i + 2 * g + 2 < len(sequence):
                if sequence[i] not in AADict or sequence[i + g + 1] not in AADict or sequence[i + 2 * g + 2] not in AADict:
                    print(f'{sequence[i]} {sequence[i+1]} {sequence[i + g + 1]} {sequence[i + 2 * g + 2]} ')
                fea = AADict[sequence[i]]+'.'+AADict[sequence[i + g + 1]]+'.'+AADict[sequence[i + 2 * g + 2]]
                myDict[fea] = myDict[fea] + 1

        maxValue, minValue = max(myDict.values()), min(myDict.values())
        for f in features:
            res.append((myDict[f] - minValue) / maxValue)

    return res

# extract features from the input sequence list
#read per file
def CalCJ(seqList, gap = 0, **kw):
    AAGroup = {
        'g1': 'AGV',
        'g2': 'ILFP',
        'g3': 'YMTS',
        'g4': 'HNQW',
        'g5': 'RK',
        'g6': 'DE',
        'g7': 'CX'  #SHS148k contains a subsequence 'XD' 
    }

    myGroups = sorted(AAGroup.keys())

    AADict = {}
    for g in myGroups:
        for aa in AAGroup[g]:
            AADict[aa] = g

    features = [f1 + '.'+ f2 + '.' + f3 for f1 in myGroups for f2 in myGroups for f3 in myGroups]

    encodings = []
    header = ['#', 'label']
    for f in features:
        header.append(f)
    encodings.append(header)

    result = []
    for seq in seqList:
        result.append(CalculateKSCTriad(seq, 0, features, AADict))
    return np.array(result,dtype=float)


#read per file
def CTriad(fastas, gap = 0, **kw):
    AAGroup = {
        'g1': 'AGV',
        'g2': 'ILFP',
        'g3': 'YMTS',
        'g4': 'HNQW',
        'g5': 'RK',
        'g6': 'DE',
        'g7': 'C'
    }

    myGroups = sorted(AAGroup.keys())

    AADict = {}
    for g in myGroups:
        for aa in AAGroup[g]:
            AADict[aa] = g

    features = [f1 + '.'+ f2 + '.' + f3 for f1 in myGroups for f2 in myGroups for f3 in myGroups]

    encodings = []
    header = ['#', 'label']
    for f in features:
        header.append(f)
    encodings.append(header)

    for i in fastas:
        name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
        code = [name, label]
        if len(sequence) < 3:
            print('Error: for "CTriad" encoding, the input fasta sequences should be greater than 3. \n\n')
            return 0
        code = code + CalculateKSCTriad(sequence, 0, features, AADict)
        encodings.append(code)

    return encodings

def CalPos(seqList, **kw):
    result = []
    for seq in seqList:
        length = len(seq)
        tmp = (1+1+length)*length/2.0
        result.append(tmp/length)
    return np.array(result,dtype=float).reshape((len(seqList),-1))


def CalAAC(seqList, **kw):
    AA = kw['order'] if 'order' in kw else 'ACDEFGHIKLMNPQRSTVWY'
    #AA = 'ARNDCQEGHILKMFPSTWYV'
    result = []
    for seq in seqList:
        tmp = []
        count = Counter(seq)
        for key in count:
            count[key] = count[key]/len(seq)
        for aa in AA:
            tmp.append(count[aa])
        result.append(tmp)
    return np.array(result,dtype=float)


def CalDPC(seqList:list, **kw):
    AA = kw['order'] if 'order' in kw else 'ACDEFGHIKLMNPQRSTVWY'
    result = []
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for seq in seqList:
        tmpCode = [0] * 400
        for j in range(len(seq) - 2 + 1):
            tmpCode[AADict[seq[j]] * 20 + AADict[seq[j+1]]] = tmpCode[AADict[seq[j]] * 20 + AADict[seq[j+1]]] +1
        if sum(tmpCode) != 0:
            tmpCode = [i/sum(tmpCode) for i in tmpCode]
        result.append(tmpCode)
    return np.array(result,dtype=float)

def Rvalue(aa1, aa2, AADict, Matrix):
    return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)


def CalPAAC(seqList, lambdaValue=20, w=0.05, **kw):

    for seq in seqList:
        if len(seq) < (lambdaValue + 1):
            print('Error: all the sequence length should be larger than the lambdaValue+1: ' + str(lambdaValue + 1) + '\n\n')
            print(f'seq len: {len(seq)}')
            return 0

    fName = "src/PAACData.txt"
    if not os.path.exists(fName):
        print(f'error, {fName} not found')
        return 0

    with open(fName) as f:
        records = f.readlines()

    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    AAProperty = []
    AAPropertyNames = []
    for i in range(1, len(records)):
        array = records[i].rstrip().split() if records[i].rstrip() != '' else None
        AAProperty.append([float(j) for j in array[1:]])
        AAPropertyNames.append(array[0])

    AAProperty1 = []
    for i in AAProperty:
        meanI = sum(i) / 20
        fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
        AAProperty1.append([(j - meanI) / fenmu for j in i])

    result = []


    for seq in seqList:
        theta = []
        code = []
        for n in range(1, lambdaValue + 1):
            theta.append(
                sum([Rvalue(seq[j], seq[j + n], AADict, AAProperty1) for j in range(len(seq) - n)]) / (
                    len(seq) - n))
        myDict = {}
        for aa in AA:
            myDict[aa] = seq.count(aa)
        code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
        code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
        result.append(code)
    return np.array(result,dtype=float)

def CalCTDT(seqList, **kw):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity':        'LIFWCMVY',
        'polarizability':  'GASDT',
        'charge':          'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess':   'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity':        'PATGS',
        'polarizability':  'CPNVEQIL',
        'charge':          'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess':   'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity':        'HQRKNED',
        'polarizability':  'KMHFRYW',
        'charge':          'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess':   'MSPTHY'
    }

    groups = [group1, group2, group3]
    property = (
    'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
    'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
    'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    result = []
    
    for seq in seqList:
        code = []
        aaPair = [seq[j:j + 2] for j in range(len(seq) - 1)]
        for p in property:
            c1221, c1331, c2332 = 0, 0, 0
            for pair in aaPair:
                if (pair[0] in group1[p] and pair[1] in group2[p]) or (pair[0] in group2[p] and pair[1] in group1[p]):
                    c1221 = c1221 + 1
                    continue
                if (pair[0] in group1[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group1[p]):
                    c1331 = c1331 + 1
                    continue
                if (pair[0] in group2[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group2[p]):
                    c2332 = c2332 + 1
            code = code + [c1221/len(aaPair), c1331/len(aaPair), c2332/len(aaPair)]
        result.append(code)
    return np.array(result,dtype=float)

def CTDT(fastas, **kw):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity':        'LIFWCMVY',
        'polarizability':  'GASDT',
        'charge':          'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess':   'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity':        'PATGS',
        'polarizability':  'CPNVEQIL',
        'charge':          'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess':   'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity':        'HQRKNED',
        'polarizability':  'KMHFRYW',
        'charge':          'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess':   'MSPTHY'
    }

    groups = [group1, group2, group3]
    property = (
    'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
    'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
    'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    encodings = []
    header = ['#', 'label']
    for p in property:
        for tr in ('Tr1221', 'Tr1331', 'Tr2332'):
            header.append(p + '.' + tr)
    encodings.append(header)

    for i in fastas:
        name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
        code = [name, label]
        aaPair = [sequence[j:j + 2] for j in range(len(sequence) - 1)]
        for p in property:
            c1221, c1331, c2332 = 0, 0, 0
            for pair in aaPair:
                if (pair[0] in group1[p] and pair[1] in group2[p]) or (pair[0] in group2[p] and pair[1] in group1[p]):
                    c1221 = c1221 + 1
                    continue
                if (pair[0] in group1[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group1[p]):
                    c1331 = c1331 + 1
                    continue
                if (pair[0] in group2[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group2[p]):
                    c2332 = c2332 + 1
            code = code + [c1221/len(aaPair), c1331/len(aaPair), c2332/len(aaPair)]
        encodings.append(code)
    return encodings

import operator
import os
################################################
# fix the random see value so the results are re-producible
seed_value = 7

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
###############################################

import csv
import logging
from pandas import DataFrame
import time
import sys
import math
Dict_3mer_to_100vec={}

# dim: delimiter
def get_3mer_and_np100vec_from_a_line(line, dim):
    np100 = []
    line = line.rstrip('\n').rstrip(' ').split(dim)
    three_mer = line.pop(0)
    np100 += [float(i) for i in line]
    np100 = np.asarray(np100)
    return three_mer, np100

def LoadPro2Vec(fName):
    f = open(fName, "r")
    index = 0
    while True:
        line = f.readline().replace('"', '')
        if not line:
            break
        three_mer, np100vec = get_3mer_and_np100vec_from_a_line(line, '\t')
        Dict_3mer_to_100vec[three_mer] = np100vec

        index += 1
    #print ("total number of unique 3mer is ", index)

def GetFeature(ThreeMer, Feature_dict):
    if (ThreeMer not in Feature_dict):
        print("[warning]: Feature_dict can't find ", ThreeMer, ". Returning 0")
        return 0
    else:
        return Feature_dict[ThreeMer]

def RetriveFeatureFromASequence(seq, Feature_dict):
    seq = seq.rstrip('\n').rstrip(' ')
    assert (len(seq) >= 3)
    Feature = []
    for index, item in enumerate(seq):
        sta = index - 1
        end = index + 1
        if ((sta < 0) or (end >= len(seq))):
            Feature.append(Feature_dict["<unk>"])
        else:
            Feature.append(GetFeature(seq[sta:sta+3], Feature_dict))
    return Feature

def CalProtVec(seqList): 
    result = []
    LoadPro2Vec("src/protVec_100d_3grams.csv")
    for key,value in Dict_3mer_to_100vec.items():
        Dict_3mer_to_100vec[key] = np.sum(value)
    # print(Dict_3mer_to_100vec["AAA"])
    max_key = max(Dict_3mer_to_100vec.keys(), key=(lambda k: Dict_3mer_to_100vec[k]))
    min_key = min(Dict_3mer_to_100vec.keys(), key=(lambda k: Dict_3mer_to_100vec[k]))
    max_value = Dict_3mer_to_100vec[max_key]
    min_value = Dict_3mer_to_100vec[min_key]

    for key,value in Dict_3mer_to_100vec.items():
        Dict_3mer_to_100vec[key] = (Dict_3mer_to_100vec[key] - min_value) / (max_value - min_value)
    for seq in seqList:
        vec = RetriveFeatureFromASequence(seq, Dict_3mer_to_100vec)
        result.append( sum(vec) / (len(vec)-2) )

    result = np.array(result,dtype='float').reshape(len(seqList),-1)
    #print(f'ProtVec shape: {result.shape}')
    return result

