import numpy as np
import random
import torch
import os
import math
import time
import Models
from collections import defaultdict 
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
def sorted_pair(id1,id2):
	if id1<id2:
		return [id1,id2]
	return [id2,id1]

def check_files_exist(fList:list):
	for fName in fList:
		if not os.path.isfile(fName):
			print(f'error,input file {fName} does not exist')
			exit()
	return
#encode binary vecotr into single num
def encode_inter(vec,num=7):
	result = 0
	for i,v in enumerate(vec):
		result += v*pow(2,num-1-i)
	return result
#		print([1,1,0,0,0,1,0])
#		a = utils.encode_inter([1,1,0,0,0,1,0])
def decode_inter(value,num=7):
	result = []
	for i in range(num):
		tmp = pow(2,num-1-i)
		v = math.floor(value/tmp)
		#print(f'{value} {tmp} {v}')
		value -= v*tmp
		result.append(v)
	return np.array(result,dtype=float)
def sort_dir_by_value(dict): #note: decesending order
	keys = list(dict.keys())
	values = list(dict.values())
	sorted_value_index = np.argsort(values)[::-1]
	sorted_dict = {keys[i]: values[i] for i in sorted_value_index}
	return sorted_dict


class Metrictor_PPI:
	def __init__(self, pre_y, truth_y, is_binary=False):
		self.TP = 0
		self.FP = 0
		self.TN = 0
		self.FN = 0

		if is_binary:
			length = pre_y.shape[0]
			for i in range(length):
				if pre_y[i] == truth_y[i]:
					if truth_y[i] == 1:
						self.TP += 1
					else:
						self.TN += 1
				elif truth_y[i] == 1:
					self.FN += 1
				elif pre_y[i] == 1:
					self.FP += 1
			self.num = length

		else:
			N, C = pre_y.shape
			for i in range(N):
				for j in range(C):
					if pre_y[i][j] == truth_y[i][j]:
						if truth_y[i][j] == 1:
							self.TP += 1
						else:
							self.TN += 1
					elif truth_y[i][j] == 1:
						self.FN += 1
					elif truth_y[i][j] == 0:
						self.FP += 1
			self.num = N * C
	
	def append_result(self,path='test.txt',e=None,train_loss=0.0,valid_loss=0.0):
		self.acc = (self.TP + self.TN) / (self.num + 1e-10)
		self.pre = self.TP / (self.TP + self.FP + 1e-10)
		self.recall = self.TP / (self.TP + self.FN + 1e-10)
		self.microF1 = 2 * self.pre * self.recall / (self.pre + self.recall + 1e-10)
		record = f'epoch {e},acc {self.acc:.2f}, microF1 {self.microF1:.4f}, precision {self.pre:.2f},recall {self.recall:.2f}, train loss {train_loss:.4f}, valid loss {valid_loss:.4f}'#,loss {loss}'
		with open(path,'a') as f:
			f.write(record+'\n')
		return record

	def show_result(self, is_print=False, file=None):
		self.Accuracy = (self.TP + self.TN) / (self.num + 1e-10)
		self.Precision = self.TP / (self.TP + self.FP + 1e-10)
		self.Recall = self.TP / (self.TP + self.FN + 1e-10)
		self.F1 = 2 * self.Precision * self.Recall / (self.Precision + self.Recall + 1e-10)
		if is_print:
			print_file("Accuracy: {}".format(self.Accuracy), file)
			print_file("Precision: {}".format(self.Precision), file)
			print_file("Recall: {}".format(self.Recall), file)
			print_file("F1-Score: {}".format(self.F1), file)


