from collections import defaultdict 
import numpy as np
import random
import utils
import time
import embedding
import re 
import os
import torch
import json
import math
import torch_geometric
from gensim.models import Word2Vec
from torch.utils.data import DataLoader
import dgl
import networkx as nx
import Models
labelDir = {'reaction':0, 'binding':1, 'ptmod':2, 'activation':3,'inhibition':4, 'catalysis':5, 'expression':6}
amino_list = ['A','G','V','I','L','F','P','Y','M','T','S','H','N','Q','W','R','K','D','E','C']

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
class PPIData(object):
	def __init__(self,args,class_num = 7):
		self.args,self.seqPath, self.relPath,self.mapPath = args,args.i1,args.i2,args.i4
		self.seqs, self.name2index = readSeqs(self.seqPath)
		self.seqsNum = len(self.name2index)
		#pari list: 'name1_name2', interList: binary vector represnts type, edgeList [id1,id2] 
		self.pairList,self.interList,self.pair2index,self.neighIndex,self.edgeList  = readInteraction(self.relPath,self.name2index)
		prot_node, prot_edge, prot_kneg = read_structure_data(args.i4,args.i)
		#self.prot_strut_data = ProteinDatasetDGL(prot_node, prot_edge, prot_kneg)

		self.prot_encode = torch.tensor(seqEncoding(self.seqs,args.L),dtype=torch.float)
		self.prot_embed = self.embed_prot(prot_node, prot_edge, prot_kneg)
		#self.prot_embed = torch.cat([self.embed_prot(prot_node, prot_edge, prot_kneg),pcFeatures],dim=1)
		self.split_dict = {}
		if args.m == 'bfs':
			self.split_dict['train_index'], self.split_dict['valid_index'] = self.split_dataset_bfs(test_percentage=0.2,edgeList=self.edgeList)
		elif args.m == 'dfs':
			self.split_dict['train_index'], self.split_dict['valid_index'] = self.split_dataset_dfs(test_percentage=0.2,edgeList=self.edgeList)
		elif args.m == 'random':
			self.split_dict['train_index'], self.split_dict['valid_index'] = self.split_dataset_random(test_percentage=0.2)			
		elif args.m == 'read':
			self.split_dict['train_index'], self.split_dict['valid_index'] = self.read_test_set(args.i3)		
		if args.m != 'read' or args.pr != 0.0:
			self.save_valid_set(args)
		if args.pr != 0.0:
			self.split_dict['train_index'] = remove_link(self.split_dict['train_index'],args.pr)
		
		edge_index1 = create_edge_indices(self.edgeList,self.interList,len(self.interList[0]))#the edge list for each individual graph
		edge_index1 = [torch.tensor(e,dtype=torch.long).transpose(0,1) for e in edge_index1]
		sparse_adj1 = [create_sparse_tensor(e,self.seqsNum) for e in edge_index1]

		edge_index2 = torch.tensor(self.edgeList, dtype=torch.long).transpose(0,1) #the overall edge list

		sparse_adj2 = create_sparse_tensor(edge_index2,self.seqsNum)
		self.data = torch_geometric.data.Data(embed1=self.prot_embed,encode1=self.prot_encode,edge1=edge_index1,edge2=edge_index2,edge_attr = torch.tensor(self.interList, dtype=torch.long),sparse_adj1=sparse_adj1,sparse_adj2=sparse_adj2) 
		self.data.train_mask = self.split_dict['train_index']
		self.data.val_mask = self.split_dict['valid_index']

	def get_edge_info():
		return self.name2index,self.edgeList

	def analyze_hiearchical(self,aly_data,output='h_level.csv'):
		index2name = {v: k for k, v in self.name2index.items()} 
		aly_data = aly_data.cpu().detach().numpy()
		hierarchicalLevel = [ np.linalg.norm(row) for row in aly_data ]

		nodeDegree = np.zeros((len(self.name2index)),float)
		edgeLevel = np.zeros((len(self.name2index)),float)
		linkNum = np.zeros((len(self.name2index)),float)
		avgDegree = []

		for i,edge in enumerate(self.edgeList):
			nodeDegree[edge[0]] += np.sum(self.interList[i])
			nodeDegree[edge[1]] += np.sum(self.interList[i])
			linkNum[edge[0]] += 1.0
			linkNum[edge[1]] += 1.0


		for i in range(len(self.name2index)):
			avgDegree.append(nodeDegree[i]/linkNum[i])

		with open(output,'w') as f:
			f.write('name,hierarchical level,avg node degrees,node degrees,edgeNum\n')
			for i in range(len(self.name2index)):
				f.write(f'{index2name[i]},{hierarchicalLevel[i]},{avgDegree[i]},{nodeDegree[i]},{linkNum[i]}\n')

		return


	def embed_prot(self,prot_node, prot_edge, prot_kneg):
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		prot_strut_data = ProteinDatasetDGL(prot_node, prot_edge, prot_kneg)

		param = json.loads(open("src/param_configs.json", 'r').read())['STRING']['bfs']
		vae_dataloader = DataLoader(prot_strut_data, batch_size=512, shuffle=True, collate_fn=collate)
		vae_model = Models.CodeBook(param, DataLoader(prot_strut_data, batch_size=512, shuffle=False,collate_fn=collate)).to(device)#collate_fn=collate
		vae_model.load_state_dict(torch.load('src/vae_model.ckpt'))
		prot_embed = vae_model.Protein_Encoder.forward(vae_model.vq_layer)
		del vae_model
		torch.cuda.empty_cache()
		print('finish prot embedding')
		return prot_embed

# protein_data, ppi_g, ppi_list, labels, ppi_split_dict = load_data(param['dataset'], param['split_mode'], param['seed'])
	def construct_heterograph(self,prot_node, prot_edge, prot_kneg):
		self.prot_strut_data = []
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		for i in range(len(prot_edge)):
			prot_seq = []
			for j in range(prot_node[i].shape[0]-1):
				prot_seq.append((j, j+1))
				prot_seq.append((j+1, j))

			# prot_g = dgl.graph(prot_edge[i]).to(device)
			prot_g = dgl.heterograph({('amino_acid', 'SEQ', 'amino_acid') : prot_seq, 
									  ('amino_acid', 'STR_KNN', 'amino_acid') : prot_kneg[i],
									  ('amino_acid', 'STR_DIS', 'amino_acid') : prot_edge[i]}).to(device)
			prot_g.ndata['x'] = torch.FloatTensor(prot_node[i]).to(device)
			self.prot_strut_data.append(prot_g)

		return
		#self.s2p =  #map id of STRING database to uniprot
		#self.seqFeature = seqEmbedding(self.seqs)

	def save_valid_set(self,args):
		index2pair = {v: k for k, v in self.pair2index.items()} 
		if args.m != 'read' or args.i3[-4:] == 'data':
			valid_pair = []
			train_pair = []
			for index in self.split_dict['valid_index']:
				valid_pair.append(index2pair[index])
			for index in self.split_dict['train_index']:
				train_pair.append(index2pair[index])
			result = {'valid_index':valid_pair,'train_index':train_pair,'seqPath':self.seqPath,'interPath':self.relPath}
			jsobj = json.dumps(result)
			with open(args.o+'.json', 'w') as f:
				f.write(jsobj)

		return

	def split_dataset_bfs(self,node_to_edge_index=None,test_percentage=0.2,edgeList=None,src_path=None): #list of interaction, percentage of
		if not node_to_edge_index:
			node_to_edge_index = defaultdict(list)
			for i in range(len(edgeList)):
				id1, id2 = edgeList[i][0], edgeList[i][1]
				node_to_edge_index[id1].append(i)
				node_to_edge_index[id2].append(i)

		node_num = len( node_to_edge_index.keys() )
		test_size = int(len(edgeList) * test_percentage)
		test_set = []
		queue = []
		visited = []

		random_index = random.randint(0, node_num-1)
		while len( node_to_edge_index[random_index] ) > 5:
			random_index = random.randint(0, node_num-1)
		queue.append(random_index)
		print(f'root level {len( node_to_edge_index[random_index])}')
		count = 0
		#print(node_to_edge_index[random_index])

		while len(test_set) < test_size:
			if len(queue) == 0:
				print('bfs split meet root level 0, terminate process')
				exit()
				# while(random_index in visited):
				# 	random_index = random.randint(0, node_num-1)
				# queue.append(random_index)
			cur_node = queue.pop(0) 
			visited.append(cur_node)
			for edge_index in node_to_edge_index[cur_node]:
				if edge_index not in test_set:
					test_set.append(edge_index)
					id1,id2 = edgeList[edge_index][0],edgeList[edge_index][1]
					next_node = id1
					if id1 == cur_node:
						next_node = id2
					if next_node not in visited and next_node not in queue:
						queue.append(next_node)
				else:
					continue
		
		#test_set = np.array(test_set,dtype=int)
		training_set = self.construct_training_set(test_set)
		return training_set,test_set

	def split_dataset_dfs(self,node_to_edge_index=None,test_percentage=0.2,edgeList=None,src_path=None):
		if not node_to_edge_index:
			node_to_edge_index = defaultdict(list)
			for i in range(len(edgeList)):
				id1, id2 = edgeList[i][0], edgeList[i][1]
				node_to_edge_index[id1].append(i)
				node_to_edge_index[id2].append(i)

		node_num = len( node_to_edge_index.keys() )
		test_size = int(len(edgeList) * test_percentage)
		test_set = []
		stack = []
		visited = []

		random_index = random.randint(0, node_num-1)
		while len( node_to_edge_index[random_index] ) > 5:
			random_index = np.random.randint(0, node_num-1)
		print(f'random index {random_index},root level {len( node_to_edge_index[random_index])}')

		stack.append(random_index)

		while(len(test_set) < test_size):
			if len(stack) == 0:
				print('dfs split meet root level 0')
				exit()

			cur_node = stack[-1]
			if cur_node in visited:
				flag = True
				for edge_index in node_to_edge_index[cur_node]:
					if flag:
						id1,id2 = edgeList[edge_index][0],edgeList[edge_index][1]
						next_node = id1 if id2 == cur_node else id2
						if next_node in visited:
							continue
						else:
							stack.append(next_node)
							flag = False
					else:
						break
				if flag:
					stack.pop()
				continue
			else:
				visited.append(cur_node)
				for edge_index in node_to_edge_index[cur_node]:
					if edge_index not in test_set:
						test_set.append(edge_index)
		#test_set = np.array(test_set,dtype=int)
		training_set = self.construct_training_set(test_set)
		return training_set,test_set

	def split_dataset_random(self,test_percentage=0.2):
		all_indices = [i for i in range(len(self.edgeList))]
		test_size = int(test_percentage*len(all_indices))
		test_set = random.sample(all_indices,test_size)
		training_set = self.construct_training_set(test_set)
		return training_set,test_set


	def construct_training_set(self,test_indices):
		all_indices = [i for i in range(len(self.edgeList))]
		training_indices = list( set(all_indices).difference( set(test_indices) ) )
		assert len(self.edgeList) == (len(training_indices)+len(test_indices)), "error, the size of training and test set doesn't match"		
		return training_indices

	# 	return training_set,test_set
	def read_test_set(self,save_path):
		valid_index = []
		if save_path[-4:] == 'json':
			with open(save_path, 'r') as f:
				self.ppi_split_dict = json.load(f)
			print(len(self.ppi_split_dict['valid_index']))
			print(len(self.edgeList))

			for pair in self.ppi_split_dict['valid_index']:
				pair = pair.split('__')

				newPair = utils.sorted_pair(pair[0],pair[1])
				newPair = newPair[0] +'__'+ newPair[1]
				if self.pair2index[newPair] == -1:
					print(f'error, unfound pair {newPair}')
					exit()
				valid_index.append(self.pair2index[newPair])

			if 'train_index' in self.ppi_split_dict.keys():
				train_index = []
				for pair in self.ppi_split_dict['train_index']:
					pair = pair.split('__')
					newPair = utils.sorted_pair(pair[0],pair[1])
					newPair = newPair[0] +'__'+ newPair[1]
					if self.pair2index[newPair] == -1:
						print(f'error, unfound pair {newPair}')
						exit()
					train_index.append(self.pair2index[newPair])
				return train_index, valid_index				

		elif save_path[-4:] == 'data':
			with open(save_path, 'r') as f:
				fName1 = f.readline()
				fName2 = f.readline()
				lines = f.readlines()
				for line in lines:
					pair = line.strip().split('\t')
					newPair = utils.sorted_pair(pair[0],pair[1])
					newPair = newPair[0] +'__'+ newPair[1]
					if self.pair2index[newPair] == -1:
						print(f'error, unfound pair {newPair}')
						exit()
					valid_index.append(self.pair2index[newPair])
		#print(len(valid_index))
		# if len(valid_index) < 4000:
		# 	exit()
		return self.construct_training_set(valid_index),valid_index

	def compute_centrality(self,output='center.csv'):
		index2name = {v: k for k, v in self.name2index.items()} 
		G=nx.Graph()
		for i in range(self.seqsNum):
			G.add_node(i)
		for i,edge in enumerate(self.edgeList):
			G.add_edge(*edge)		
		bc = nx.betweenness_centrality(G) 

		cc = nx.closeness_centrality(G)

		with open(output,'w') as f:
			f.write('name,betweenness_centralitycloseness_centrality\n')
			for i in range(len(self.name2index)):
				f.write(f'{index2name[i]},{bc[i]},{cc[i]}\n')
		exit()
		return




def readInteraction(relPath,name2index,level=3):
	labelNum = len(labelDir)
	#pairList and pair2index can be considered as inverse index
	pairList,interList = [],[]
	pair2index = defaultdict(lambda:-1) 
	utils.check_files_exist([relPath])
	edgeList = []
	with open(relPath,'r') as file:
		header  = file.readline() #assume the first line is header
		lines = file.readlines() 
		for line in lines:
			tmp=re.split(',|\t',line.strip('\n'))
			protein1,protein2,mode,is_dir = tmp[0], tmp[1], labelDir[tmp[2]], tmp[4]
			if name2index[protein1] == -1 or name2index[protein2] == -1:
				print(f'error, unrecognzied sequence {tmp}')
				exit()
			newPair = utils.sorted_pair(protein1,protein2)
			newIndex = [name2index[newPair[0]],name2index[newPair[1]]]
			newPair = newPair[0] +'__'+ newPair[1]
			if pair2index[newPair] == -1:
				pair2index[newPair] = len(pairList)
				pairList.append(newPair)
				interList.append(np.zeros((labelNum,),dtype=int))
				edgeList.append(newIndex)
			interList[pair2index[newPair]][mode] = 1

	adj = []
	for i in range(len(name2index)):
		adj.append([])
		for j in range(labelNum):
			adj[i].append([])

	for i in range(len(interList)):	
		interaction, pair = interList[i],edgeList[i]	
		for j,inter in enumerate(interaction):
			if inter == 1:
				adj[pair[0]][j].append(pair[1])
				adj[pair[1]][j].append(pair[0]) 

	return pairList,interList,pair2index,adj,edgeList

def readSeqs(seqPath):
	name2index = defaultdict(lambda:-1)
	seqs = []
	utils.check_files_exist([seqPath])
	count = 0
	with open(seqPath,'r') as file:
		lines = file.readlines()
		for line in lines:
			tmp=re.split(',|\t',line.strip('\n'))
			if name2index[tmp[0]] == -1:
				seqs.append(tmp[1])
				name2index[tmp[0]] = count
				count += 1
	#replace unrecognzied aminoa acid
	for i,seq in enumerate(seqs):
		tmp = []
		for s in seq:
			if s not in amino_list:
				tmp.append(s)
		if len(tmp) > 0:
			for t in tmp:
				seqs[i] = seqs[i].replace(t,'')	
	return seqs,name2index


def read_structure_data(symbol,src_file,folder='features'):
	vae_path = 'src/vae_model.ckpt'

	if symbol is None:
		symbol = re.split('/|[.]',src_file)[-2]
		prefix = f'{folder}/{symbol}'
	else:
		prefix = symbol

	print(f'=========read structure data with prefix {prefix}=======')
	suffix = ['node','edge','kneg']
	for s in suffix:
		fName = f'{prefix}_{s}.pt'
		print(fName)
		if not os.path.isfile(fName):
			print(f'error, {fName} not found')
			exit()

	prot_node = torch.load(f'{prefix}_{suffix[0]}.pt')
	prot_edge = torch.load(f'{prefix}_{suffix[1]}.pt')
	prot_kneg = torch.load(f'{prefix}_{suffix[2]}.pt')
	print(f'=====finish reading structure data=====')

	return prot_node, prot_edge, prot_kneg


def generate_structure_feature(args,atom_file = 'src/atom_feature.txt'):

	symbol = args.o

	seq_file = args.i1
	int_file = args.i2
	folder = args.sf
	threshold = 8

	file_list = []
	not_found = []
	with open(seq_file,'r') as f:
		lines = f.readlines()
		for line in lines:
			fName = folder + re.split('\t| |,',line)[0] + '.pdb'
			if not os.path.isfile(fName):
				print(f'error, structure file {fName} not found')
				exit()
				#not_found.append(re.split('\t| |,',line)[0])				
			if os.path.isfile(fName):
				file_list.append(fName)
	# with open('add.txt','w') as f:
	# 	for index in not_found:
	# 		f.write(f'{index}\n')
	# print(len(not_found))

	atom_dict = {}
	with open(atom_file,'r') as file:
		for line in file:
			line = re.split(' |\t',line)
			atom_dict[line[0]] = np.array([float(x) for x in line[1:]])

	node_list = [] #the (sequential) atom feature of each protein
	r_edge_list = [] # the connection between atoms of a protein, considered as graph structure
	k_edge_list = [] # the k nearset atom of each atom

	start_time = time.time()
	for fName in file_list:
		r_contacts, k_contacts, atoms = pdb_to_cm(fName, threshold)
		x = np.zeros((len(atoms),7))
		for i in range(len(atoms)):
			x[i] = atom_dict[ atoms[i] ]
		node_list.append(x) 
		r_edge_list.append(r_contacts)
		k_edge_list.append(k_contacts)

	print(f"---{len(file_list)} proteins, {time.time() - start_time} seconds for structure feature construction---")
	

	#os.system(f'mkdir -p {folder}')
	torch.save(node_list,f'{symbol}_node.pt')
	torch.save(r_edge_list,f'{symbol}_edge.pt')
	torch.save(k_edge_list,f'{symbol}_kneg.pt')
	#np.save(f'{folder}/{symbol}_edge.npy',np.array(r_edge_list))
	#np.save(f'{folder}/{symbol}_kneg.npy',np.array(k_edge_list))

	print(f'feature files saved to pt file with prefix {symbol}')
	



def pdb_to_cm(fName, threshold):
	atoms, ajs = read_atoms(fName)
	r_contacts = compute_contacts(atoms, threshold)
	k_contacts = knn(atoms)
	return r_contacts, k_contacts, ajs

def read_atoms(fName, chain="."):
	pattern = re.compile(chain)
	atoms = []
	ajs = []
	with open(fName,'r') as f:
		lines = f.readlines()
		for line in lines:
			line = line.strip()
			if line.startswith("ATOM"):
				type = line[12:16].strip()
				chain = line[21:22]
				if type == "CA" and re.match(pattern, chain):
					x = float(line[30:38].strip())
					y = float(line[38:46].strip())
					z = float(line[46:54].strip())
					ajs_id = line[17:20]
					atoms.append((x, y, z))
					ajs.append(ajs_id)           
	return atoms, ajs

def compute_contacts(atoms, threshold):
	contacts = []
	for i in range(len(atoms)-2):
		for j in range(i+2, len(atoms)):
			if dist(atoms[i], atoms[j]) < threshold:
				contacts.append((i, j))
				contacts.append((j, i))
	return contacts

def knn(atoms, k=5):
	x = np.zeros((len(atoms), len(atoms)))
	for i in range(len(atoms)):
		for j in range(len(atoms)):
			x[i, j] = dist(atoms[i], atoms[j])
	index = np.argsort(x, axis=-1)
	
	contacts = []
	for i in range(len(atoms)):
		num = 0
		for j in range(len(atoms)):
			if index[i, j] != i and index[i, j] != i-1 and index[i, j] != i+1:
				contacts.append((i, index[i, j]))
				num += 1
			if num == k:
				break

	return contacts


def dist(p1, p2):
	dx = p1[0] - p2[0]
	dy = p1[1] - p2[1]
	dz = p1[2] - p2[2]
	return math.sqrt(dx**2 + dy**2 + dz**2)



class ProteinDatasetDGL(torch.utils.data.Dataset):
	def __init__(self,  prot_node, prot_r_edge, prot_k_edge):
		self.prot_graph_list = []
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		for i in range(len(prot_r_edge)):
			prot_seq = []
			for j in range(prot_node[i].shape[0]-1):
				prot_seq.append((j, j+1))
				prot_seq.append((j+1, j))

			# prot_g = dgl.graph(prot_edge[i]).to(device)
			prot_g = dgl.heterograph({('amino_acid', 'SEQ', 'amino_acid') : prot_seq, 
									  ('amino_acid', 'STR_KNN', 'amino_acid') : prot_k_edge[i],
									  ('amino_acid', 'STR_DIS', 'amino_acid') : prot_r_edge[i]}).to(device)
			prot_g.ndata['x'] = torch.FloatTensor(prot_node[i]).to(device)

			self.prot_graph_list.append(prot_g)

		# with open("../data/processed_data/{}_protein_graphs.pkl".format(dataset), "wb") as tf:
		#     pickle.dump(self.prot_graph_list, tf)

	def __len__(self):
		return len(self.prot_graph_list)

	def __getitem__(self, idx):
		return self.prot_graph_list[idx]

def collate(samples):
    return dgl.batch(samples)

def create_edge_indices(edgeList,interList,class_num = 7):
	result = []
	for i in range(class_num):
		result.append([])
	for i,inter in enumerate(interList):
		for j in range(class_num):
			if inter[j] == 1:
				result[j].append(edgeList[i])
	return result

def create_sparse_tensor(edges,node_num):
	values = torch.ones(edges.shape[1])
	shape = torch.Size((node_num,node_num))
	return torch.sparse.FloatTensor(edges,values,shape)

def seq_padding(seqList,maxLen=512):
	paddedSeq = []
	seqNum = len(seqList)
	for i in range(seqNum):
		if len(seqList[i]) >= maxLen:
			paddedSeq.append(seqList[i][0:maxLen])
		else:
			paddedSeq.append(seqList[i])
			paddedSeq[i] += ' '*(maxLen-len(seqList[i]))
	return paddedSeq

def seqEncoding(seqs:list,maxLen=512,modelPath='src/vec5_CTC.txt',w2vPath = 'src/wv_swissProt_size_20_window_16.model',PSSM=None):
	result = []
	paddedSeq = seq_padding(seqs,maxLen)

	fMatrix = embedding.CalAAC(seqs)
	#fMatrix = np.concatenate((fMatrix,embedding.CalPAAC(seqs)),axis=1)  
	fMatrix = np.concatenate((fMatrix,embedding.CalCTDT(seqs)),axis=1)
	# v1 = w2v(paddedSeq,w2vPath,20,maxLen)
	# v2 = word2type(paddedSeq,modelPath,maxLen)
	# result = np.concatenate((v1,v2),axis=2)
	return fMatrix

def word2type(paddedSeq,modelPath=None,maxLen=512):
	seqNum = len(paddedSeq)
	model = {}
	size = None
	with open(modelPath,'r') as file:
		for line in file:
			line = re.split(' |\t',line)
			model[line[0]] = np.array([float(x) for x in line[1:]])
			if size is None:
				size = len(line[1:])
	
	result = []
	for i in range(seqNum):
		tmp = []
		for j in range(maxLen):
			if paddedSeq[i][j] == ' ':
				tmp.append(np.zeros(size)) 
			else:
				tmp.append(model[paddedSeq[i][j]])
		result.append(tmp)
	result = np.array(result,dtype=float)
	return result


def w2v(paddedSeq ,modelPath=None,size=20,maxLen=512):
	model = Word2Vec.load(modelPath)
	result = []
	seqNum =  len(paddedSeq)
	size = len(model.wv[paddedSeq[0][0]])
	print(f'embedding size {size}')
	print(f'padding with maxLen {maxLen}')	

	for i in range(seqNum):
		tmp = []
		for j in range(maxLen):
			if paddedSeq[i][j] == ' ':
				tmp.append(np.zeros(size))
			else:
				tmp.append(model.wv[paddedSeq[i][j]])
		result.append(np.array(tmp))
	return np.array(result)


# def atom2vec(atmosList,modelPath='process/vec5_atom.txt'):
# 	with open(modelPath,'r') as file:
# 		for line in file:
# 			line = re.split(' |\t',line)
# 			model[line[0]] = np.array([float(x) for x in line[1:]])
# 			if size is None:
# 				size = len(line[1:])
# 	result = []
# 	for atmos in atmoList:
# 		tmp = []
# 		for atom in atmos:
# 			tmp.append(model[atom])
# 		result.append(tmp)
# 	return result


# def seqEmbedding(seqs:list,w2vPath=None,PSSMPath=None):
# 	fMatrix = embedding.CalAAC(seqs)
# 	fMatrix = np.concatenate((fMatrix,embedding.CalCJ(seqs)),axis=1) #dimension 343
# 	#fMatrix = np.concatenate((fMatrix,embedding.CalDPC(seqs)),axis=1) #dimension 400
# 	fMatrix = np.concatenate((fMatrix,embedding.CalPAAC(seqs)),axis=1)  #dimension 50
# 	fMatrix = np.concatenate((fMatrix,embedding.CalCTDT(seqs)),axis=1) #dimension 39
# 	fMatrix = np.concatenate((fMatrix,embedding.CalProtVec(seqs)),axis=1) #dimension 1
# 	fMatrix = np.concatenate((fMatrix,embedding.CalPos(seqs)),axis=1)  #dimension 1
# 	fMatrix = fMatrix.astype(float)

# 	fMatirx = utils.normalize(fMatrix)

# 	return fMatrix


def remove_link(indices,rate=0.2):
	indices = [0,1,3,5,7,11,22]

	n = int(len(indices) * (1-rate))
	random.shuffle(indices)
	return indices[0:n]

