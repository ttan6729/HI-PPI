import sys
import os
import argparse
import torch
import torch.nn as nn
import math
import json
import random
import time
sys.path.append('src')
import Models
import ppi_data
import utils

def str2bool(v):
	"""
	Converts string to bool type; enables command line 
	arguments in the format of '--arg1 true --arg2 false'
	"""
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')
	return

#graph: embed1: data, edge1: local , edge2 global edges 
def train(model,data,loss_fn,optimizer,device,result_prefix=None,batch_size=512,epochs=100,scheduler=None,global_best_f1=0.0,args=None):
	with open(result_prefix+'.txt','w') as f:
		f.write('')
	#torch.backends.cudnn.benchmark =  True
	#torch.backends.cudnn.enabled =  True
	print('begin training')
	best_f1,best_epoch = 0.0,0
	result = None
	scaler = torch.cuda.amp.GradScaler()
	val_size = len(data.val_mask)
	aly_data = None
	for epoch in range(epochs):
		f1_sum,loss_sum ,recall_sum,precision_sum = 0.0,0.0,0.0,0.0
		steps = math.ceil(len(data.train_mask)/batch_size)
		model.train()
		random.shuffle(data.train_mask)
		train_loss_sum = 0.0
		for step in range(steps):
			torch.cuda.empty_cache()
			if step == steps-1:
				train_edge_id = data.train_mask[step*batch_size:]
			else:
				train_edge_id = data.train_mask[step*batch_size:(step+1)*batch_size]
			
			#output = model(x=data.embed1,edge_index=data.edge2,sparse_adj=data.sparse_adj2,edge_id=train_edge_id) #edge index: list
			output = model(data=data,edge_id=train_edge_id)
			label = data.edge_attr[train_edge_id]
			label = label.type(torch.FloatTensor).to(device)
			loss = loss_fn(output,label)
			train_loss_sum += loss
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()			
		#validation
		model.eval()	
		valid_pre_result_list = []
		valid_label_list = []
		valid_loss_sum = 0.0
		#torch.save()#save model
		steps = math.ceil(len(data.val_mask) / batch_size)
		saved_pred = []

		with torch.no_grad(): #validation set
			for step in range(steps):
				if step == steps-1:
					valid_edge_id = data.val_mask[step*batch_size:]
				else:
					valid_edge_id = data.val_mask[step*batch_size:(step+1)*batch_size]

				output = model(data=data,edge_id=valid_edge_id)

				label = data.edge_attr[valid_edge_id]
				label = label.type(torch.FloatTensor).to(device)
				loss = loss_fn(output,label)
				valid_loss_sum += loss.item()

				m = nn.Sigmoid()
				pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

				valid_pre_result_list.append(pre_result.cpu().data)
				valid_label_list.append(label.cpu().data)
				saved_pred.append(m(output).to(device).cpu().data )


		valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
		valid_label_list = torch.cat(valid_label_list, dim=0)

		saved_pred = torch.cat(saved_pred,dim=0)
		#print(saved_pred.size())

		metrics = utils.Metrictor_PPI(valid_pre_result_list, valid_label_list)
		record = metrics.append_result(result_prefix+'.txt',epoch+1,train_loss_sum,valid_loss_sum)
		print(record)

		recall_sum += metrics.recall
		precision_sum += metrics.pre
		f1_sum += metrics.microF1
		loss_sum += loss.item()
		valid_loss = valid_loss_sum / steps

		if best_f1 < metrics.microF1: #epoch == epochs -1 :
			best_f1 = metrics.microF1
			best_epoch = epoch
			result =  {'pred':saved_pred,'actual':valid_label_list}
			if args.aly:
				torch.save(model.state_dict(),result_prefix+'_weighs.pt' )
			#torch.save()
	
	if global_best_f1 < best_f1:
		global_best_f1 = best_f1
		torch.save(result,result_prefix+'.pt')

	if args.aly:
		#model.load_state_dict(torch.load(result_prefix+'_weighs.pt', weights_only=True))
		aly_data = model.compute_hiearchical_level(data=data)


	return global_best_f1, aly_data



def get_args_parser():
	parser = argparse.ArgumentParser('PwPPI',add_help=False)
	parser.add_argument('-m',default=None,type=str,help='mode, optinal value: bfs,dfs,rand,read,data')
	# parser.add_argument('-m',default='s',type=str,help='mode')
	parser.add_argument('-o',default='output',type=str)
	parser.add_argument('-t', default='default', type=str,help='for test distintct models')
	# parser.add_argument('-sf', default=None,type=str,help='optional input, contains path for sequence and relation file')
	parser.add_argument('-i',default=None,type=str,help='path for sequnce and relation file')
	parser.add_argument('-i1',default=None,type=str,help='sequence file')
	parser.add_argument('-i2',default=None,type=str,help='relation file')
	parser.add_argument('-i3',default=None,type=str,help='file path of test set indices (for read mode)')
	parser.add_argument('-i4',default=None,type=str,help='prefix for the pre generated structure')
	parser.add_argument('-s1',default='/home/user1/code/PPI4/data/structure_data',type=str,help='file path for structure file')
	#parser.add_argument('-i4',default='../data/map1.csv',type=str,help='file path for map STRING id to uniref id')
	parser.add_argument('-e',default=50,type=int,help='epochs')
	parser.add_argument('-b', default=256, type=int,help='batch size')
	parser.add_argument('-ln', default=2, type=int,help='graph layer num')
	parser.add_argument('-L', default=128, type=int,help='length for sequence padding')
	parser.add_argument('-Loss', default='CE', type=str,help='loss function')
	parser.add_argument('-ff', default='CnM', type=str,help='feature fusion option, default mul')
	parser.add_argument('-hl', default=512, type=int,help='hidden layer')
	parser.add_argument('-sv',default=False,type=str2bool,help='if save dataset path')
	parser.add_argument('-cuda',default=True,type=str2bool,help='if use cuda')
	parser.add_argument('-force',default=True,type=str2bool,help='if write to existed output file')
	parser.add_argument('-mainfold',default='Hyperboloid',type=str,help='any of the following: Euclidean, Hyperboloid, PoincareBall')
	parser.add_argument('-pr',default=0.0,type=float,help='perturbation ratio')
	parser.add_argument('-sf',default=None,type=str,help='pfolder that contain pdb file')
	return parser

python3 main.py -m data -i1 [seq file] -i2 [interaction file]


#cd /home/user1/code/PPI4/HI-PPI && source /home/user1/code/PPIKG/env/bin/activate
#python3 main.py -m bfs -t HI-PPI -i data/27K.txt -i4 features/27K -o test -e 100 -mainfold Hyperboloid
#python3 main.py -m bfs -t HI-PPI -i 27K.txt -i4 features/27K -o test -e 100 -mainfold Hyperboloid
def main(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(torch.cuda.is_available())
	print(device)

	if args.i:
		with open(args.i,'r') as f:
			args.i1 = f.readline().strip()
			args.i2 = f.readline().strip()
	
	if not args.force:
		if os.path.isfile(args.o+'.txt'):
			print('output name already exists')
			exit()

	if args.m == 'data':
		ppi_data.generate_structure_feature(args)
		return

	PPIData = ppi_data.PPIData(args)
	data = PPIData.data
	data.to(device)
	if args.t == 'HI-PPI':
		model = Models.HIPPI(data.embed1.shape[-1],args=args,layer_num=args.ln,in_len=args.L).to(device)	
	elif args.t == 'ab1':
		model = Models.ablation1(data.embed1.shape[-1],args=args,layer_num=args.ln,in_len=args.L).to(device)
	elif args.t == 'ab2':
		model = Models.ablation2(data.embed1.shape[-1],args=args,layer_num=args.ln,in_len=args.L).to(device)

	if args.Loss=='CE':
		loss_fn = nn.BCEWithLogitsLoss().to(device)
	elif args.Loss=='AS':
		loss_fn = Models.AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True).to(device)
	optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=5e-4)


	start = time.time()
	best_f1,aly_data = train(model,data,loss_fn,optimizer,device,args.o,args.b,args.e,None,0.0,args)
	end = time.time()
	mins = (end - start)/60

	print(f'\nbest F1 score: {best_f1:.4f}, running time: {mins:.2f} minutes')
	print(f'output save to {args.o}.txt')
	if args.o:
		with open(args.o+'.txt','r+') as file:
			file_data = file.read()
			file.seek(0,0)
			command = ' '.join(arg for arg in sys.argv)
			line = f'command: {command}\n'
			line += f'best F1 score: {best_f1:.4f}\n'
			line += f'training time: {mins:.2f} minutes\n'
			line += f'mode: {args.m}\n'		
			line += f'layer num: {args.ln}\n'	
			line += f'filePath: {args.i1} {args.i2}\n'
			if args.i3:
				line += f'valid set path: {args.i3}\n'
			line += f'model: {args.t}\n'
			line +=f'Loss function: {args.Loss}\n'
			line +=f'max length of seqs: {args.L}\n'
			#line +=f'feature 1 shape {graph.f1.size()}\n'# feature 2 shape {graph.f2.size()}\n'
			line +=f'epoch: {args.e}\n'
			line +=f'feature fusion mode: {args.ff}\n'
			line +=f'mainfold: {args.mainfold}\n'
			
			file.write(line + '\n' + file_data)


	if args.aly:
		PPIData.analyze_hiearchical(aly_data)

	return best_f1

#python3 main.py -m read -t HPPI19 -i 27K.txt -i3 /home/user1/code/PPIKG/multiSet/o2/27k_bfs10.data -i4 features/27K -o ../result/test -e 100 -mainfold Hyperboloid
if __name__ == "__main__":
	parser = argparse.ArgumentParser('PPIM', parents=[get_args_parser()])
	args = parser.parse_args()	
	best_f1 = main(args)

