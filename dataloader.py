import numpy as np
import h5py
import os
from tqdm import tqdm
import random

import pickle
from PIL import Image
import io


DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_IMG = os.path.join(DIR, './dataset1/train_sequence_img.pkl')
VAL_IMG = os.path.join(DIR, './dataset1/val_sequence_img.pkl')
TRAIN_ODO = os.path.join(DIR, './dataset1/train_sequence_odo.pkl')
VAL_ODO = os.path.join(DIR, './dataset1/val_sequence_odo.pkl')
TRAIN_DATA = os.path.join(DIR, './dataset1/train_data_256_128.h5')
VAL_DATA = os.path.join(DIR, './dataset1/val_data_256_128.h5')

class DataLoader:

	def __init__(self, batch_size):
		print('Loading img sequence  ...')
		with open(TRAIN_IMG, 'rb') as f:
			self.all_im_seq_train = pickle.load(f)
		with open(VAL_IMG, 'rb') as f:
			self.all_im_seq_val = pickle.load(f)    
		print('Done.')

		print('Loading odo data ...')
		with open(TRAIN_ODO, 'rb') as f:
			self.train_odo_data = pickle.load(f)
		with open(VAL_ODO, 'rb') as f:
			self.val_odo_data = pickle.load(f)    
		print('Done.')
        
		print('Loading trian img data ...')
		self.train_data_file = TRAIN_DATA
		print('Loading val img data ...')      
		self.val_data_file = VAL_DATA
		print('Done.')

# 		with h5py.File( self.train_data_file, 'r') as f:
# 			self.all_train_keys = f.keys();
		self.all_train_grps = self.all_im_seq_train        
		self.batch_size = batch_size
# 		self.grp_list = list(range(0,len(self.all_train_grps)-1))
# 		self.count_list=0
        
# 	def shuffle_grp(self,):
# 		self.count_list=0        
# 		random.shuffle(self.grp_list)
# # 		print(self.grp_list)         
            
	def train_data_batch(self,):
		all_train_grps =self.all_train_grps;
# 		grp_list = self.grp_list
		data_X_s = [];
		data_X_o = [];
		data_Y = [];
# 		num = self.count_list;
		count=0        
		width=256;
		height=128;
		with h5py.File( self.train_data_file, 'r') as f:
			while 1:
# 				if num >2973:
# 					num=num-2974
				grp_idx = random.randint(0,len(self.all_train_grps)-1);
				seq_idx = random.randint(0,24);
				curr_seq = all_train_grps[grp_idx][seq_idx:seq_idx+5]
				curr_seq_segs = [];
				curr_seq_odo = [];
				for frame in curr_seq:
					img_arry = np.zeros((128, 256, 24))
					g=np.array(f[frame])
					img = Image.open(io.BytesIO(g))
					for i in range(height):
						for j in range(width):
							gg=img.getpixel((j,i))
							img_arry[i,j,gg]=1.0
					curr_seq_segs.append(img_arry);
					curr_seq_odo.append( np.expand_dims(self.train_odo_data[frame], axis=0) )
					if len(curr_seq_segs) < 5:
						continue;
				data_X_s.append( np.concatenate(curr_seq_segs[0:4], axis = -1) )
				data_X_o.append( np.concatenate(curr_seq_odo, axis = 0) )
				data_Y.append( curr_seq_segs[-1] )
				count += 1
				if count == self.batch_size:
# 					self.count_list=self.count_list+count                    
					break;
			data_X_s = np.array(data_X_s);
			data_X_o = np.array(data_X_o);
			data_Y = np.array(data_Y);
# 			print(self.count_list)
			return ( data_X_s, data_X_o, data_Y)

	def get_val_data(self,num_of_test_examples,test_samples):
		all_val_grps = self.all_im_seq_val;
		data_X_s = [];
		data_X_o = [];
		data_Y = [];
		data_Y_ = [];
		width=256;
		height=128;
# 		all_val_odo_keys = self.val_odo_data.keys();
		with h5py.File( self.val_data_file, 'r') as f:
			for i in tqdm(range(0,len(all_val_grps))):
				if i == num_of_test_examples:
					break;    
				grp_idx = i;
				seq_idx = 20 - 5;
				curr_seq =all_val_grps[grp_idx][seq_idx:seq_idx+5]
# 				curr_seq =[x.split('/')[-1][0:-4] for x in curr_seq]

				curr_seq_segs = [];
				curr_seq_odo = [];
				for frame in curr_seq:
					img_arry = np.zeros((128, 256, 24))
					g=np.array(f[frame])
					img = Image.open(io.BytesIO(g))
					for i in range(height):
						for j in range(width):
							gg=img.getpixel((j,i))
							img_arry[i,j,gg]=1.0
					curr_seq_segs.append(img_arry);
					curr_seq_odo.append( np.expand_dims(self.val_odo_data[frame], axis=0) );
					if len(curr_seq_segs) < 5:
						continue;
                        
# 				city = curr_seq[-1].split('_')[0];
# 				seq_num = curr_seq[-1].split('_')[1];
# 				frame_num = curr_seq[-1].split('_')[2];

				data_X_s.append( np.concatenate(curr_seq_segs[0:4], axis = -1) )
				data_X_o.append( np.concatenate(curr_seq_odo, axis = 0) )

				data_Y_.append(curr_seq_segs[-1] )

			data_X_s = np.array(data_X_s);
			data_X_o = np.array(data_X_o);
			data_Y_ = np.array(data_Y_);
			f.close();

			data_X_s = np.expand_dims(data_X_s, axis=1);
			data_X_s = np.repeat(data_X_s,test_samples,axis=1)
			data_X_s = np.reshape(data_X_s,(data_X_s.shape[0]*test_samples,data_X_s.shape[2],data_X_s.shape[3],data_X_s.shape[4]))
            
			data_X_o = np.expand_dims(data_X_o, axis=1);
			data_X_o = np.repeat(data_X_o,test_samples,axis=1)
			data_X_o = np.reshape(data_X_o,(data_X_o.shape[0]*test_samples,data_X_o.shape[2],data_X_o.shape[3]));

# 			data_Y = np.load('val_gt.npy')[:num_of_test_examples]

			data_Y=data_Y_

			return ( data_X_s, data_X_o, data_Y, data_Y_)



