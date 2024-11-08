import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

class CustomDataSet(Dataset):
	def __init__(self, main_dir, transform, height=720, width=1280, train=True, dec_strds=[5, 4, 4, 3, 2]):
		self.main_dir = main_dir
		self.transform = transform
		frame_idx, self.frame_path = [], []
		accum_img_num = []
		all_imgs = os.listdir(main_dir)
		all_imgs.sort()

		num_frame = 0 
		for img_id in all_imgs:
			self.frame_path.append(img_id)
			frame_idx.append(num_frame)  # if 135 frames in total, this list will store 0, 1, 2, ..., 133, 134
			num_frame += 1          

		# import pdb; pdb.set_trace; from IPython import embed; embed()
		accum_img_num.append(num_frame)
		# the id for first frame is 0 and the id for last is 1
		self.frame_idx = []
		print('---------------len frame_idx----------------' , len(frame_idx))
		for i in range(len(frame_idx)):
			x = frame_idx[i]
			self.frame_idx.append(float(x) / (len(frame_idx) - 1))
		self.accum_img_num = np.asfarray(accum_img_num)

		self.height = height
		self.width = width
		self.dec_strds = dec_strds

	def __len__(self):
		return len(self.frame_idx)

	def __getitem__(self, idx):
		valid_idx = int(idx)
		img_id   = self.frame_path[valid_idx]
		img_id_p = self.frame_path[valid_idx-1] if valid_idx!=0 else self.frame_path[valid_idx]
		img_id_f = self.frame_path[valid_idx+1] if valid_idx!=len(self.frame_idx)-1 else self.frame_path[valid_idx]

		img_id_pp = self.frame_path[valid_idx-1] if valid_idx>1 else self.frame_path[valid_idx]
		img_id_ff = self.frame_path[valid_idx+1] if valid_idx<len(self.frame_idx)-2 else self.frame_path[valid_idx]


		img_name = os.path.join(self.main_dir, img_id)
		img_name_p = os.path.join(self.main_dir, img_id_p)
		img_name_f = os.path.join(self.main_dir, img_id_f)
		img_name_pp = os.path.join(self.main_dir, img_id_pp)
		img_name_ff = os.path.join(self.main_dir, img_id_ff)

		image = Image.open(img_name).convert("RGB")
		image_p = Image.open(img_name_p).convert("RGB")
		image_f = Image.open(img_name_f).convert("RGB")
		image_pp = Image.open(img_name_pp).convert("RGB")
		image_ff = Image.open(img_name_ff).convert("RGB")
		
		if image.size != (self.width, self.height):
			image = image.resize((self.width, self.height))
			image_p = image_p.resize((self.width, self.height))
			image_f = image_f.resize((self.width, self.height))
			image_pp = image_pp.resize((self.width, self.height))
			image_ff = image_ff.resize((self.width, self.height))
			# image = transforms.CenterCrop((self.height, self.width))(image)
			# image_p = transforms.CenterCrop((self.height, self.width))(image_p)
			# image_f = transforms.CenterCrop((self.height, self.width))(image_f)
			# image_pp = transforms.CenterCrop((self.height, self.width))(image_pp)
			# image_ff = transforms.CenterCrop((self.height, self.width))(image_ff)

		tensor_image = self.transform(image)
		tensor_image_p = self.transform(image_p)
		tensor_image_f = self.transform(image_f)
		tensor_image_pp = self.transform(image_pp)
		tensor_image_ff = self.transform(image_ff)
		
		if tensor_image.size(1) > tensor_image.size(2):
			tensor_image = tensor_image.permute(0,2,1)
		frame_idx = torch.tensor(self.frame_idx[idx])

		data_dict = {
			"img_id": frame_idx,
			"img_gt": tensor_image, #3 h w
			"img_p" : tensor_image_p, #3 h w
			"img_f" : tensor_image_f, #3 h w
		}
        
		for i, strd in enumerate(self.dec_strds):
			tensor_image = F.avg_pool2d(tensor_image, kernel_size=strd, stride=strd)
			data_dict[f'img_{i+1}'] = tensor_image

		return data_dict