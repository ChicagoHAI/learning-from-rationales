import torch

def scatter_mean_batch(src: torch.Tensor, indices:torch.Tensor):
	'''
	Calculate the mean of src across the indices specified in indices, and output them in a tensor of size
	batch size x max sentence count
	:param src:
	:param indices:
	:param dim:
	:return:
	'''
	src_ones = torch.ones_like(src)
	sums = torch.zeros((src.shape[0], indices.max()+1), device=src.device)
	counts = torch.zeros((src.shape[0], indices.max()+1), device=src.device)
	sums = torch.scatter_add(sums,1,indices,src)
	counts = torch.scatter_add(counts,1,indices,src_ones)
	means = sums / (counts + 0.0001)

	return means
	
	# phi = ts.scatter_mean(phi.float(),ids,dim=1)


def scatter_mean_gather_batch(src: torch.Tensor, indices:torch.Tensor):
	'''
	Calculate the mean of src across specified indices, then gather them back to the original dimensions of src
	:param src:
	:param indices:
	:param dim:
	:return:
	'''
	means = scatter_mean_batch(src, indices)
	meaned_src = torch.gather(means, 1, indices)
	return meaned_src