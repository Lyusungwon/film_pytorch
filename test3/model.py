import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import numpy as np


class MLP(nn.Module):
	def __init__(self, layers, dropout=None, dropout_rate=None, last=False):
		super(MLP, self).__init__()
		self.layers = layers
		self.dropout = dropout
		self.dropout_rate = dropout_rate
		self.last = last
		net = []
		for n, (inp, outp) in enumerate(zip(layers, layers[1:])):
			net.append(nn.Linear(inp, outp))
			net.append(nn.ReLU(inplace=True))
			if self.dropout == n + 1:
				net.append(nn.Dropout(self.dropout_rate))
		if self.last:
			net = net[:-1]
		net = nn.ModuleList(net)
		self.net = nn.Sequential(*net)
		print(self.net)

	def forward(self, x):
		x = x.view(x.size()[0], -1)
		x = self.net(x)
		return x


class Conv(nn.Module):
	def __init__(self, layer_config, channel_size, batch_norm, d_inner, n_head, d_k, d_v):
		super(Conv, self).__init__()
		self.layer_config = layer_config
		self.channel_size = channel_size
		self.batch_norm = batch_norm
		self.input_h = 64
		self.input_w = 64
		prev_filter = self.channel_size
		net = nn.ModuleList([])
		for num_filter, kernel_size, stride in layer_config:
			net.append(nn.Conv2d(prev_filter, num_filter, kernel_size, stride, (kernel_size - 1)//2))
			if batch_norm:
				self.input_h = self.input_h // 2
				self.input_w = self.input_w // 2
				net.append(nn.LayerNorm([num_filter, self.input_h, self.input_w]))
			net.append(nn.ReLU(inplace=True))
			net.append(SelfAttentionLayer(num_filter, d_inner, n_head, d_k, d_v, dropout=0.1))
			prev_filter = num_filter + d_v
		self.net = nn.Sequential(*net)
		print(self.net)

	def forward(self, x):
		x = self.net(x)
		return x


class Text_embedding(nn.Module):
	def __init__(self, color_size, question_size, embedding_size):
		super(Text_embedding, self).__init__()
		self.color_embedding = nn.Embedding(color_size, embedding_size, padding_idx=None)
		self.question_embedding = nn.Embedding(question_size, embedding_size, padding_idx=None)

	def forward(self, x):
		c_embedded = self.color_embedding(x[:, 0])
		q_embedded = self.question_embedding(x[:, 1])
		text_embedded = torch.cat([c_embedded, q_embedded], 1)
		return text_embedded


class SelfAttentionLayer(nn.Module):
	''' Compose with two layers '''

	def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
		super(SelfAttentionLayer, self).__init__()
		self.slf_attn = MultiHeadAttention(
			n_head, d_model, d_k, d_v, dropout=dropout)
		self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

	def forward(self, enc_input):
		enc_output, enc_slf_attn = self.slf_attn(
			enc_input, enc_input, enc_input)
		output = torch.cat([enc_input, enc_output], 1)
		# enc_output = self.pos_ffn(enc_output)
		# enc_output = enc_output.view(enc_output.size()[0], -1)
		return output


class ScaledDotProductAttention(nn.Module):
	''' Scaled Dot-Product Attention '''

	def __init__(self, temperature, attn_dropout=0.1):
		super().__init__()
		self.temperature = temperature
		self.dropout = nn.Dropout(attn_dropout)
		self.softmax = nn.Softmax(dim=2)

	def forward(self, q, k, v):

		attn = torch.bmm(q, k.transpose(1, 2))
		attn = attn / self.temperature

		attn = self.softmax(attn)
		attn = self.dropout(attn)
		output = torch.bmm(attn, v)

		return output, attn


class MultiHeadAttention(nn.Module):
	''' Multi-Head Attention module '''

	def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
		super().__init__()

		self.n_head = n_head
		self.d_k = d_k
		self.d_v = d_v

		self.w_qs = nn.Linear(d_model, n_head * d_k)
		self.w_ks = nn.Linear(d_model, n_head * d_k)
		self.w_vs = nn.Linear(d_model, n_head * d_v)
		nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
		nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
		nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

		self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
		self.layer_norm = nn.LayerNorm(d_model)

		self.fc = nn.Linear(n_head * d_v, d_model)
		nn.init.xavier_normal_(self.fc.weight)

		self.dropout = nn.Dropout(dropout)

	def forward(self, q, k, v):

		d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

		sz_b, len_q, _ = q.size()
		sz_b, len_k, _ = k.size()
		sz_b, len_v, _ = v.size()

		residual = q
		q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
		k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
		v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

		q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
		k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
		v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

		output, attn = self.attention(q, k, v)

		output = output.view(n_head, sz_b, len_q, d_v)
		output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

		output = self.dropout(self.fc(output))
		output = self.layer_norm(output + residual)

		return output, attn


class PositionwiseFeedForward(nn.Module):
	''' A two-feed-forward-layer module '''

	def __init__(self, d_in, d_hid, dropout=0.1):
		super().__init__()
		self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
		self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
		self.layer_norm = nn.LayerNorm(d_in)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		residual = x
		output = x.transpose(1, 2)
		output = self.w_2(F.relu(self.w_1(output)))
		output = output.transpose(1, 2)
		output = self.dropout(output)
		output = self.layer_norm(output + residual)
		return output



