# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 12:47:20 2018

@author: sabarnikundu
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable 
import torch.nn.functional as F
from utils import getDataset
import torchvision.transforms as transforms
import torch.optim as optim

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        
    def forward(self, image_tensor):
        features = self.resnet(image_tensor)
        features = features.view(features.size(0), -1)
        features = self.linear(features)
        return features


   
class EncoderRNN(nn.Module):
    def __init__(self, rnn_op_size, ques_length = 30, embed_size = 256, hidden_size = 512,num_layers=1,bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.embed = nn.Linear(1, embed_size)
        self.lstm = nn.LSTMCell(embed_size, hidden_size, bias=True)
        self.num_layers = num_layers
        self.direction = 2 if bidirectional else 1
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.ques_length = 30
    def forward(self,ques_tensor):
        ques_tensor = ques_tensor.transpose(1,0).unsqueeze(2)
        embed_ques_words = self.embed(ques_tensor)
        h = torch.zeros(ques_tensor.shape[1],self.hidden_size)
        c = torch.zeros(ques_tensor.shape[1],self.hidden_size)
        for word in embed_ques_words:
            h,c = self.lstm(word,(h,c))
        embed_ques = h
        return embed_ques
    


class FCanswers(nn.Module):
   def __init__(self,embed_size):
       super(FCanswers, self).__init__()
       self.fc1 = nn.Linear(1,256)
       self.fc2 = nn.Linear(256,256)
       self.fc3 = nn.Linear(256,256)   
       
   def forward(self, answer_tensor):
       answer = F.relu(self.fc1(answer_tensor))
       answers = F.relu(self.fc2(answer))
       answer = self.fc3(answer)
       #print(answers.shape)
       return answer
        
       
#
#class FullyConnected(nn.Module):
 #  def __init__(self):
  #      super(FullyConnected, self).__init__()
   #     self.fc1 = nn.Linear()
    #    self.fc2 = nn.Linear()
     #   self.fc3 = nn.Linear()   
        
  # def forward(self, x):
  #     x = torch.cat((features.unsqueeze(1), embed_ques), 1)
   #    x = F.relu(self.fc1(x))
    #   x = F.relu(self.fc2(x))
     #  x = self.fc3(x)
      # return x
#    
#    
#class MSELoss(_Loss):
#      def __init__(self, size_average=True, reduce=True):
#          super(MSELoss, self).__init__(size_average, reduce)
#
#      def forward(self, answer, fc2_concat):
#          _assert_no_grad(fc2_concat)
#          out = F.mse_loss(answer,fc2_concat, size_average=self.size_average, reduce=self.reduce) 
#          return out

    
#class DecoderRNN(nn.Module):
#    def __init__(self, embed_size, vocab_size,vocab_a_size, hidden_size, ques, num_layers):
#        super(DecoderRNN, self).__init__()
#        self.embed = nn.Embedding(vocab_size,vocab_a_size, embed_size)
#        self.lstm = nn.LSTM(embed_size, hidden_size, vocab_a_size, num_layers, batch_first=True)
#        self.linear = nn.Linear(hidden_size, vocab_size)
#        self.init_weights()
#    
#    def init_weights(self):
#        """Initialize weights."""
#        self.embed.weight.data.uniform_(-0.1, 0.1)
#        self.linear.weight.data.uniform_(-0.1, 0.1)
#        self.linear.bias.data.fill_(0)
#        
#    def forward(self, features, lengths, question, answer):
#        """Decode image feature vectors and generates answer."""
#        embeddings_a  = self.embed(answer)
#        embeddings_q = self.embed(question) #embedding for answers and had to do embeddings for questions
#        embeddings_q = torch.cat((features.unsqueeze(1), embeddings_q,embeddings_a), 1)#embeddings_q concat with the features extracted of the image
#        packed = pack_padded_sequence(embeddings_q, lengths, batch_first=True) 
#        hiddens, _ = self.lstm(packed)
#        outputs = self.linear(hiddens[0])
#        softmax = self.softmax(outputs)
#        return softmax
#    
#    def sample(self, features,embeddings_q,states=None):
#        """Samples answer for given set of image features and question (Greedy search)."""
#        sampled_ids = []
#        inputs = torch.cat((features.unsqueeze(1),embeddings_q),1)
#        for i in range(20):                                    # maximum sampling length
#            hiddens, states = self.lstm(inputs, states)        # (batch_size, 1, hidden_size), 
#            outputs = self.linear(hiddens.squeeze(1))          # (batch_size, vocab_size)
#            predicted = outputs.max(1)[1]
#            sampled_ids.append(predicted)
#            inputs = self.embed(predicted)
#            inputs = inputs.unsqueeze(1)                       # (batch_size, 1, embed_size)
#        sampled_ids = torch.cat(sampled_ids, 1)                # (batch_size, 20)
#        return sampled_ids.squeeze()


class MainNet(nn.Module):
    def __init__(self, cnn_op_size, rnn_seq_length, rnn_embed_len, rnn_op_size,ans_op):
        super(MainNet,self).__init__()
        self.Image_encoder = EncoderCNN(cnn_op_size)
        self.Ques_encoder = EncoderRNN(rnn_op_size,rnn_seq_length,rnn_embed_len)
        self.Answer_net = FCanswers(ans_op)
        self.FC1 = nn.Linear(768,512)
        self.FC2 = nn.Linear(512,256)
        
    
    def forward(self,img,ques,ans):
        img_embed = self.Image_encoder(img)
        ques_embed = self.Ques_encoder(ques)
        ans_net = self.Answer_net(ans)
        concat = torch.cat((ques_embed,img_embed), 1)
        fc_concat = F.relu(self.FC1(concat))
        fc2_concat = F.relu(self.FC2(fc_concat))
        
        #return img_embed, ques_embed, ans_net,concat, fc2_concat,loss_b,loss_b1
        return ans_net,fc2_concat
if __name__ == "__main__":
    transform = transforms.Compose([ 
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor()])
    ques_stoi_path = "../ques_stoi.tsv" 
    ans_itos_path  = "../ans_itos.tsv"
    samples_tsv_path = "../train.tsv" 
    image_path = "../train2014"
    d = getDataset(ques_stoi_path, ans_itos_path,samples_tsv_path,image_path,transform)
    dataloader = torch.utils.data.DataLoader(dataset = d,
                                             batch_size = 4,
                                             shuffle = False,
                                             num_workers = 2)
    net = MainNet(256,30,256,256,256)
    loss = nn.MSELoss()
    optimizer = optim.Adam(net.parameters())
    for i,batch in enumerate(dataloader):
        optimizer.zero_grad()
        imgs,ques,ans = batch
        ansout,queryout = net(imgs,ques,ans)
        queryloss = loss(queryout, ansout.data)
        ansloss = loss(ansout, queryout.data)
        print (i,len(dataloader),queryloss,ansloss)
        queryloss.backward(retain_graph=True)
        ansloss.backward()
        optimizer.step()
        #print(out[2].shape)
        
        
