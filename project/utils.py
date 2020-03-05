# -*- coding: utf-8 -*-

# IMPORTS
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
class getDataset(Dataset):
    def __init__(self,ques_stoi_path,ans_itos_path,samples_tsv_path,image_path,transform):
        self.image_path = image_path
        self.transform = transform
        with open(ques_stoi_path, "r") as f:
            words = f.readlines()
            w=[]
            wi=[]
            for word in words:
                word.split('\t')
                word_ids = word.split('\t')
                w.append(word_ids[0])
                wi.append(int(word_ids[1].split('\n')[0]))
                #words = word_ids[0]
                #words_ids = word_ids[1]
            self.ques_word_ids = dict(zip(w,wi))
        with open(ans_itos_path, "r") as f:
            words = f.readlines()
            w=[]
            wi=[]
            for word in words:
                word.split('\t')
                word_ids = word.split('\t')
                wi.append(int(word_ids[0]))
                w.append(word_ids[1])
                #words = word_ids[0]
                #words_ids = word_ids[1]
            self.ans_word_ids = dict(zip(wi,w))
        with open(samples_tsv_path, "r") as f:
            self.list_of_questions = f.readlines()
        
    def __getitem__(self,idx):
        ques_id, ques, image_id, ans_id = self.list_of_questions[idx].split('\t')
        # create image tensor
        img = Image.open(self.image_path+"/COCO_train2014_%012d"%(int(image_id))+".jpg").convert('RGB')
        img_tensor = self.transform(img)
        # question tensor
        question_words = ques.split()
        question_ids = [self.ques_word_ids[word] for word in question_words]
        ques_tensor = torch.ones(30)
        for i,item in enumerate(question_ids):
            ques_tensor[i] = item
        # answer tensor
        answer_tensor = torch.Tensor([int(ans_id)])
        return img_tensor,ques_tensor,answer_tensor
    def __len__(self):
        return len(self.list_of_questions)
    
if __name__ == "__main__":
    ques_stoi_path = "../ques_stoi.tsv" 
    ans_itos_path  = "../ans_itos.tsv"
    samples_tsv_path = "../train.tsv" 
    image_path = "../train2014"
    transform = transforms.Compose([ 
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor()])
    d = getDataset(ques_stoi_path, ans_itos_path,samples_tsv_path,image_path,transform)
    print (d)