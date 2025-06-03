import os,sys,torch,pickle,pathlib
from tqdm import tqdm
from ..utility import SimilarityCalculator
from .local.clip import* 

class ClipTransform:
    def __init__(self,model_sel) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.available_models = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        self.available_models = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/16', 'ViT-B/32', 'ViT-L/14', 'ViT-L/14@336px']
        self.model_name = self.available_models[model_sel]
        _, self.preprocess = clip.load(self.model_name, self.device)

    def __call__(self, pic):
        pic = self.preprocess(pic)
        return pic

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class ClipWrapper():
    def __init__(self,model_sel=5,classes=[''],base_text='',generate_text=True) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_sel = model_sel
        self.available_models = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/16', 'ViT-B/32', 'ViT-L/14', 'ViT-L/14@336px']
        self.parameters = ['38,316,896','56,259,936','87,137,080','167,328,912','420,380,352','86,192,640','87,849,216','303,966,208','304,293,888']
        self.model_name = self.available_models[model_sel]
        print("Clip: " + self.model_name)
        self.model, self.preprocess = clip.load(self.model_name, self.device)
        total_params = sum(p.numel() for p in self.model.visual.parameters())
        print(f'total parameters: {total_params:,}')
        self.classes = classes
        #print(clip.available_models())
        self.base_text = base_text
        print('base text: ' + self.base_text)
        self.similarity_calculator = SimilarityCalculator()
        if generate_text: self.generate_text_features()
        self.extra_infor = {}
        #self.generate_word_bank_2()
        
    
    def retrieve_from_word_bank(self,input_features,topk=1,dis_func='L1'):
        self.similarity_calculator.topk = topk
        values, indices, similarity = self.similarity_calculator(self.word_bank_features,input_features,dis_func=dis_func)
        indices = indices.flatten()
        features = self.word_bank_features[indices]
        similarity = similarity[0][indices]
        word_list = [self.word_bank[i] for i in indices]
        tokens = self.word_bank_tokens[indices]
        tokens = [it[it!=0] for it in tokens]
        tokens = [it[it!=49406] for it in tokens]
        tokens = [it[it!=49407] for it in tokens]
        return word_list,similarity,features,tokens
        
    def generate_word_bank(self):
        current_path =  os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".")
        with open(current_path + '/supplement/clip/word_bank.txt','r') as file: lines = file.readlines()
        self.word_bank = [it.replace('\n','') for it in lines]
        if not pathlib.Path(current_path + f'/supplement/clip/{self.model_sel}_word_bank.pt').is_file():
            print('generate word bank')
            text_inputs = torch.cat([clip.tokenize(c) for c in self.word_bank]).to(self.device)
            self.word_bank_tokens = text_inputs
            with torch.no_grad(): 
                data_list = []
                batch_size = 256
                for i in range(0,len(text_inputs),batch_size):
                    if i+batch_size > len(text_inputs): data_list.append(self.model.encode_text(text_inputs[i:len(text_inputs)]))
                    else: data_list.append(self.model.encode_text(text_inputs[i:i+batch_size]))
                text_features = torch.cat((data_list))
            text_features /= text_features.norm(dim=-1, keepdim=True)
            self.word_bank_features = text_features
            torch.save(text_features.to(torch.float).to('cpu'),current_path + f'/supplement/clip/{self.model_sel}_word_bank.pt')
            torch.save(self.word_bank_tokens.to(torch.float).to('cpu'),current_path + f'/supplement/clip/{self.model_sel}_word_bank_tokens.pt')
        else:
            self.word_bank_features = torch.load(current_path + f'/supplement/clip/{self.model_sel}_word_bank.pt')
            self.word_bank_tokens = torch.load(current_path + f'/supplement/clip/{self.model_sel}_word_bank_tokens.pt')
        print(f'word bank size: {self.word_bank_features.shape}')
    
    def generate_word_bank_2(self):
        current_path =  os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".")
        if not pathlib.Path(current_path + f'/supplement/clip/{self.model_sel}_word_bank_2.pt').is_file():
            print('generate word bank')
            start = torch.ones(10000,1)*49406
            end = torch.ones(10000,1)*49407
            self.word_bank_tokens = []
            for i in range(75):
                zeros =  torch.zeros(10000,74-i)
                text_inputs = torch.randint(0, 49406, (10000, i+1))
                text_inputs = torch.cat((start,text_inputs,end,zeros),dim=1)
                self.word_bank_tokens.append(text_inputs)
            self.word_bank_tokens = torch.cat((self.word_bank_tokens))
            self.word_bank_tokens = self.word_bank_tokens.to(torch.long)
            self.word_bank_tokens = self.word_bank_tokens.to(self.device)
            
            with torch.no_grad(): 
                data_list = []
                batch_size = 256
                for i in tqdm(range(0,len(self.word_bank_tokens),batch_size)):
                    if i+batch_size > len(self.word_bank_tokens): data_list.append(self.model.encode_text(self.word_bank_tokens[i:len(self.word_bank_tokens)]))
                    else: data_list.append(self.model.encode_text(self.word_bank_tokens[i:i+batch_size]))
                text_features = torch.cat((data_list))
            text_features /= text_features.norm(dim=-1, keepdim=True)
            self.word_bank_features = text_features
            torch.save(text_features.to(torch.float).to('cpu'),current_path + f'/supplement/clip/{self.model_sel}_word_bank_2.pt')
            torch.save(self.word_bank_tokens.to(torch.float).to('cpu'),current_path + f'/supplement/clip/{self.model_sel}_word_bank_tokens_2.pt')
        else:
            self.word_bank_features = torch.load(current_path + f'/supplement/clip/{self.model_sel}_word_bank_2.pt')
            self.word_bank_tokens = torch.load(current_path + f'/supplement/clip/{self.model_sel}_word_bank_tokens_2.pt')
        
        print(f'word bank size: {self.word_bank_features.shape}')
    
    def generate_text_features(self):
        print('classes:')
        print(self.classes)
        text_inputs = torch.cat([clip.tokenize(self.base_text + ' ' + c) for c in self.classes]).to(self.device)
        with torch.no_grad(): self.text_features = self.model.encode_text(text_inputs)
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        #self.text_features = self.text_features.softmax(dim=-1)
    
    def generate_img_features(self,images):
        image_input = images.to(self.device) if len(images.shape) == 4 else images.unsqueeze(0).to(self.device)
        with torch.no_grad(): image_features = self.model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        #image_features = image_features.softmax(dim=-1)
        return image_features
    
    def get_predictions_softmax(self,images,topk=1):
        image_features = self.generate_img_features(images)
        image_features = image_features[:, None, :]
        similarity = torch.abs(image_features - self.text_features)
        similarity = similarity.sum(dim=-1)
        similarity = torch.neg(similarity)
        values, indices = similarity.topk(topk)
        return values, indices
    
    def get_predictions(self,images,topk=1):
        image_features = self.generate_img_features(images)
        similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        values, indices = similarity.topk(topk)
        return values, indices
    
    def print_result(self,images):
        values, indices = self.get_predictions(images,topk=len(self.classes))
        values, indices = values[0], indices[0]
        print("\nTop predictions:\n")
        for value, index in zip(values, indices):
            print(f"{self.classes[index]:>16s}: {100 * value.item():.2f}%")
    
    def eval_dataset(self,dataset):
        acc,nums = 0,0
        for images, labels in tqdm(dataset):
            values, indices = self.get_predictions(images)
            indices = torch.flatten(indices)
            result = torch.eq(labels.to(self.device),indices.to(self.device))
            acc += torch.sum(result)
            nums += len(labels)
        acc = acc.to(torch.float) / nums
        print(acc)
        return acc
    
    def convert_dataset(self,dataset,name,path):
        def convert_one(dataset):
            data  = {'data':[],'targets':[]}
            for images, labels in tqdm(dataset):
                img_features = self.generate_img_features(images)
                
                data['data'].append(img_features.detach().cpu().type(torch.float))
                data['targets'].append(labels.detach().cpu().type(torch.long))
                #break
            data['data'] = torch.cat(data['data'],0)
            data['targets'] = torch.cat(data['targets'],0)
            return data
        
        train_data = convert_one(dataset['train'])
        test_data = convert_one(dataset['test'])
        validate_data = convert_one(dataset['validate'])
        data = {'train':train_data,'test':test_data,'validate':validate_data}
        torch.save(data,path + f'/{name}_clip_{self.model_name.replace("/","_")}.pt')
    
    def inference_dataset(self,dataset):
        data  = {'data':[],'targets':[]}
        for images, labels in tqdm(dataset):
            img_features = self.generate_img_features(images)
            
            data['data'].append(img_features.detach().cpu().type(torch.float))
            data['targets'].append(labels.detach().cpu().type(torch.long))
        
        data['data'] = torch.cat(data['data'],0)
        data['targets'] = torch.cat(data['targets'],0)
        return data


if __name__ == '__main__':
    cw = ClipWrapper(5)
    cw.generate_text_features()
    #test = clip.tokenize('ff adsf fdas')
    #image = torch.rand((3,224,224))
    #image = cw.preprocess(image)
    #cw.generate_img_features(image)
    