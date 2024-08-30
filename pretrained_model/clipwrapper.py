import os,sys,clip,torch,pickle
from pytorch_lib import*
from python_lib import*
from tqdm import tqdm

class ClipTransform:
    def __init__(self,model_sel) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.available_models = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        self.model_name = self.available_models[model_sel]
        _, self.preprocess = clip.load(self.model_name, self.device)

    def __call__(self, pic):
        pic = self.preprocess(pic)
        return pic

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class ClipWrapper():
    def __init__(self,model_sel=5,classes=[''],base_text='') -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.available_models = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        self.model_name = self.available_models[model_sel]
        print("Clip: " + self.model_name)
        self.model, self.preprocess = clip.load(self.model_name, self.device)
        self.classes = classes
        #print(clip.available_models())
        self.base_text = base_text
        print('base text: ' + self.base_text)
        self.generate_text_features()
    
    def generate_text_features(self):
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
        batch = 0
        acc = 0
        for images, labels in tqdm(dataset):
            values, indices = self.get_predictions(images)
            indices = torch.flatten(indices)
            result = torch.eq(labels.to(self.device),indices.to(self.device))
            acc += torch.sum(result)/len(labels)
            batch += 1
        acc /= batch
        print(acc)
    
    def convert_dataset(self,dataset,name,path):
        def convert_one(dataset):
            data  = {'data':[],'targets':[]}
            for images, labels in tqdm(dataset):
                img_features = self.generate_img_features(images)
                
                data['data'].append(img_features.cpu().type(torch.float))
                data['targets'].append(labels.cpu().type(torch.long))
                #break
            data['data'] = torch.cat(data['data'],0)
            data['targets'] = torch.cat(data['targets'],0)
            return data
        
        train_data = convert_one(dataset['train'])
        test_data = convert_one(dataset['test'])
        validate_data = convert_one(dataset['validate'])
        data = {'train':train_data,'test':test_data,'validate':validate_data}
        torch.save(data,path + f'/{name}_clip_{self.model_name.replace("/","_")}.pt')