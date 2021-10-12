import os, time, json, torch, torchvision
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from torchvision.models import resnet50
from PIL import Image

'''设置随机数放置漂移'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1937)

'''定义训练集DataSet'''
class ImageDataSet(Dataset):
    def __init__(self, dataset_path, transform, enhance ,featureEngine):
        #self.transform: 将PIL图片转化为tensor（经过预处理）
        self.transform = transform
        
        #self.category_reflect: 将物种类类别对应到某一类label
        self.category_reflect = {}
        for label, category in enumerate(os.listdir(dataset_path)):
            self.category_reflect[category] = label
        
        #self.dataset: 所有的tensor, label
        self.dataset = list()
        for category in os.listdir(dataset_path):
            category_path = os.path.join(dataset_path, category)+'/'
            category_label = self.category_reflect[category]
            for ptr in os.listdir(category_path):
                ptr_path = os.path.join(category_path, ptr)
                if os.path.splitext(ptr_path)[1] == '.png':
                    img = Image.open(ptr_path)
                    image_tensor = featureEngine(transform(img)[:3])
                    self.dataset.append((image_tensor, category_label))
                    for hense_time in range(2):
                        image_tensor = featureEngine(enhance(img)[:3])
                        self.dataset.append((image_tensor, category_label))
        print('all data has been loaded')

    def __getitem__(self, index):
        img_tensor, label = self.dataset[index]
        return img_tensor, label

    def __len__(self):
        return len(self.dataset)
    
'''定义验证集DataSet'''
class ValidDataSet(Dataset):
    def __init__(self, dataset_path, transform, featureEngine, reflect):
        #self.transform: 将PIL图片转化为tensor（经过预处理）
        self.transform = transform
        
        #self.category_reflect: 将物种类类别对应到某一类label
        self.category_reflect = reflect
        for label, category in enumerate(os.listdir(dataset_path)):
            self.category_reflect[category] = label
        
        #self.dataset: 所有的tensor, label
        self.dataset = list()
        for category in os.listdir(dataset_path):
            category_path = os.path.join(dataset_path, category)+'/'
            category_label = self.category_reflect[category]
            for ptr in os.listdir(category_path):
                ptr_path = os.path.join(category_path, ptr)
                if os.path.splitext(ptr_path)[1] == '.png':
                    img = Image.open(ptr_path)
                    image_tensor = featureEngine(transform(img)[:3])
                    self.dataset.append((image_tensor, category_label))
        print('all valid data has been loaded')

    def __getitem__(self, index):
        img_tensor, label = self.dataset[index]
        return img_tensor, label

    def __len__(self):
        return len(self.dataset)

'''定义测试集Dataset'''
class TestDataSet(Dataset):
    def __init__(self, dataset_path, transform, featureEngine, reflect):
        #self.transform: 将PIL图片转化为tensor（经过预处理）
        self.transform = transform
        
        #self.category_reflect: 将物种类类别对应到某一类label
        self.category_reflect = reflect
        
        #self.dataset: 所有的tensor, label
        self.dataset = list()
        self.image_order = list()
        for image in os.listdir(dataset_path):
            image_index = os.path.splitext(image)[0]
            self.image_order.append(image_index)
            image_path = os.path.join(dataset_path, image)
            img = Image.open(image_path)
            image_tensor = featureEngine(transform(img)[:3])
            image_label = 233
            self.dataset.append((image_tensor, image_label))
        print('all test data has been loaded')

    def __getitem__(self, index):
        img_tensor, label = self.dataset[index]
        return img_tensor, label

    def __len__(self):
        return len(self.dataset)
    
    

'''Ensemble1：从骨干网络(ResNet50)中抽取各层的编码特征''' 
def extract_features(X):
    features = []
    for i in range(len(bone_net)):
        X = bone_net[i](X)
        if i != 0:
            features.append(X)
    return features

'''Ensemble2：将抽取出的各层编码结果转化为同样的维度大小[bs, 256]'''
def depose_features(features):
    deposedFeatures = []
    for index, fet in enumerate(features):
        fet = torch.relu(reshape_net[index](fet).squeeze(dim=3).squeeze(dim=2))
        if index != 0:
            fet = scale_net[index-1](fet)
        deposedFeatures.append(fet)
    return deposedFeatures

'''ChannelAttention: 将Ensemble2中得到的各层编码进行合并，利用通道维注意力结构筛选重要特征'''
def predict(features):
    composed_feature = torch.cat(features, dim=1)
    attention_feature = attention_net(composed_feature)*composed_feature
    return output_net(attention_feature)

'''将Ensemble与Attent合并成为pipeline'''
def combine(X):
    features = extract_features(X)
    deposedFeatures = depose_features(features)
    y_hat = predict(deposedFeatures)
    return y_hat    
    

'''模型评价'''
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()

def idx2name(y_hat, reflect):
    return [reflect[item] for item in y_hat]

'''初始化训练集，验证集'''
#对于原图像
ToTensor = torchvision.transforms.ToTensor()
Reshape = torchvision.transforms.Resize((224,224))
transform = torchvision.transforms.Compose([Reshape, ToTensor])
#对于增广图像
shape_aug = torchvision.transforms.RandomResizedCrop(224, scale=(0.1, 1), ratio=(0.5, 2))
color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
enhance = torchvision.transforms.Compose([shape_aug, color_aug, ToTensor])
#特征工程
Normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
imageDataSet = ImageDataSet('train/', transform, enhance= enhance, featureEngine=Normalize)
validDataSet = ValidDataSet('valid/', transform, featureEngine=Normalize, reflect= imageDataSet.category_reflect)
testDataSet = TestDataSet('test_flatten/', transform, featureEngine=Normalize, reflect= imageDataSet.category_reflect)

batch_size = 128
train_iter = torch.utils.data.DataLoader(imageDataSet, batch_size=batch_size, shuffle=True)
valid_iter = torch.utils.data.DataLoader(validDataSet, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(testDataSet, batch_size=batch_size)

'''定义网络'''
#使用ResNet50预训练网络
resnet = resnet50(pretrained=True)
resnet = resnet.to(device)
input_net = nn.Sequential(*[resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool])
bone_net = nn.Sequential(*[input_net, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4])

#depose_feature函数适配网络
reshape_net = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1,1)),nn.AdaptiveAvgPool2d(output_size=(1, 1)),nn.AdaptiveAvgPool2d(output_size=(1, 1)),nn.AdaptiveAvgPool2d(output_size=(1, 1))).to(device)
scale_net = nn.Sequential(*[nn.Linear(512,256), nn.Linear(1024,256), nn.Linear(2048, 256)]).to(device)

#predict函数适配网络
attention_net = nn.Sequential(nn.Linear(1024,512), nn.Tanh(), nn.Linear(512,1024),nn.LayerNorm(1024), nn.Sigmoid()).to(device)
output_net = nn.Sequential(nn.Linear(1024, 22)).to(device)

'''模型训练'''
lr = 1e-4
optimizer = optim.Adam([{'params': scale_net.parameters(),'lr':lr},{'params':attention_net.parameters()},{'params': output_net.parameters(),'lr':lr},{'params':reshape_net.parameters(), 'lr':lr},{'params':bone_net.parameters(),'lr':lr/10}],lr=lr)
loss = torch.nn.CrossEntropyLoss()
print('model start training')
epoch = 8
for epc in range(epoch):
    count = 0
    for X,y in train_iter:
        X,y = X.to(device), y.to(device)
        y_hat = combine(X)
        l = loss(y_hat, y).sum()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        acc = accuracy(y_hat, y)
        count += 1
        print(epc, count, l.data.item(), acc)

del X,y, imageDataSet, train_iter
print('-----------------------EVALUATION---------------------')
print('model accuracy:')
acc, count = 0., 0
for X, y in valid_iter:
    X,y = X.to(device), y.to(device)
    y_hat = combine(X)
    acc += accuracy(y_hat, y)
    count += 1
    del X, y, y_hat
acc /= count
print(acc)
del validDataSet, valid_iter

print('-----------------------RESULT-------------------------')
print('create .json file')
revReflect = {v : k for k, v in testDataSet.category_reflect.items()}
for X, y in test_iter:
    X,y = X.to(device), y.to(device)
    y_hat = combine(X).argmax(dim=1).cpu().detach().numpy().tolist()

image_cate = sorted(zip(testDataSet.image_order, idx2name(y_hat,revReflect)), key=lambda x:int(x[0]))
result = {str(image_index): str(category) for image_index, category in image_cate }
with open('test_data.json', 'w') as file:
    json.dump(result, file)
    print(result)
print('-------------------------finish------------------------')
