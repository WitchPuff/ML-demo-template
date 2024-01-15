from models import AlexNet, DenseNet
from utils import train, predict
from utils import get_loaders
dataset = 'data\kvasir-dataset-v2'
model = DenseNet()
img_size = model.input_size
trainset, validset, testset = get_loaders(img_size=model.input_size, batch_size=16)
# train(model, trainset, validset, testset, epochs=40, pretrained=True, initial_lr=1e-4, weight_decay=1e-5)
predict(model, r'data\kvasir-dataset-v2\dyed-lifted-polyps\0a447c72-3a6f-43ac-a236-ca588f0435d4.jpg')