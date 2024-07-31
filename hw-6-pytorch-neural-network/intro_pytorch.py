import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def get_data_loader(training = True):
	custom_transform=transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
		])

	train_set=datasets.FashionMNIST('./data',train=True,download=True,transform=custom_transform)
	test_set=datasets.FashionMNIST('./data',train=False,transform=custom_transform)
	
	if(training == True):
		loader = torch.utils.data.DataLoader(train_set, batch_size = 64)
	else:
		loader = torch.utils.data.DataLoader(test_set, batch_size = 64)

	return loader


def build_model():
	model = nn.Sequential(
		nn.Flatten(),
		nn.Linear(28*28, 128),
		nn.ReLU(),
		nn.Linear(128, 64),
		nn.ReLU(),
		nn.Linear(64, 10)
	)
	return model



def train_model(model, train_loader, criterion, T):
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	model.train()

	for epoch in range(T):
		total = 0
		correct = 0
		running_loss = 0.0
		for i, data in enumerate(train_loader, 0):
			inputs, labels = data
			optimizer.zero_grad()
			
			outputs = model(inputs)

			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
				
			running_loss += loss.item()
			
		print("Train Epoch: " + str(epoch) + "\tAccuracy: " + str(correct) + "/" + str(total) + "(" + str(round(correct * 100 / total,2)) + "%) Loss: " + str(round(running_loss/len(train_loader),3)))

def evaluate_model(model, test_loader, criterion, show_loss = True):
	model.eval()

	correct = 0
	total = 0
	running_loss = 0.0	
	
	with torch.no_grad():
		for data in test_loader:
			inputs, labels = data
			outputs = model(inputs)
			
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

			loss = criterion(outputs, labels)
			running_loss += loss.item()
	
	if(show_loss == True):
		print("Average loss: " + str(round(running_loss/len(test_loader),4)))
	print("Accuracy: " + str(round(correct * 100 / total, 2)) + "%")
	
def predict_label(model, test_images, index):
	model.eval()
	with torch.no_grad():
		outputs = model(test_images[index].unsqueeze(0))
		prob = F.softmax(outputs, dim=1)[0]
		top_prob, top_indices = torch.topk(prob, 3)
		class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
		for prob, index in zip(top_prob, top_indices):
			print(f"{class_names[index]}: {prob.item()*100:.2f}%")

"""
def main():
	train_loader = get_data_loader()
	#print(type(train_loader))
	#print(train_loader.dataset)
	test_loader = get_data_loader(False)

	model = build_model()
	#print(model)

	criterion = nn.CrossEntropyLoss()
	train_model(model, train_loader, criterion, 5)

	evaluate_model(model, test_loader, criterion, show_loss = False)
	evaluate_model(model, test_loader, criterion, show_loss = True)

	test_images = next(iter(test_loader))[0]
	predict_label(model, test_images, 1)

main()
"""
