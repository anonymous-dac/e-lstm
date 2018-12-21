import os,sys 
sys.path.append(os.getcwd())
import logging
import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms
from model import RNN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


sequence_length = 28 
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100 
num_epochs = 20
learning_rate = 0.001 
torch.manual_seed(1111)

log_dir = './MNIST/logs/'
model_dir = './MNIST/models/'

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

###############################################################################
# Logging
###############################################################################
model_config_name = 'nhid:{}-nlayer:{}-epoch:{}'.format(
    hidden_size, num_layers, num_epochs
)

log_file_name = "{}.log".format(model_config_name)
log_file_path = os.path.join(log_dir, log_file_name)
logging.basicConfig(filename=log_file_path, 
                    level=logging.INFO, datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    )
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(stream=sys.stdout))


###############################################################################
# Loading data
###############################################################################

train_dataset = torchvision.datasets.MNIST(root='./MNIST/data/', train=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root='./MNIST/data/', train=False, transform=transforms.ToTensor())


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    # states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
    #           torch.zeros(num_layers, batch_size, hidden_size).to(device))
    hidden = model.init_hidden(batch_size)

    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        hidden = repackage_hidden(hidden)
        # Forward pass
        outputs, hidden = model(images, hidden)
        loss = criterion(outputs, labels)
        l1_regularization = torch.tensor(0, dtype=torch.float32).to(device)
        lambda_ls = torch.tensor(0.0001).to(device)
        for param in model.parameters():
            l1_regularization += torch.norm(param, 1).to(device)
        loss = loss + l1_regularization * lambda_ls
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            logger.info('|Epoch [{}/{}] | Step [{}/{}] | Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

logger.info("Training process finish")
# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    # states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
    #           torch.zeros(num_layers, batch_size, hidden_size).to(device))
    hidden = model.init_hidden(batch_size)
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        hidden = repackage_hidden(hidden)
        outputs, hidden = model(images, hidden)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    logger.info('Test Accuracy of the model on the 10000 test images: {:.5f}'.format(100.0 * correct / total)) 

torch.save(model.state_dict(), './MNIST/models/{}.ckpt'.format(model_config_name))