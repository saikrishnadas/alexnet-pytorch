#import packages
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets,transforms
from torch import optim

#switch to gpus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#parameters
PATH_TO_TRAINING_IMG = "your training folder"
PATH_CHECKPOINT_OUTPUT = "your output folder"
NUM_CLASSES = 1000
DEVICE_IDS = [0,1,2,3] #1 gpus [0],2 gpus [0,1]
IMG_SIZE = 227
BATCH_SIZE = 128 #reduce the batch size if "Cuda out of memory"

NUM_OF_EPOCHS = 90

class AlexNet(nn.Module):

  def __init__(self,num_classes=NUM_CLASSES):

    super().__init()
    #input size of 3 x 227 x 227 
    self.net = nn.Sequential(
        nn.Conv2d(3,96,kernel_size=11,stride=4),#(b x 96 x 55 x 55)
        nn.ReLU(),
        nn.LocalResponseNorm(n=5,alpha=0.0001,beta=0.75,k=2),
        nn.MaxPool2d(3,2),#(b x 96 x 27 x 27)
        nn.Conv2d(96,256,kernel_size=5,padding=2),#(b x 256 x 27 x 27)
        nn.ReLU(),
        nn.LocalResponseNorm(n=5,alpha=0.0001,beta=0.75,k=2),
        nn.MaxPool2d(3,2),#(b x 256 x 13 x 13)
        nn.Conv2d(256,384,kernel_size=3,padding=2), #(b x 384 x 13 x 13)
        nn.ReLU(),
        nn.Conv2d(384,384,kernel_size=3,padding=2),#(b x 384 x 13 x 13)
        nn.ReLU(),
        nn.Conv2d(384,256,kernel_size=3,padding=2),#(b x 256 x 13 x 13)
        nn.ReLU(),
        nn.MaxPool2d(3,2)#(b x 256 x 6 x 6)
    )
  #Linear layers or fully-connected layers
  self.Classifer = nn.Sequential(
      nn.Dropout(0.5,inplace=True),
      nn.Linear(in_features =(256*6*6),out_features= 4096),
      nn.ReLU(),
      nn.Dropout(0.5,inplace=True),
      nn.Linear(in_features = 4096,out_features= 4096),
      nn.ReLU(),
      nn.Dropout(0.5,inplace=True),
      nn.Linear(in_features =4096,out_features= 1000),
  )

  self.init_bias()
  #initialize bias
  def init_bias(self):
    for layer in self.net:
        if isinstance(layer, nn.Conv2d):
            nn.init.normal_(layer.weight, mean=0, std=0.01)
            nn.init.constant_(layer.bias, 0)
    # bias=1 for conv2d 2nd, 4th, and 5th layers
    nn.init.constant_(self.net[4].bias, 1) #4th layer
    nn.init.constant_(self.net[10].bias, 1) #10th layer
    nn.init.constant_(self.net[12].bias, 1) #12th layer

    def forward(self,x):
      #x is the input tensor
      x = self.net(x)
      x = x.view(-1,256 * 6 * 6) #flatten the input
      
      return self.Classifer(x) #output tensor


if __name__ == '__main__':

  seed = torch.initial_seed()
  print('Seed Used {}'.format(seed))

  #create the model
  alexnet = AlexNet(num_classes = NUM_CLASSES).to(device)
  #Train on multiple GPUs
  alexnet = torch.nn.parallel.DataParallel(alexnet,device_ids=DEVICE_IDS) 
  print(alexnet)
  print("Model Created!")

  #transform the input data
  transforms = transforms.Compose([
                                   transforms.CenterCrop(IMG_SIZE),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ColorJitter(contrast=0.1,brightness=0.1),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  #create training dataset
  dataset = datasets.ImageFolder(PATH_TO_TRAINING_IMG,transforms=transforms)

  #load data
  dataloader = torch.utils.data.DataLoader(datasets,shuffle=True,pin_memory=True,num_wokers = 8,batch_size=BATCH_SIZE)
  print("Data Loaded!")

  optimizer = optim.Adam(alexnet.parameters(),lr=0.0001)
  print("Optimizer created!")

  #define loss function
  criterion = nn.CrossEntropyLoss()
  print("Loss function applied!")

  #learning rate decay
  lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=30,gamma=0.1)
  print("Learning Rate decay scheduled!")

  #start training
  Print("Training Started!......")

  # number of epochs to train the model 
  n_epochs = NUM_OF_EPOCHS
  total_step = 1

  for epoch in range(1, n_epochs):
    lr_schedulers.step()
    # keep track of training and validation loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    model.train()
    for data, target in dataloader:
        # move tensors to GPU if CUDA is available
        data, target = data.to(device), target.to(device)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = alexnet(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)

        total_steps += 1

    # calculate average losses
    train_loss = train_loss/len(dataloader)
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \Total Steps Loss: {:.6f}'.format(
        epoch, train_loss, total_steps))
    
    
    #Save the model checkpoints
    checkpoint_path = os.path.join(PATH_CHECKPOINT_OUTPUT, 'alexnet_{}.pkl'.format(epoch + 1))
    state = {
            'epoch': epoch,
            'total_steps': total_steps,
            'optimizer': optimizer.state_dict(),
            'model': alexnet.state_dict(),
            'seed': seed,
        }
    torch.save(state, checkpoint_path)

  