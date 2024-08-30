import torch
import datetime
import pandas as pd
from torch.utils.data import Dataset, random_split
import torch.optim as optim
from torchvision.transforms import Compose, CenterCrop, Resize

#check if cuda available
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

#set hyperparameters to search - this was cleared for final training
lr = [0.0001]
ts0 = datetime.datetime.now().strftime('%a%d%b%Y_%I_%M')

for l in lr:
    #timestamp for current set of hyper parameters
    ts = datetime.datetime.now().strftime('%a%d%b%Y_%I_%M')

    #set # of epochs and batch size
    epochs = 10
    batch_size = 32

    #load training set
    trainset = nn_safe_reg.NNDataset(annotations_file="new_train_data.csv",img_dir="pace",
                                transform=Compose([CenterCrop((320, 320)), Resize((160, 160))]))

    #split into training and validation
    train, valid = random_split(trainset, [25000, 1141])
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(valid, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    #initialize network for training
    net = nn_safe_reg.NN().to(device)
    optimizer = optim.Adam(net.parameters(), lr=l)
    net.train(True)

    #set training loss criteria
    criteria = torch.nn.MSELoss()
    valid_criteria = torch.nn.MSELoss()

    for epoch in range(epochs):
        #reset running loss each epoch
        running_loss = 0
        valid_loss = 0

        #count # of samples
        num_samps = 0
        valid_num_samps = 0

        for i, data in enumerate(trainloader, 0):
            images, u, labels = data
            optimizer.zero_grad()

            output = net(images.to(device), u.to(device))
            loss = criteria(output, labels.to(device).float())

            #back propagation
            loss.backward()
            optimizer.step()

            #compute loss per sample
            running_loss += loss.item() * batch_size
            num_samps += batch_size
            loss_per_samp = running_loss/num_samps

        for i, data in enumerate(validloader, 0):
            #find validation loss
            images, u, labels = data

            output = net(images.to(device), u.to(device))
            loss = valid_criteria(output, labels.to(device).float())

            valid_loss += loss.item() * batch_size
            valid_num_samps += batch_size
            valid_loss_per_samp = valid_loss/valid_num_samps

        #print loss data
        print(f'[Epoch{epoch + 1:5d}], training loss: {loss_per_samp}')
        print(f'[Epoch{epoch + 1:5d}], validation loss: {valid_loss_per_samp}')

        #record loss data
        loss_data = [{"epoch": epoch,
                      "Train loss": loss_per_samp,
                      "Valid loss": valid_loss_per_samp}]
        df = pd.DataFrame(loss_data)
        df.to_csv(f'./{ts}_{l}_nn.csv', mode='a', index=False, header=False)

    #save final model
    PATH = f'./{l}_nn.pth'
    torch.save(net, PATH)

    #save loss data in csv
    train_data = [{"lr":l,
                   "Train loss": loss_per_samp,
                   "Valid loss": valid_loss_per_samp}]
    df2 = pd.DataFrame(train_data)
    df2.to_csv(f'./train_data_{ts0}_nn.csv', mode='a', index=False, header=False)
