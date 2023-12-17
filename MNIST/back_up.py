def test_new():
    def test_net(num,data,net):
        with torch.no_grad():
            print(net.net.name)
            pred,feature = net.net(data)
            pred = F.softmax(pred,dim=1)
            #print(pred)
            sum = torch.sum(pred, 0)
            max_idxs = pred.argmax(1)
            logits = torch.zeros((len(pred),10))
            for i in range(len(pred)):
                logits[i][int(max_idxs[i])] = 1
            logits=torch.sum(logits, 0)
            print(sum)
            print(logits)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    net1 = Net(net = FNN_1(),load = True,model_path=current_path+'/model/')
    net2 = Net(net = ResNet_1(),load = True,model_path=current_path+'/model/')
    #net2 = Net(net = FNN_2(),load = True,model_path=current_path+'/model/')
    net1.net.eval()
    net2.net.eval()
    #training_data = datasets.MNIST(root=upper_upper_path+"/datasets",train=True,download=True,transform=ToTensor(),)
    test_data = datasets.MNIST(root=upper_upper_path+"/datasets",train=False,download=True,transform=ToTensor(),)
    #test_data = reset_dataset(test_data,[0])
    for i in range(10):
        num = i
        idx = test_data.targets==num
        data = test_data.data[idx]
        data = data[:,None,:]
        data = data.to(torch.float32)
        data = data.to(device)
        data = data/255
        #print(tmp)
        print(num,len(data))
        #test_net(num,data,net1)
        test_net(num,data,net2)
        print()
        #break