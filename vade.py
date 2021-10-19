import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import datasets, models, transforms
import numpy as np
import math
from sklearn.mixture import GaussianMixture
import util
import debug_tools
from torch.utils.data import TensorDataset, DataLoader, Dataset


class vade(nn.Module):
    def __init__(self, input_dim=784, z_dim=10, n_centroids=10, binary=True,
        encodeLayer=[500,500,2000], decodeLayer=[2000,500,500]):
        super(self.__class__, self).__init__()
        self.z_dim = z_dim
        self.n_centroids = n_centroids
        #self.encoder = util.bigEncNet(nChannels)
        #res18 = models.resnet18(pretrained=False)
        #self.encoder = util.res_cutlayers(res18,1)
        self.encoder = util.buildNetwork([input_dim] + encodeLayer)
        self.decoder = util.buildNetwork([z_dim] + decodeLayer)
        # predicting latent parameters
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._enc_log_sigma = nn.Linear(encodeLayer[-1], z_dim)
        # last layer of the decoder
        self._dec = nn.Linear(decodeLayer[-1], input_dim)
        self._dec_act = None
        #self.predict_prob = nn.Sequential(util.buildNetwork([input_dim] + encodeLayer),nn.Linear(encodeLayer[-1], z_dim),nn.Softmax(dim=1))
        self.predict_prob = nn.Sequential(nn.Linear(z_dim, n_centroids),nn.Softmax(dim=1))

        if binary:
            self._dec_act = nn.Sigmoid()

        self.u_p, self.theta_p, self.lambda_p = util.create_gmmparam(self.n_centroids, self.z_dim, False)


    def pretrain_model(self,loader,num_epochs,b_size):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=0.002)
        mse_loss = nn.MSELoss(reduction='sum')      
        train_loss = 0
        self.train()
        for epoch in range(num_epochs):
            print(epoch)
            for i,(inputs,cluster_index) in enumerate(loader):
                inputs = inputs.float()
                if use_cuda:
                    inputs = inputs.cuda()
                optimizer.zero_grad()
                z, outputs, mu, logvar, prob = self.forward(inputs)
                loss = mse_loss(outputs,inputs)
                train_loss += loss.item()*len(inputs)
                loss.backward()
                optimizer.step()
                # print("    #Iter %3d: Reconstruct Loss: %.3f" % (
                #     batch_idx, recon_loss.item()))


    def initialize_gmm(self, loader,b_size):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        data = []
        self.eval()
        for i, (inputs,cluster_index) in enumerate(loader):
            inputs = inputs.float()
            if use_cuda:
                inputs = inputs.cuda()
            z, outputs, mu, logvar, prob = self.forward(inputs)
            data.append(z.data.cpu().numpy())
        data = np.concatenate(data)
        gmm = GaussianMixture(n_components=self.n_centroids,covariance_type='diag')
        gmm.fit(data)
        self.u_p.data.copy_(torch.from_numpy(gmm.means_.T.astype(np.float32)))
        self.lambda_p.data.copy_(torch.from_numpy(gmm.covariances_.T.astype(np.float32)))
        self.theta_p.data.copy_(torch.from_numpy(gmm.weights_.T.astype(np.float32)))




    def shuffle(self,length,seed):
        np.random.seed(seed)
        new_order = list(np.arange(length))
        np.random.shuffle(new_order)
        return new_order



    def forward(self,x,train=False):
        h = self.encoder(x)
        mu = self._enc_mu(h)
        logvar = self._enc_log_sigma(h)
        z = util.reparameterize(mu, logvar,train)
        prob = self.predict_prob(z)
        return z, self.decode(z), mu, logvar, prob



    def decode(self,z):
        h = self.decoder(z)
        x = self._dec(h)
        if self._dec_act is not None:
            x = self._dec_act(x)
        return x
        
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def fit(self, trainloader, validloader, which_cluster=0, lr=0.001, b_size=128, num_epochs=10, 
                  anneal=False, direct_predict_prob=False):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, weight_decay=1e-4)
        cluster_train_prev = 0
        cluster_test_prev = 0
        prior_mu = []
        prior_var = []
        overall_latent = []
        overall_label = []
        overall_mean = []
        overall_logvar = []
        for epoch in range(num_epochs):
            # train 1 epoch
            count = 0
            self.train()
            if anneal:
                epoch_lr = util.adjust_learning_rate(lr, optimizer, epoch)
            train_loss = 0
            #new_order = np.array(self.util.shuffle(len(trainloader),epoch))
            cluster_train = []
            cluster_truth = []
            temp_latent = []
            temp_label = []
            temp_mean = []
            temp_logvar = []
            posterior_prob = []
            prior_mu.append(self.u_p.data.detach().cpu().numpy().transpose(1,0))
            prior_var.append(self.lambda_p.data.detach().cpu().numpy().transpose(1,0))

            for i, (inputs,cluster_index) in enumerate(trainloader):
                cluster_index = cluster_index.numpy()[:,which_cluster]
                inputs = inputs.float()
                
                if use_cuda:
                    inputs = inputs.cuda()

                optimizer.zero_grad()
                #self.check_nan()
                z, outputs, mu, logvar, prob = self.forward(inputs,True)

                temp_label.append(cluster_index)
                temp_latent.append(z.detach().cpu().numpy())
                temp_mean.append(mu.detach().cpu().numpy())
                temp_logvar.append(logvar.detach().cpu().numpy())

                if direct_predict_prob:
                    lambda_k = prob
                else:
                    lambda_k = util.vade_get_lambda_k(z, self.u_p, self.lambda_p, self.theta_p, self.n_centroids)

                loss,bce,logpzc,qentropy,logpc,logqcx = util.vade_loss_function(outputs, inputs, z, mu, logvar, 
                    lambda_k, self.u_p, self.lambda_p, self.theta_p, self.n_centroids)
                count += debug_tools.check_prob_kl(mu, logvar, lambda_k, self.u_p, self.lambda_p, self.theta_p,self.n_centroids)

                train_loss += loss.item()
                loss.backward()
                optimizer.step()

                #self.lambda_p.data.clamp_(1e-5)
                #self.theta_p.data = torch.softmax(self.theta_p.data,0)

                lambda_k = lambda_k.data.cpu().numpy()
                posterior_prob.append(lambda_k)
                cluster_train.append(np.argmax(lambda_k, axis=1))
                cluster_truth.append(cluster_index.astype(np.int))

            overall_label.append(np.concatenate(temp_label))
            overall_latent.append(np.concatenate(temp_latent))
            overall_mean.append(np.concatenate(temp_mean))
            overall_logvar.append(np.concatenate(temp_logvar))

            print(count/(len(trainloader)*256))

            cluster_train = np.concatenate(cluster_train)
            cluster_truth = np.concatenate(cluster_truth)
            posterior_prob = np.concatenate(posterior_prob)
            acc = util.cluster_acc(cluster_train,cluster_truth)
            #print(cluster_train)
            #if epoch>0:
            #    print(np.sum(cluster_train==cluster_train_prev)/len(cluster_train))
            cluster_train_prev = cluster_train
            #print(len(np.where(cluster_train==0)[0])/len(cluster_train))
            #print(len(np.where(cluster_train==1)[0])/len(cluster_train))
            #print(len(np.where(cluster_train==2)[0])/len(cluster_train))
            #sprint(debug_tools.discrete_entropy(cluster_train))
            #print(debug_tools.check_clus_consistency(cluster_train,cluster_truth,summary=True,approx_true_error=True))
            #print(debug_tools.check_confidence(posterior_prob,cluster_train,cluster_truth))
            print(acc[0])
            # validate
            self.eval()
            valid_loss = 0.0
            cluster_test = []
            cluster_truth = []

            for i, (inputs,cluster_index) in enumerate(validloader):
                inputs = inputs.float()
                cluster_index = cluster_index.numpy()[:,which_cluster]
                if use_cuda:
                    inputs = inputs.cuda()
                z, outputs, mu, logvar, prob = self.forward(inputs)

                #self.lambda_p.data.clamp_(1e-3)
                #self.theta_p.data = torch.softmax(self.theta_p.data,0)
                if direct_predict_prob:
                    lambda_k = prob
                else:
                    lambda_k = util.vade_get_lambda_k(z, self.u_p, self.lambda_p, self.theta_p, self.n_centroids)

                loss,bce,logpzc,qentropy,logpc,logqcx = util.vade_loss_function(outputs, inputs, z, mu, logvar, 
                    lambda_k, self.u_p, self.lambda_p, self.theta_p, self.n_centroids)

                valid_loss += loss.item()
                # total_loss += valid_recon_loss.item() * inputs.size()[0]
                # total_num += inputs.size()[0]

                lambda_k = lambda_k.data.cpu().numpy()
                cluster_test.append(np.argmax(lambda_k, axis=1))
                cluster_truth.append(cluster_index.astype(np.int))


            cluster_test = np.concatenate(cluster_test)
            cluster_truth = np.concatenate(cluster_truth)
            #if epoch>0:
            #    print(np.sum(cluster_test==cluster_test_prev)/len(cluster_test))
            cluster_test_prev = cluster_test
            acc = util.cluster_acc(cluster_test,cluster_truth)
            # valid_loss = total_loss / total_num
            print("#Epoch %3d: lr: %.5f, Train Loss: %.5f, Valid Loss: %.5f, Cluster ACC:%.5f" % (
                epoch, epoch_lr, train_loss / len(trainloader), valid_loss / len(validloader), acc[0] ))
        overall_label = np.array(overall_label)
        overall_latent = np.array(overall_latent)
        overall_mean = np.array(overall_mean)
        overall_logvar = np.array(overall_logvar)
        prior_mu = np.array(prior_mu)
        prior_var = np.array(prior_var)
        '''
        np.save("fix_1overall_label",overall_label)
        np.save("fix_1overall_latent",overall_latent)
        np.save("fix_1latent_mean",overall_mean)
        np.save("fix_1latent_logvar",overall_logvar)
        np.save("fix_1prior_mean",prior_mu)
        np.save("fix_1prior_var",prior_var)
        '''


