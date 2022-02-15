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
import dgc.util as util
import dgc.likelihoods as likelihoods
import dgc.debug_tools as tools
from torch.utils.data import TensorDataset, DataLoader, Dataset


class dgc(nn.Module):
    def __init__(self, input_dim=784, y_dim = 100, z_dim=10, n_centroids=10, task = 'regression', binary=True,
        multi_decoder = False, encodeLayer=[500,500,2000], decodeLayer=[2000,500,500]):
        super(self.__class__, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.n_centroids = n_centroids
        self.multi_decoder = multi_decoder
        self.task = task
        #self.encoder = util.bigEncNet(nChannels)
        #res18 = models.resnet18(pretrained=False)
        #self.encoder = util.res_cutlayers(res18,1)
        self.encoder = util.buildNetwork([input_dim] + encodeLayer,activation="sigmoid")
        self.decoder = util.buildNetwork([z_dim] + decodeLayer,activation='sigmoid')
        # predicting latent parameters
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._enc_log_sigma = nn.Linear(encodeLayer[-1], z_dim)
        # last layer of the decoder
        if self.multi_decoder:
            self._dec = nn.Linear(decodeLayer[-1], input_dim*self.n_centroids)
        else:
            self._dec = nn.Linear(decodeLayer[-1], input_dim)
        self._dec_act = None



        if self.task == 'regression':
            # Do regression
            #self.predict_prob = nn.Sequential(util.buildNetwork([input_dim] + encodeLayer),nn.Linear(encodeLayer[-1], z_dim),nn.Softmax(dim=1))
            self.predict_prob = nn.Sequential(nn.Linear(encodeLayer[-1], self.n_centroids),nn.Softmax(dim=1))
            #self.prob_ensemble = nn.Sequential(nn.Linear(self.z_dim, self.y_dim*self.n_centroids),nn.Sigmoid())
            self.out_mu = nn.Sequential(
                #nn.Linear(z_dim,128),
                #nn.ReLU(),
                nn.Linear(z_dim,self.y_dim*self.n_centroids),
            #    nn.ReLU(),
            #    nn.Linear(256,256),
            #    nn.ReLU(),
            #    nn.Linear(256,256),
            #    nn.ReLU(),
            #    nn.Linear(256,128),
            #    nn.ReLU(),
            #    nn.Linear(128,self.y_dim*self.n_centroids),
                #nn.Sigmoid(),(),
            )
            self.out_log_sigma = nn.Linear(z_dim,self.y_dim*self.n_centroids)
            #self.prob_ensemble = util.buildNetwork([z_dim,512,self.y_dim*self.n_centroids])
        else:
            # do classification
            self.prob_ensemble = nn.Linear(self.z_dim, self.y_dim*self.n_centroids)


        if binary:
            self._dec_act = nn.Tanh()

        self.u_p, self.theta_p, self.lambda_p = util.create_gmmparam(self.n_centroids, self.z_dim)



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
        for i, batch in enumerate(loader):
            inputs = batch[0].float()
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




    def forward(self,x,train=False):
        h = self.encoder(x)
        mu = self._enc_mu(h)
        logvar = self._enc_log_sigma(h)
        prob = self.predict_prob(h)
        z = util.reparameterize(mu, logvar,train)
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

    def fit(self, trainloader, validloader,  lr=0.001, num_epochs=10,
                  anneal=False, direct_predict_prob=False, extractor = None):
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
            #new_order = np.arange(len(trainloader))
            cluster_train = []
            cluster_truth = []
            side_info_true = []
            side_info_pred = []
            side_info_true_test = []
            side_info_pred_test = []
            temp_latent = []
            temp_label = []
            temp_mean = []
            temp_logvar = []
            posterior_prob = []
            posterior_prob_test = []
            prior_mu.append(self.u_p.data.detach().cpu().numpy().transpose(1,0))
            prior_var.append(self.lambda_p.data.detach().cpu().numpy().transpose(1,0))

            for i, (inputs,y_label,cluster_index) in enumerate(trainloader):
                b_size = inputs.shape[0]
                side_info_true.append(y_label.numpy())
                cluster_index = cluster_index.numpy()
                y_label = y_label.unsqueeze(1).float()
                inputs = inputs.float()
                
                if use_cuda:
                    inputs = inputs.cuda()
                    y_label = y_label.cuda()

                optimizer.zero_grad()

                #self.check_nan()
                z, outputs, mu, logvar, prob = self.forward(inputs,True)
                y_preds_mu = torch.sigmoid(self.out_mu(z))
                y_preds_log_var = self.out_log_sigma(z)
                # Calculate response likelihood under different clusters
                log_y_likelihood = util.ensemble_gaussian_loglike(y_label,y_preds_mu,y_preds_log_var,self.n_centroids)


                temp_label.append(cluster_index)
                temp_latent.append(z.detach().cpu().numpy())
                temp_mean.append(mu.detach().cpu().numpy())
                temp_logvar.append(logvar.detach().cpu().numpy())

                #y_preds = self.prob_ensemble(z)

                if direct_predict_prob:
                    lambda_k = prob
                else:
                    #lambda_k = util.dgc_get_lambda_k(z, mu, logvar, y_preds,
                    #    self.u_p, self.lambda_p, self.theta_p, self.n_centroids)
                    lambda_k = util.dgc_get_lambda_k(z, log_y_likelihood, self.u_p, self.lambda_p, self.theta_p, self.n_centroids)

                if self.multi_decoder:
                    outputs = outputs.reshape(outputs.shape[0],self.n_centroids,self.input_dim)
                    outputs = torch.sum(outputs*lambda_k.unsqueeze(-1),1)

                loss = util.dgc_loss_function(outputs, inputs, z, mu, logvar, log_y_likelihood,
                    lambda_k, self.u_p, self.lambda_p, self.theta_p, self.n_centroids)
                #loss += torch.mean(entropy_regu)
                #loss,bce,logpzc,qentropy,logpc,logqcx = util.vade_loss_function(outputs, inputs, z, mu, logvar, 
                #    lambda_k, self.u_p, self.lambda_p, self.theta_p, self.n_centroids)

                train_loss += loss.item()
                loss.backward()
                optimizer.step()

                #self.lambda_p.data.clamp_(1e-3)
                #self.theta_p.data = torch.softmax(self.theta_p.data,0)

                lambda_k = lambda_k.data.cpu().numpy()
                posterior_prob.append(lambda_k)
                side_info_pred.append(y_preds_mu.detach().cpu().numpy())
                cluster_train.append(np.argmax(lambda_k[:len(cluster_index)], axis=1))
                cluster_truth.append(cluster_index.astype(np.int))

            overall_label.append(np.concatenate(temp_label))
            overall_latent.append(np.concatenate(temp_latent))
            overall_mean.append(np.concatenate(temp_mean))
            overall_logvar.append(np.concatenate(temp_logvar))

            #print(count/(len(trainloader)*256))
            side_info_pred = np.concatenate(side_info_pred)
            side_info_true = np.concatenate(side_info_true)
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
            #print(debug_tools.discrete_entropy(cluster_train))
            #print(debug_tools.check_clus_consistency(cluster_train,cluster_truth,summary=True,approx_true_error=True))
            #print(debug_tools.check_confidence(posterior_prob,cluster_train,cluster_truth))
            print(acc[0])
            #print(util.pred_acc(side_info_pred,posterior_prob,side_info_true,'multi_binary_classification','max'))
            # validate
            self.eval()
            valid_loss = 0.0
            cluster_test = []
            cluster_truth = []
            for i, (inputs,test_y_label,cluster_index) in enumerate(validloader):
                b_size = inputs.shape[0]
                inputs = inputs.float() 
                cluster_index = cluster_index.numpy()
                side_info_true_test.append(test_y_label.numpy())
                #test_y_label = test_y_label.unsqueeze(1).float()
                if use_cuda:
                    inputs = inputs.cuda()
                    if extractor:
                        extractor = extractor.cuda()
                if extractor:
                    with torch.no_grad():
                        y_label = extractor(inputs)
                        y_label = y_label.unsqueeze(1)
                        if use_cuda:
                            y_label = y_label.cuda()
                    y_true_label = test_y_label.numpy()
                else:
                    y_label = test_y_label.unsqueeze(1).float()
                    if use_cuda:
                        y_label = y_label.cuda()


                #self.check_nan()
                z, outputs, mu, logvar, prob = self.forward(inputs,True)
                y_preds_mu = torch.sigmoid(self.out_mu(z))
                y_preds_log_var = self.out_log_sigma(z)
                # Calculate response likelihood under different clusters
                log_y_likelihood = util.ensemble_gaussian_loglike(y_label,y_preds_mu,y_preds_log_var,self.n_centroids)


                if direct_predict_prob:
                    lambda_k = prob
                else:
                    #lambda_k = util.dgc_get_lambda_k(z, mu, logvar, y_preds,
                    #    self.u_p, self.lambda_p, self.theta_p, self.n_centroids)
                    lambda_k = util.dgc_get_lambda_k(z, log_y_likelihood, self.u_p, self.lambda_p, self.theta_p, self.n_centroids)

                if self.multi_decoder:
                    outputs = outputs.reshape(outputs.shape[0],self.n_centroids,self.input_dim)
                    outputs = torch.sum(outputs*lambda_k.unsqueeze(-1),1)

                loss = util.dgc_loss_function(outputs, inputs, z, mu, logvar, log_y_likelihood,
                    lambda_k, self.u_p, self.lambda_p, self.theta_p, self.n_centroids)
                #loss += torch.mean(entropy_regu)
                #loss,bce,logpzc,qentropy,logpc,logqcx = util.vade_loss_function(outputs, inputs, z, mu, logvar, 
                #    lambda_k, self.u_p, self.lambda_p, self.theta_p, self.n_centroids)

                valid_loss += loss.item()

                lambda_k = lambda_k.data.cpu().numpy()
                cluster_test.append(np.argmax(lambda_k, axis=1))
                cluster_truth.append(cluster_index.astype(np.int))
                posterior_prob_test.append(lambda_k)
                side_info_pred_test.append(y_preds_mu.detach().cpu().numpy())
                if extractor:
                    side_info_true_test.append(y_true_label.numpy())

            cluster_test = np.concatenate(cluster_test)
            cluster_truth = np.concatenate(cluster_truth)
            side_info_pred_test = np.concatenate(side_info_pred_test)
            side_info_true_test = np.concatenate(side_info_true_test)
            posterior_prob_test = np.concatenate(posterior_prob_test)
            #if epoch>0:
            #    print(np.sum(cluster_test==cluster_test_prev)/len(cluster_test))
            cluster_test_prev = cluster_test
            acc = util.cluster_acc(cluster_test,cluster_truth)
            #print(util.pred_acc(side_info_pred_test,posterior_prob_test,side_info_true_test,'multi_binary_classification','max'))
            print("#Epoch %3d: lr: %.5f, Train Loss: %.5f, Valid Loss: %.5f, Cluster ACC:%.5f" % (
                epoch, epoch_lr, train_loss / len(trainloader), valid_loss / len(validloader), acc[0]))
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




