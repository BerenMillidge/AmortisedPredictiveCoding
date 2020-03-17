#experimenting with weight decay as in: https://arxiv.org/pdf/1910.12151.pdf
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import torch
from functions import *
import sys
import torchvision

class PredictiveCodingLayer(object):
  def __init__(self,input_size, output_size, batch_size, fn,fn_deriv,learning_rate,wd_lambda):
    self.input_size = input_size
    self.output_size = output_size
    self.batch_size = batch_size
    self.fn = fn
    self.fn_deriv = fn_deriv
    self.learning_rate = learning_rate
    self.wd_lambda = wd_lambda
    self.weights = np.random.normal(0,0.1,[input_size, output_size])
    print(input_size, batch_size)
    self.mu = np.random.normal(0,1,[output_size,batch_size])

  def update_mu(self, pe, pe_below):
    self.mu+= self.learning_rate * (-pe + (np.dot(self.weights.T, pe_below *self.fn_deriv(np.dot(self.weights, self.mu)))) - (self.wd_lambda * self.mu))

  def step(self,pe_below,pred,use_top_down_pe=True):
    if use_top_down_pe:
      pe = self.mu - pred
    else:
      pe = np.zeros_like(self.mu)
    self.mu+= self.learning_rate * (-pe + (np.dot(self.weights.T, pe_below *self.fn_deriv(np.dot(self.weights, self.mu)))) - (self.wd_lambda * self.weights))
    return pe, self.predict()

  def predict(self):
    return self.fn(np.dot(self.weights, self.mu))

  def update_weights(self,pe_below):
    self.weights += self.learning_rate * (np.dot(pe_below * self.fn_deriv(np.dot(self.weights, self.mu)), self.mu.T))

class PredictiveCodingNetwork(object):

  def __init__(self, layer_sizes,batch_size,learning_rate, f,df,n_inference_steps):
    self.layer_sizes = layer_sizes
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.f = f
    self.df = df
    self.n_inference_steps = n_inference_steps
    self.layers = []
    for i in range(len(self.layer_sizes)-1):
      self.layers.append(PredictiveCodingLayer(self.layer_sizes[i], self.layer_sizes[i+1], self.batch_size, self.f,self.df,self.learning_rate))
    self.predictions = [[]for i in range(len(self.layer_sizes))]
    self.prediction_errors = [[] for i in range(len(self.layer_sizes))]
    self.L = len(self.layers)

  def reset_mus(self):
    # to prevent horrible and weird state corruption/dependencies between trials which give all manner of lovely bugs.
    for layer in self.layers:
        layer.mu = np.random.normal(0,1,[layer.output_size,layer.batch_size])

  def infer(self, img_batch,label_batch):
    self._reset_mus()
    prediction_errors = [[] for i in range(self.L)]
    self.layers[-1].mu = deepcopy(label_batch)
    for i in reversed(range(self.L)):
      self.predictions[i] = self.layers[i].predict()
    for n in range(self.n_inference_steps):
      self.prediction_errors[0] = img_batch - self.predictions[0]
      self.predictions[-1] = np.zeros_like(self.layers[-1].mu)
      for l in range(self.L):
        self.prediction_errors[l+1] = self.layers[l].mu - self.predictions[l+1]
        if l !=self.L-1:
          self.layers[l]._update_mu(self.prediction_errors[l+1],self.prediction_errors[l])
        else:
          self.layers[l]._update_mu(np.zeros_like(self.layers[l].mu), self.prediction_errors[l])
        self.predictions[l] = self.layers[l].predict()
      self.layers[-1].mu = deepcopy(label_batch)
      self.predictions[-1] = self.layers[-1].mu

    for l in range(self.L):
      self.layers[l].update_weights(self.prediction_errors[l])
    return self.prediction_errors, self.predictions



  def test(self, img_batch, label_batch):
    self._reset_mus()
    prediction_errors = [[] for i in range(self.L)]
    for i in reversed(range(self.L)):
      self.predictions[i] = self.layers[i].predict()
    for n in range(self.n_inference_steps * 10):
      self.prediction_errors[0] = img_batch - self.predictions[0]
      self.predictions[-1] = np.zeros_like(self.layers[-1].mu)
      for l in range(self.L):
        self.prediction_errors[l+1] = self.layers[l].mu - self.predictions[l+1]
        if l !=self.L-1:
          self.layers[l]._update_mu(self.prediction_errors[l+1],self.prediction_errors[l])
        else:
          self.layers[l]._update_mu(np.zeros_like(self.layers[l].mu), self.prediction_errors[l])
        self.predictions[l] = self.layers[l].predict()


    pred_labels = self.layers[-1].mu
    pred_imgs = self.predictions[0]
    return pred_imgs, pred_labels

  def plot_batch_results(self,pred_labels,true_labels):
    batch_size = pred_labels.shape[1]
    for b in range(batch_size):
      plt.plot(pred_labels[:,b])
      plt.show()
      print(true_labels[:,b])


  def train(self, imglist, labellist, n_epochs):
    prediction_errors = []
    for n in range(n_epochs):
      print("Epoch ", n)
      batch_pes = []
      for (img_batch,label_batch) in zip(imglist, labellist):
        pes, preds = self.infer(img_batch,label_batch)

      if n % 10 == 0:
        tot_acc = 0
        for (img_batch, label_batch) in zip(imglist, labellist):
          pred_imgs, pred_labels = self.test(img_batch,label_batch)
          tot_acc += self.accuracy(pred_labels, label_batch)
        print("Accuracy: ", tot_acc/num_batches)

class AmortisationLayer(object):
  def __init__(self, forward_size, backward_size, learning_rate, batch_size, qf, dqf):
    self.forward_size = forward_size
    self.backward_size = backward_size
    self.learning_rate = learning_rate
    self.batch_szie = batch_size
    self.qf = qf
    self.dqf = dqf
    self.weights = np.random.normal(0,0.1,[self.forward_size, self.backward_size])

  def predict(self, state):
    return self.qf(np.dot(self.weights, state))

  def update_weights(self, amortised_prediction_errors, state):
    self.weights += self.learning_rate * (np.dot(amortised_prediction_errors * self.dqf(np.dot(self.weights, state)),state.T))


class AmortisedPredictiveCodingNetwork(object):
  def __init__(self, layer_sizes,batch_size,learning_rate,amortised_learning_rate,n_inference_steps_train,n_inference_steps_test, f,df,qf,dqf,inference_threshold=0.1,wd_lambda = 0.05):
    self.layer_sizes = layer_sizes
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.amortised_learning_rate = amortised_learning_rate
    self.n_inference_steps_train = n_inference_steps_train
    self.n_inference_steps_test = n_inference_steps_test
    self.f = f
    self.df = df
    self.qf = qf
    self.dqf = dqf
    self.inference_threshold = inference_threshold
    self.layers = []
    self.q_layers = []
    self.wd_lambda = wd_lambda
    for i in range(len(self.layer_sizes)-1):
      self.layers.append(PredictiveCodingLayer(self.layer_sizes[i], self.layer_sizes[i+1], self.batch_size, self.f,self.df,self.learning_rate,self.wd_lambda))
    #initialize amortised networks
    for i in range(len(self.layer_sizes)-1):
      self.q_layers.append(AmortisationLayer(self.layer_sizes[i+1], self.layer_sizes[i], self.amortised_learning_rate,self.batch_size, self.qf,self.dqf))
    self.predictions = [[]for i in range(len(self.layer_sizes))]
    self.prediction_errors = [[] for i in range(len(self.layer_sizes))]
    self.amortised_predictions = [[] for i in range(len(self.layer_sizes))]
    self.amortised_prediction_errors = [[] for i in range(len(self.layer_sizes))]
    self.L = len(self.layers)
    self.n_layers = len(self.layer_sizes)

  def reset_mus(self):
    for layer in self.layers:
        layer.mu = np.random.normal(0,1,[layer.output_size,layer.batch_size])



  def amortisation_pass(self, img_batch,initialize=True):
      self.amortised_predictions[0] = deepcopy(img_batch)
      for i in range(self.L):
          self.amortised_predictions[i+1] = self.q_layers[i].predict(self.amortised_predictions[i])
          self.layers[i].mu = deepcopy(self.amortised_predictions[i+1])



  def forward_pass(self, img_batch, label_batch, test=False):
    # reset model
    n_inference_steps = self.n_inference_steps_test if test else self.n_inference_steps_train
    self.reset_mus()
    prediction_errors = [[] for i in range(self.n_layers)]

    #set the highest level mus to the label if training. Else let them vary freely (they will become the prediction).
    #run an amortisation pass to initialize all variational parameters.
    self.amortisation_pass(img_batch,initialize=True)
    if not test:
        self.layers[-1].mu = deepcopy(label_batch)

    # variational predictions (predictions go top down so have to go through layers in reverse order)
    for i in reversed(range(self.L)):
        self.predictions[i] = self.layers[i].predict()

    # perform variational updates
    for n in range(n_inference_steps):
        # set lowest prediction errors from the input stimulus
        self.prediction_errors[0] = img_batch - self.predictions[0]
        #set highest predictions to zero since there are no top-down predictions entering the highest layer
        self.predictions[-1] = np.zeros_like(self.layers[-1].mu)
        #perform variational updates for each layer
        for i in range(self.L):
            self.prediction_errors[i+1] = self.layers[i].mu - self.predictions[i+1]
            if i != self.L-1:
                self.layers[i].update_mu(self.prediction_errors[i+1], self.prediction_errors[i])
            else:
                self.layers[i].update_mu(np.zeros_like(self.layers[i].mu), self.prediction_errors[i])
            self.predictions[i] = self.layers[i].predict()
            #set reset the final layer mus to the batch label
        if not test:
            self.layers[-1].mu = deepcopy(label_batch)
            self.predictions[-1] = deepcopy(self.layers[-1].mu)


        F = np.sum(np.array([np.mean(np.square(pe)) for pe in self.prediction_errors]))
        if F <= self.inference_threshold:
            break

    pred_imgs = deepcopy(self.predictions[0])
    pred_labels = deepcopy(self.layers[-1].mu)
    return pred_imgs, pred_labels

  def train_batch(self, img_batch, label_batch):
    self.forward_pass(img_batch, label_batch, test=False)
    for l in range(self.L):
        self.layers[l].update_weights(self.prediction_errors[l])
        self.amortised_prediction_errors[l] = self.layers[l].mu - self.amortised_predictions[l+1]
        if l == 0:
            self.q_layers[l].update_weights(self.amortised_prediction_errors[l], self.amortised_predictions[0])
        else:
            self.q_layers[l].update_weights(self.amortised_prediction_errors[l], self.layers[l-1].mu)

    return self.prediction_errors, self.predictions,self.amortised_predictions, self.amortised_prediction_errors


  def test(self, img_batch, label_batch):
    pred_imgs, pred_labels  = self.forward_pass(img_batch, label_batch,test=True)
    return pred_imgs, pred_labels

  def amortised_test(self, img_batch):
    self.amortisation_pass(img_batch,initialize=True)
    return self.amortised_predictions[-1]

  def generate_amortised(self, img_batch):
      self.amortisation_pass(img_batch, initialize=True)
      for i in reversed(range(self.L)):
          self.predictions[i] = self.layers[i].predict()
      return self.predictions[0] #these should be the generated versions

  def generate_variational(self, label_batch):
      #how is this going to work? we could just do downwawrd predictions... will this work or should there be some iterations?
      self.layers[-1].mu = deepcopy(label_batch)
      for i in reversed(range(self.L)):
          self.predictions[i] = self.layers[i].predict()
      return self.predictions[0] #these should be the generated versions

  def plot_batch_results(self,pred_labels,amortised_labels, true_labels):
    batch_size = pred_labels.shape[1]
    for b in range(batch_size):
      print("True labels: ", true_labels[:,b])
      print("Variational predictions: ")
      plt.plot(pred_labels[:,b])
      plt.show()
      print("Amortised Predictions: ",)
      plt.plot(amortised_labels[:,b])
      plt.show()

  def accuracy(self,pred_labels, true_labels):
      #print("IN accuracy", pred_labels, true_labels)
      correct = 0
      batch_size = pred_labels.shape[1]
      for b in range(batch_size):
        if np.argmax(pred_labels[:,b]) == np.argmax(true_labels[:,b]):
          correct +=1
      return correct / batch_size


  def train(self, imglist, labellist,test_img_list, test_label_list, n_epochs,save_name):
    prediction_errors = []
    amortised_prediction_errors = []
    variational_accs = []
    amortised_accs = []
    test_variational_accs = []
    test_amortised_accs = []
    for n in range(n_epochs):
        print("Epoch ", n)
        batch_pes = []
        batch_qpes = []
        for (img_batch,label_batch) in zip(imglist, labellist):
            pes, preds,qpreds, qpes = self.train_batch(img_batch,label_batch)
            batch_pes.append(np.array([np.sum(pe) for pe in pes]))
            batch_qpes.append(np.array([np.sum(qpe) for qpe in qpes]))

        prediction_errors.append(np.array(batch_pes))
        amortised_prediction_errors.append(np.array(batch_qpes))

        if n % 10 == 0:
            tot_acc = 0
            q_acc = 0
            for (img_batch, label_batch) in zip(imglist, labellist):
              pred_imgs, pred_labels = self.test(img_batch,label_batch)
              tot_acc += self.accuracy(pred_labels, label_batch)
              pred_qlabels = self.amortised_test(img_batch)
              q_acc += self.accuracy(pred_qlabels, label_batch)
            print("Accuracy: ", tot_acc/len(imglist))
            print("Amortised Accuracy: ", q_acc / len(imglist))
            variational_accs.append(tot_acc/len(imglist))
            amortised_accs.append(q_acc / len(imglist))
            print("TEST ACCURACIES")
            tot_acc = 0
            q_acc = 0
            for (test_img_batch, test_label_batch) in zip(test_img_list, test_label_list):
                pred_imgs, pred_labels = self.test(test_img_batch, test_label_batch)
                tot_acc += accuracy(pred_labels, test_label_batch)
                pred_qlabels = self.amortised_test(test_img_batch)
                q_acc += accuracy(pred_qlabels, test_label_batch)
            print("Test Variational Accuracy: ", tot_acc/len(test_img_list))
            print("Test Amortised Accuracy: ", q_acc / len(test_img_list))
            test_variational_accs.append(tot_acc/len(test_img_list))
            test_amortised_accs.append(q_acc / len(test_img_list))
            if save_name is not None:
                np.save(save_name + "_variational_acc.npy", np.array(deepcopy(variational_accs)))
                np.save(save_name + "_amortised_acc.npy", np.array(deepcopy(amortised_accs)))
                np.save(save_name + "_test_variational_acc.npy", np.array(deepcopy(test_variational_accs)))
                np.save(save_name+ "_test_amortised_acc.npy", np.array(deepcopy(test_amortised_accs)))
                #save the weights:
                for (i,(layer, qlayer)) in enumerate(zip(self.layers, self.q_layers)):
                    np.save(save_name + "_layer_"+str(i)+"_weights.npy",layer.weights)
                    np.save(save_name + "_layer_"+str(i)+"_amortisation_weights.npy",qlayer.weights)

            if False == True:
                amortised_generated_imgs = self.generate_amortised(imglist[0])
                print("Image mean: ", np.mean(np.abs(amortised_generated_imgs)))
                print("Amortised Generation: ")
                for i in range(5):
                    plt.imshow(amortised_generated_imgs[:,i].reshape(28,28))
                    plt.show()
                variational_generated_imgs = self.generate_variational(labellist[0])
                print("variational image mean: ", np.mean(np.abs(variational_generated_imgs)))
                print("Variational Generated Imgs: ")
                for i in range(5):
                    plt.imshow(variational_generated_imgs[:,i].reshape(28,28))
                    plt.show()


    prediction_errors = np.array(prediction_errors)
    amortised_prediction_errors = np.array(amortised_prediction_errors)
    prediction_errors = torch.mean(torch.from_numpy(prediction_errors),dim=1).numpy()
    amortised_prediction_errors = torch.mean(torch.from_numpy(amortised_prediction_errors),dim=1).numpy()
    if save_name is None:
        np.save(save_name + "_variational_acc.npy", np.array(variational_accs))
        np.save(save_name + "_amortised_acc.npy", np.array(amortised_accs))
        np.save(save_name + "_test_variational_acc.npy", np.array(test_variational_accs))
        np.save(save_name+ "_test_amortised_acc.npy", np.array(test_amortised_accs))
        #save the weights:
        for (i,(layer, qlayer)) in enumerate(zip(self.layers, self.q_layers)):
            np.save(save_name + "_layer_"+str(i)+"_weights.npy",layer.weights)
            np.save(save_name + "_layer_"+str(i)+"_amortisation_weights.npy",qlayer.weights)
    print("Accuracy plots")
    plt.title("Variational Accuracy")
    plt.xlabel("Epoch number")
    plt.ylabel("Proportion correct")
    plt.plot(variational_accs)
    plt.show()
    plt.title("Amortised Accuracy")
    plt.xlabel("Epoch number")
    plt.ylabel("Proportion correct")
    plt.plot(amortised_accs)
    plt.show()
    print("Test Accuracy plots")
    plt.title("Variational Accuracy")
    plt.xlabel("Epoch number")
    plt.ylabel("Proportion correct")
    plt.plot(test_variational_accs)
    plt.show()
    plt.title("Amortised Accuracy")
    plt.xlabel("Epoch number")
    plt.ylabel("Proportion correct")
    plt.plot(test_amortised_accs)
    plt.show()

def run_amortised(save_name):
    batch_size = 10
    num_batches = 10
    num_test_batches = 20
    n_inference_steps_train = 100
    n_inference_steps_test = 1000
    learning_rate = 0.01
    amortised_learning_rate = 0.001
    layer_sizes = [784, 300, 100, 10]
    n_layers = len(layer_sizes)
    n_epochs = 101
    inference_thresh = 0.5
    weight_decay = 0.00005

    train_set = torchvision.datasets.MNIST("MNIST_train", download=True, train=True)
    test_set = torchvision.datasets.MNIST("MNIST_test", download=True, train=False)
    #num_batches = len(train_set)// batch_size
    print("Num Batches",num_batches)
    img_list = [
        np.array(
            [
                np.array(train_set[(n * batch_size) + i][0]).reshape([784, 1]) / 255.0
                for i in range(batch_size)
            ]
        ).T.reshape([784, batch_size])
        for n in range(num_batches)
    ]
    label_list = [
        np.array([onehot(train_set[(n * batch_size) + i][1]) for i in range(batch_size)]).T
        for n in range(num_batches)
    ]
    test_img_list = [
        np.array(
            [
                np.array(test_set[(n * batch_size) + i][0]).reshape([784, 1]) / 255.0
                for i in range(batch_size)
            ]
        ).T.reshape([784, batch_size])
        for n in range(num_test_batches)
    ]
    test_label_list = [
        np.array([onehot(test_set[(n * batch_size) + i][1]) for i in range(batch_size)]).T
        for n in range(num_test_batches)
    ]
    pred_net = AmortisedPredictiveCodingNetwork(
        layer_sizes,
        batch_size,
        learning_rate,
        amortised_learning_rate,
        n_inference_steps_train=n_inference_steps_train,
        n_inference_steps_test=n_inference_steps_test,
        f=tanh,
        df=tanhderiv,
        qf=tanh,
        dqf=tanhderiv,
        inference_threshold=inference_thresh,
        wd_lambda = weight_decay,
    )
    pred_net.train(img_list, label_list,test_img_list, test_label_list, n_epochs,save_name)

if __name__ == "__main__":
    sname = str(sys.argv[1]) if len(sys.argv) >1 else None
    run_amortised(sname)
