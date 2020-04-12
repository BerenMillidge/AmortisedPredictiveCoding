import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import torch
from functions import *
from transformer import *
import sys
import torchvision

def create_posdef_matrix(N):
    A = np.abs(np.random.normal(0,0.05,[N,N]))
    return 0.5 * (np.dot(A.T, A)) + (0.1 * np.identity(N))

class PredictiveCodingLayer(object):
  def __init__(self,input_size, output_size, batch_size, fn,fn_deriv,learning_rate):
    self.input_size = input_size
    self.output_size = output_size
    self.batch_size = batch_size
    self.fn = fn
    self.fn_deriv = fn_deriv
    self.learning_rate = learning_rate
    self.weights = np.random.normal(0,0.1,[input_size, output_size])
    print(input_size, batch_size)
    self.mu = np.random.normal(0,1,[output_size,batch_size])
    self.precisions = create_posdef_matrix(self.output_size)
    #self.precisions = np.identity(self.output_size)

  def update_mu(self, pe, pe_below):
    self.mu+= self.learning_rate * (-pe + (np.dot(self.weights.T, pe_below *self.fn_deriv(np.dot(self.weights, self.mu)))))

  def step(self,pe_below,pred,use_top_down_pe=True):
    if use_top_down_pe:
      pe = self.mu - pred
    else:
      pe = np.zeros_like(self.mu)
    self.mu+= self.learning_rate * (-pe + (np.dot(self.weights.T, pe_below *self.fn_deriv(np.dot(self.weights, self.mu)))))
    return pe, self.predict()


  def predict(self):
    return self.fn(np.dot(self.weights, self.mu))

  def update_weights(self,pe_below):
    self.weights += self.learning_rate * (np.dot(pe_below * self.fn_deriv(np.dot(self.weights, self.mu)), self.mu.T))

  def update_precisions(self, pe):
    #pi_delta = -(np.dot(np.dot(self.precisions.T, pe),np.dot(pe.T, self.precisions.T))) - self.precisions
    # so this suffices to destroy the performance entirely... BUT WHY!?
    #surely it shouldn't do anything? i.e. if the variance of the precisions really is independent
    #there is NO reason WHY it should destroy it
    # so we have the interesting additional question of why it's somehow decreasing F hugely
    # while accuracy is maintained to be really poor.... how is this possible?
    # okay, so precisions are now WORKING!!! well this is interesting. That was actually quite straightforward somehow!
    # well that's an advance in and of itslef. Let's try ot continue it!
    # the key to success is it starting off from an identity matrix andthen hardly advancing, but HOW much of a difference does this make...
    #I mean in theory we EXPECT the precision of a datapoint to be high for that datapoint right?
    # what even is the precision. As a covariance matrix the precision matrix tells us HOW related similar items are...
    #so OF COURSE! it should be similar to itself. I don't think that's unreasonable at all!
    # but anyhow, I think this is actually a pretty good result in and of itself that I should keep in mind.
    # key behaviours / goals will be to write this up. Get correct occlusion results sorted.
    # and look at the biologically plausible experiments for stuff I've done over the weekend.
    # in fact
    #prec_pe = np.dot(self.precisions, pe)
    pi_delta = -np.dot(pe, pe.T) + self.precisions
    self.precisions += 0.1 * self.learning_rate * pi_delta
    #print("Symmetry Check: ", np.sum(self.precisions - self.precisions.T))
    print("Divergence from identity:", np.mean(self.precisions - np.identity(self.precisions.shape[0])))
    self.precisions = np.clip(self.precisions,a_min=1e-5,a_max=1e5)

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
        self.prediction_errors[l+1] = np.dot((self.layers[l].mu - self.predictions[l+1]).T, self.layers[l].precisions)
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
  def __init__(self, layer_sizes,batch_size,learning_rate,amortised_learning_rate,n_inference_steps_train,n_inference_steps_test, f,df,qf,dqf,inference_threshold=0.1):
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
    for i in range(len(self.layer_sizes)-1):
      self.layers.append(PredictiveCodingLayer(self.layer_sizes[i], self.layer_sizes[i+1], self.batch_size, self.f,self.df,self.learning_rate))
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



  def forward_pass(self, img_batch, label_batch, test=False,save_accuracy_steps = False):
    # reset model
    n_inference_steps = self.n_inference_steps_test if test else self.n_inference_steps_train
    self.reset_mus()
    prediction_errors = [[] for i in range(self.n_layers)]

    #set the highest level mus to the label if training. Else let them vary freely (they will become the prediction).
    #run an amortisation pass to initialize all variational parameters.
    if save_accuracy_steps:
        accuracies = []
    self.amortisation_pass(img_batch,initialize=True)
    if save_accuracy_steps:
        accuracies.append(accuracy(self.amortised_predictions[-1],label_batch))

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
            self.prediction_errors[i+1] = np.dot(self.layers[i].precisions, self.layers[i].mu - self.predictions[i+1])
            if i != self.L-1:
                self.layers[i].update_mu(self.prediction_errors[i+1], self.prediction_errors[i])
            else:
                self.layers[i].update_mu(np.zeros_like(self.layers[i].mu), self.prediction_errors[i])
            self.predictions[i] = self.layers[i].predict()
            #set reset the final layer mus to the batch label
        if not test:
            self.layers[-1].mu = deepcopy(label_batch)
            self.predictions[-1] = deepcopy(self.layers[-1].mu)
        if save_accuracy_steps:
            accuracies.append(accuracy(self.layers[-1].mu,label_batch))

        if not save_accuracy_steps:
            F = np.sum(np.array([np.mean(np.square(pe)) for pe in self.prediction_errors]))
            if F <= self.inference_threshold:
                break

    pred_imgs = deepcopy(self.predictions[0])
    pred_labels = deepcopy(self.layers[-1].mu)
    if save_accuracy_steps:
        return pred_imgs, pred_labels, np.array(accuracies)
    else:
        return pred_imgs, pred_labels

  def train_batch(self, img_batch, label_batch):
    self.forward_pass(img_batch, label_batch, test=False)
    for l in range(self.L):
        self.layers[l].update_weights(self.prediction_errors[l])
        self.layers[l].update_precisions(self.prediction_errors[l+1])
        self.amortised_prediction_errors[l] = self.layers[l].mu - self.amortised_predictions[l+1]
        #if l == 0:
        #    self.q_layers[l].update_weights(self.amortised_prediction_errors[l], self.amortised_predictions[0])
        #else:
        #    self.q_layers[l].update_weights(self.amortised_prediction_errors[l], self.layers[l-1].mu)

    return self.prediction_errors, self.predictions,self.amortised_predictions, self.amortised_prediction_errors


  def test(self, img_batch, label_batch,save_accuracy_steps=False):
    return self.forward_pass(img_batch, label_batch,test=True,save_accuracy_steps=save_accuracy_steps)

  def amortised_test(self, img_batch):
    self.amortisation_pass(img_batch,initialize=True)
    return self.amortised_predictions[-1]

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
            print(f"Test Variational Accuracy: {tot_acc/len(test_img_list)}")
            print(f"Test Amortised Accuracy: {q_acc / len(test_img_list)}")
            test_variational_accs.append(tot_acc/len(test_img_list))
            test_amortised_accs.append(q_acc / len(test_img_list))
            np.save(save_name + "_variational_acc.npy", np.array(deepcopy(variational_accs)))
            np.save(save_name + "_amortised_acc.npy", np.array(deepcopy(amortised_accs)))
            np.save(save_name + "_test_variational_acc.npy", np.array(deepcopy(test_variational_accs)))
            np.save(save_name+ "_test_amortised_acc.npy", np.array(deepcopy(test_amortised_accs)))
            for l in range(len(self.layers)):
                print("Layer: " +str(l) + " precisions")
                plt.imshow(self.layers[l].precisions)
                plt.show()
            #save the weights:
            #for (i,(layer, qlayer)) in enumerate(zip(self.layers, self.q_layers)):
            #    np.save(save_name + "_layer_"+str(i)+"_weights.npy",layer.weights)
            #    np.save(save_name + "_layer_"+str(i)+"_amortisation_weights.npy",qlayer.weights)


    prediction_errors = np.array(prediction_errors)
    amortised_prediction_errors = np.array(amortised_prediction_errors)
    prediction_errors = torch.mean(torch.from_numpy(prediction_errors),dim=1).numpy()
    amortised_prediction_errors = torch.mean(torch.from_numpy(amortised_prediction_errors),dim=1).numpy()
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
    ## now test occlusion
    print("OCCLUSION TEST ACCURACIES")
    tot_acc = 0
    q_acc = 0
    accuracy_list = []
    for (test_img_batch, test_label_batch) in zip(test_img_list, test_label_list):
        occluded_batch = apply_batch(test_img_batch,occlude,(5,5),5,0)
        pred_imgs, pred_labels,accuracies = self.test(occluded_batch, test_label_batch,save_accuracy_steps=True)
        tot_acc += accuracy(pred_labels, test_label_batch)
        pred_qlabels = self.amortised_test(occluded_batch)
        q_acc += accuracy(pred_qlabels, test_label_batch)
        accuracy_list.append(accuracies)
        print(accuracies.shape)
    print(f"Occluded Variational Accuracy: {tot_acc/len(test_img_list)}")
    print(f"Occluded Amortised Accuracy: {q_acc / len(test_img_list)}")
    accuracy_list = np.mean(np.array(accuracy_list),axis=0)
    np.save(save_name+"_accuracy_list.npy", accuracy_list)
    plt.title("Occluded Accuracy through Variational Inference")
    plt.xlabel("Variational Step")
    plt.ylabel("Accuracy")
    plt.plot(accuracy_list)
    plt.show()
    print("Beginning occluder test")
    num_occluders = [1,3,5,7,9]
    n_occluder_accuracy_list = []
    num_occluder_accs = []
    q_num_occluder_accs = []
    for (i,n_occ) in enumerate(num_occluders):
        accuracy_list = []
        tot_acc = 0
        q_acc = 0
        for (test_img_batch, test_label_batch) in zip(test_img_list, test_label_list):
            occluded_batch = apply_batch(test_img_batch,occlude,(5,5),n_occ,0)
            pred_imgs, pred_labels,accuracies = self.test(occluded_batch, test_label_batch,save_accuracy_steps=True)
            tot_acc += accuracy(pred_labels, test_label_batch)
            pred_qlabels = self.amortised_test(occluded_batch)
            q_acc += accuracy(pred_qlabels, test_label_batch)
            accuracy_list.append(accuracies)
        n_occluder_accuracy_list.append(np.mean(np.array(accuracy_list),axis=0))
        num_occluder_accs.append(tot_acc / len(test_img_list))
        q_num_occluder_accs.append(q_acc / len(test_img_list))
    n_occluder_accuracy_list = np.array(n_occluder_accuracy_list)
    print("OCCLUDER ACCURACY LIST SIZE: ", n_occluder_accuracy_list.shape)
    np.save(save_name + "_n_occluder_acc_list.npy",np.array(n_occluder_accuracy_list))
    np.save(save_name + "_num_occluder_variational_acclist.npy",np.array(num_occluder_accs))
    np.save(save_name + "_num_occluder_amortised_acclist.npy",np.array(q_num_occluder_accs))

    # plot
    fig = plt.figure()
    plt.title("Average Variational Inference improvement for n occluders")
    plt.xlabel("Variational Step")
    plt.ylabel("Accuracy")
    for i,n_occ in enumerate(num_occluders):
        print("LIST: ", n_occluder_accuracy_list[i].shape)
        plt.plot(n_occluder_accuracy_list[i], label=str(n_occ) + " occluders")
    plt.legend()
    plt.show()

    occluder_sizes = [1,2,3,4,5,6,7]
    occluder_size_accuracy_list = []
    num_occluder_accs = []
    q_num_occluder_accs = []
    for (i,occ_size) in enumerate(occluder_sizes):
        accuracy_list = []
        tot_acc = 0
        q_acc = 0
        for (test_img_batch, test_label_batch) in zip(test_img_list, test_label_list):
            occluded_batch = apply_batch(test_img_batch,occlude,(occ_size,occ_size),5,0)
            pred_imgs, pred_labels,accuracies = self.test(occluded_batch, test_label_batch,save_accuracy_steps=True)
            tot_acc += accuracy(pred_labels, test_label_batch)
            pred_qlabels = self.amortised_test(occluded_batch)
            q_acc += accuracy(pred_qlabels, test_label_batch)
            accuracy_list.append(np.array(accuracies))
            print(accuracies.shape)
        occluder_size_accuracy_list.append(np.mean(np.array(accuracy_list),axis=0))
        num_occluder_accs.append(tot_acc / len(test_img_list))
        q_num_occluder_accs.append(q_acc / len(test_img_list))
    occluder_size_accuracy_list = np.array(occluder_size_accuracy_list)
    np.save(save_name + "_occluder_size_acc_list.npy",np.array(occluder_size_accuracy_list))
    np.save(save_name + "_occluder_size_variational_acclist.npy",np.array(num_occluder_accs))
    np.save(save_name + "_occluder_size_amortised_acclist.npy",np.array(q_num_occluder_accs))
    # plot
    fig = plt.figure()
    plt.title("Average Variational Inference improvement for occluder_size")
    plt.xlabel("Variational Step")
    plt.ylabel("Accuracy")
    for i,occ_size in enumerate(occluder_sizes):
        print("lists: " ,occluder_size_accuracy_list[i].shape)
        plt.plot(occluder_size_accuracy_list[i], label=str(occ_size) + " by " + str(occ_size))
    plt.legend()
    plt.show()





def run_amortised(save_name):
    batch_size = 10
    num_batches = 10
    num_test_batches = 10
    n_inference_steps_train = 100
    n_inference_steps_test = 1000
    learning_rate = 0.01
    amortised_learning_rate = 0.001
    layer_sizes = [784, 300, 100, 10]
    n_layers = len(layer_sizes)
    n_epochs = 101
    inference_thresh = 0.01

    train_set = torchvision.datasets.MNIST("MNIST_train", download=True, train=True)
    test_set = torchvision.datasets.MNIST("MNIST_test", download=True, train=False)
    #num_batches = len(train_set)// batch_size
    print("Num Batches",num_batches)
    img_list = [np.array([np.array(train_set[(n * batch_size) + i][0]).reshape([784, 1]) / 255.0 for i in range(batch_size)]).T.reshape([784, batch_size]) for n in range(num_batches)]
    label_list = [np.array([onehot(train_set[(n * batch_size) + i][1]) for i in range(batch_size)]).T for n in range(num_batches)]
    test_img_list = [np.array([np.array(test_set[(n * batch_size) + i][0]).reshape([784, 1]) / 255.0 for i in range(batch_size)]).T.reshape([784, batch_size]) for n in range(num_test_batches)]
    test_label_list = [np.array([onehot(test_set[(n * batch_size) + i][1]) for i in range(batch_size)]).T for n in range(num_test_batches)]

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
        inference_threshold=inference_thresh
    )
    pred_net.train(img_list, label_list,test_img_list, test_label_list, n_epochs,save_name)

if __name__ == "__main__":
    sname = str(sys.argv[1])
    run_amortised(sname)
