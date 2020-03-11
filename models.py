import numpy as np


class PredictiveCodingLayer(object):
  def __init__(self,input_size, output_size, batch_size, fn,fn_deriv,learning_rate):
    self.input_size = input_size
    self.output_size = output_size
    self.batch_size = batch_size
    self.fn = fn
    self.fn_deriv = fn_deriv
    self.learning_rate = learning_rate
    self.weights = np.random.normal(0,0.1,[output_size, input_size])
    self.mu = np.random.normal(0,1,[input_size,batch_size])

  def step(self,pe_below,pred):
    #print("pe below: ",pe_below.shape, "pred ", pred.shape, "mu ", self.mu.shape, "weights ",self.weights.shape)
    pe = self.mu - pred
    self.mu+= self.learning_rate * (-pe + (np.dot(self.weights.T, pe_below *self.fn_deriv(np.dot(self.weights, self.mu)))))
    return pe, self.predict()

  def predict(self):
    return self.fn(np.dot(self.weights, self.mu))

  def update_weights(self,pe_below):
    self.weights += self.learning_rate * (np.dot(pe_below * self.fn_deriv(np.dot(self.weights, self.mu)), self.mu.T))

class PredictiveCodingNetwork(object):
  def __init__(self, layer_sizes, learning_rate,batch_size,n_inference_steps,fn,fn_deriv):
    #layer size includes input and output
    self.layer_sizes = layer_sizes[::-1]
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.fn = fn
    self.fn_deriv = fn_deriv
    self.layers = []
    self.n_inference_steps = n_inference_steps
    for i in range(1,len(self.layer_sizes)):
      layer = PredictiveCodingLayer(self.layer_sizes[i-1],self.layer_sizes[i],self.batch_size, self.fn, self.fn_deriv, self.learning_rate)
      self.layers.append(layer)

    self.predictions = [[] for i in range(len(self.layers)+1)]
    self.prediction_errors = [[] for i in range(len(self.layers)+1)]
    self.predictions[0] = np.zeros([self.layer_sizes[0],batch_size])

  def infer(self, inp, label):
    self.layers[0].mu = label
    #initialize predictions
    for i,layer in enumerate(self.layers):
      self.predictions[i+1] = layer.predict()
      self.prediction_errors[i] = layer.mu - self.predictions[i]
    #setup bottom prediction error
    self.prediction_errors[-1]= inp - self.predictions[-1]
    #run inference step
    for n in range(self.n_inference_steps):
      for i,layer in enumerate(self.layers):
        pe,pred = layer.step(self.prediction_errors[i+1],self.predictions[i])
        self.predictions[i+1] = pred
        self.prediction_errors[i] = pe
      self.layers[0].mu = label #fix output to correct label
    #update weights
    for i,layer in enumerate(self.layers):
      layer.update_weights(self.prediction_errors[i+1])

    return self.prediction_errors

  def train(self, imglist, labellist,n_epochs):
    prediction_errors = []
    for i in range(n_epochs):
      print("Epoch: ",i)
      for (img_batch, label_batch) in zip(imglist, labellist):
        pes = self.infer(img_batch, label_batch)
        prediction_errors.append([np.sum(pe) for pe in pes])

      if i % 10 ==0:
        pred_imgs, pred_labels = self.test(imglist[0])
        accuracy = self.accuracy(pred_labels, labellist[0])
        print("Accuracy: ", accuracy)

      if i % n_epochs-1 == 0 and i !=1:
        self.plot_batch_results(pred_labels,labellist[0])

    return prediction_errors

  def test(self,img_batch):
    #initialize predictions
    for i,layer in enumerate(self.layers):
      self.predictions[i+1] = layer.predict()
    #setup bottom prediction error
    self.prediction_errors[-1]= inp - self.predictions[-1]
    #run inference step
    for n in range(self.n_inference_steps):
      for i,layer in enumerate(self.layers):
        pe,pred = layer.step(self.prediction_errors[i+1],self.predictions[i])
        self.predictions[i+1] = pred
        self.prediction_errors[i] = pe
    pred_labels = self.layers[0].mu
    pred_imgs = self.predictions[-1]
    return pred_imgs, pred_labels

  def accuracy(self,pred_labels, true_labels):
    correct = 0
    batch_size = pred_labels.shape[1]
    for b in range(batch_size):
      if np.argmax(pred_labels[:,b]) == np.argmax(true_labels[:,b]):
        correct +=1
    return correct / batch_size

  def plot_batch_results(self,pred_labels,true_labels):
    batch_size = pred_labels.shape[1]
    for b in range(batch_size):
      plt.plot(pred_labels[:,b])
      plt.show()
      print(true_labels[:,b])

class AmortisationLayer(object):
  def __init__(self, input_size, output_size,learning_rate, batch_size, q_fn, q_fn_deriv):
    self.input_size = input_size
    self.output_size = output_size
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.q_fn = q_fn
    self.q_fn_deriv = q_fn_deriv

    self.weights = np.random.normal(0,0.1, [self.input_size, self.output_size])

  def predict(self,state):
    return self.q_fn(np.dot(self.weights, state))

  def update_weights(self,amortised_prediction_errors, state):
    self.weights +=self.learning_rate * (np.dot(amortised_prediction_errors * self.q_fn_deriv(np.dot(self.weights, state)), state.T))



class AmortisedPredictiveCodingNetwork(object):
  def __init__(self, layer_sizes, learning_rate,amortised_learning_rate, batch_size, n_inference_steps, fn,fn_deriv, q_fn, q_fn_deriv):
    self.layer_sizes = layer_sizes[::-1]
    self.learning_rate = learning_rate
    self.amortised_learning_rate = amortised_learning_rate
    self.batch_size = batch_size
    self.n_inference_steps = n_inference_steps
    self.fn = fn
    self.fn_deriv = fn_deriv
    self.q_fn = q_fn
    self.q_fn_deriv = q_fn_deriv
    self.layers = []
    self.amortised_layers = []
    self.n_inference_steps = n_inference_steps
    for i in range(1,len(self.layer_sizes)):
      layer = PredictiveCodingLayer(self.layer_sizes[i-1],self.layer_sizes[i],self.batch_size, self.fn, self.fn_deriv, self.learning_rate)
      self.layers.append(layer)
      amortised_layer = AmortisationLayer(self.layer_sizes[i-1],self.layer_sizes[i], self.amortised_learning_rate,self.batch_size, self.q_fn, self.q_fn_deriv)
      self.amortised_layers.append(amortised_layer)

    self.predictions = [[] for i in range(len(self.layers)+1)]
    self.prediction_errors = [[] for i in range(len(self.layers)+1)]
    self.predictions[0] = np.zeros([self.layer_sizes[0],batch_size])
    self.amortised_predictions = [[] for i in range(len(self.layers)+1)]
    self.amortised_prediction_errors = [[] for i in range(len(self.layers))]

  def infer(self, inp, label):
    self.layers[0].mu = label
    #initialize predictions
    for i,layer in enumerate(self.layers):
      self.predictions[i+1] = layer.predict()
      self.prediction_errors[i] = layer.mu - self.predictions[i]
    self.prediction_errors[-1]= inp - self.predictions[-1]
    self.amortised_predictions[-1] = inp
    #setup amortised predictions (which run in reverse)
    for i in reversed(range(len(self.amortised_layers))):
      self.amortised_predictions[i] = self.amortised_layers[i].predict(self.amortised_predictions[i+1])
      #setup the initial condition of the variational descent to be the amortised prediction
      self.layers[i].mu = deepcopy(self.amortised_predictions[i])
    #run inference step
    self.layers[0].mu = label
    for n in range(self.n_inference_steps):
      for i,layer in enumerate(self.layers):
        pe,pred = layer.step(self.prediction_errors[i+1],self.predictions[i])
        self.predictions[i+1] = pred
        self.prediction_errors[i] = pe
      self.layers[0].mu = label #fix output to correct label
    #update weights
    for i,layer in enumerate(self.layers):
      layer.update_weights(self.prediction_errors[i+1])

    #compute amortised prediction errors and update weights
    for i,(layer,qlayer) in enumerate(zip(self.layers, self.amortised_layers)):
      q_pe = layer.mu - self.amortised_predictions[i]
      self.amortised_prediction_errors[i] =q_pe
      #print("amortised weights: ",i, len(self.amortised_layers))
      if i == len(self.amortised_layers)-1: # if bottom layer then use input as input
        qlayer.update_weights(q_pe, inp)
      else:
        qlayer.update_weights(q_pe, self.layers[i+1].mu)

    return self.prediction_errors, self.amortised_prediction_errors

  def train(self, imglist, labellist, n_epochs):
    prediction_errors = []
    amortisation_prediction_errors = []
    for i in range(n_epochs):
      print("Epoch: ",i)
      for (img_batch, label_batch) in zip(imglist, labellist):
        pes,q_pes = self.infer(img_batch, label_batch)
        prediction_errors.append([np.sum(pe) for pe in pes])
        amortisation_prediction_errors.append([np.sum(q_pe) for q_pe in q_pes])

      if i % 10 ==0:
        pred_imgs, pred_labels = self.test(imglist[0])
        print("pred_labels: ", pred_labels)
        accuracy = self.accuracy(pred_labels, labellist[0])
        print("Accuracy: ", accuracy)
        q_labels = self.amortised_test(imglist[0])
        q_accuracy = self.accuracy(q_labels, labellist[0])
        print("Amortised Accuracy: ", q_accuracy)

      if i % n_epochs-1 == 0 and i !=1:
        self.plot_batch_results(pred_labels,labellist[0])

    return prediction_errors,amortisation_prediction_errors


  def test(self,img_batch):
    #setup amortised predictions (which run in reverse)
    self.amortised_predictions[-1] = img_batch
    for i in reversed(range(len(self.amortised_layers))):
      self.amortised_predictions[i] = self.amortised_layers[i].predict(self.amortised_predictions[i+1])
      #setup the initial condition of the variational descent to be the amortised prediction
      self.layers[i].mu = deepcopy(self.amortised_predictions[i])
    #initialize predictions
    for i,layer in enumerate(self.layers):
      self.predictions[i+1] = layer.predict()
    #setup bottom prediction error
    self.prediction_errors[-1]= inp - self.predictions[-1]
    #run inference step
    for n in range(self.n_inference_steps):
      for i,layer in enumerate(self.layers):
        pe,pred = layer.step(self.prediction_errors[i+1],self.predictions[i])
        self.predictions[i+1] = pred
        self.prediction_errors[i] = pe
    pred_labels = self.layers[0].mu
    pred_imgs = self.predictions[-1]
    return pred_imgs, pred_labels

  def amortised_test(self, img_batch):
    self.amortised_predictions[-1] = img_batch
    for i in reversed(range(len(self.amortised_layers))):
      self.amortised_predictions[i] = self.amortised_layers[i].predict(self.amortised_predictions[i+1])
    q_labels = self.amortised_predictions[0]
    return q_labels

  def accuracy(self,pred_labels, true_labels):
    correct = 0
    batch_size = pred_labels.shape[1]
    for b in range(batch_size):
      if np.argmax(pred_labels[:,b]) == np.argmax(true_labels[:,b]):
        correct +=1
    return correct / batch_size

  def plot_batch_results(self,pred_labels,true_labels):
    batch_size = pred_labels.shape[1]
    for b in range(batch_size):
      plt.plot(pred_labels[:,b])
      plt.show()
      print(true_labels[:,b])
