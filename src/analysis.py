import numpy as np
import tensorflow as tf

#For corrrelated mapping
def cov(X, Y):
    X=X-X.mean(axis=0, keepdims=True)
    X=X/X.std(axis=0, keepdims=True)
    Y=Y-Y.mean(axis=0, keepdims=True)
    Y=Y/Y.std(axis=0, keepdims=True)
    return np.dot(X.T, Y)/X.shape[0]

def analysis_correlation(model, mapping, test_dataset_with_labels):
    z_list=[]
    y_list=[]
    for batch_x, batch_y in test_dataset_with_labels:
        mean, _ = model.encode(batch_x)
        batch_z, info = mapping(mean)
        z_list.append(batch_z)
        y_list.append(batch_y)
    z=np.concatenate(z_list, axis=0)
    y=np.concatenate(y_list, axis=0)
    cov_mat=cov(y, z)
    return cov_mat

#Classifier - downstream task
class classifier(tf.keras.Model):
    def __init__(self, out_dim):
        super(classifier, self).__init__()
        self.d1=tf.keras.layers.Dense(10, activation='relu')
        self.d2=tf.keras.layers.Dense(out_dim, activation=None)
    def call(self, z):
        h=self.d1(z)
        #h=z
        y=self.d2(h)
        return y

def classifier_train_step(Z, Y, classifier, optimizer, regress, classifier_loss_record):
    with tf.GradientTape() as tape:
        with tf.device('/device:GPU:0'):
            logits_pred = classifier(Z)
            if regress:
                loss = tf.reduce_sum((Y.reshape([-1,1])-logits_pred)**2, axis=-1)
            else:
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits_pred)
    trainable_variables=classifier.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    classifier_loss_record(loss)

def classifier_eval_step(Z, Y, classifier, regress, classifier_accuracy_record):
    logits_pred = classifier(Z).numpy()
    
    if regress:
        accuracy=-np.mean(np.sum((logits_pred-Y.reshape([-1,1]))**2, axis=-1), axis=0)
        ##It actually -MSE
    else:
        pred=np.argmax(logits_pred, axis=1)
        accuracy = np.mean((pred-Y)==0)
    classifier_accuracy_record(accuracy)

def classifier_train(z_input_train, y_input_train, z_input_test, y_input_test, classifier, classifier_optimizer, regress, classifier_loss_record, classifier_accuracy_record, max_epoch=1000, batch_size=100):
    if regress:
        eval_old_accuracy=-1e6
    else:
        eval_old_accuracy=0.0
    flag=0
    for epoch in range(max_epoch):
        ##Train
        classifier_loss_record.reset_states()
        for i in range(int(np.ceil(z_input_train.shape[0]/batch_size))):
            classifier_train_step(z_input_train[i*batch_size:(i+1)*batch_size], y_input_train[i*batch_size:(i+1)*batch_size], classifier, classifier_optimizer, regress,  classifier_loss_record)
        train_loss=classifier_loss_record.result()
        
        ##Eval
        classifier_accuracy_record.reset_states()
        for i in range(int(np.ceil(z_input_test.shape[0]/batch_size))):
            classifier_eval_step(z_input_test[i*batch_size:(i+1)*batch_size], y_input_test[i*batch_size:(i+1)*batch_size], classifier, regress, classifier_accuracy_record)
        eval_accuracy=classifier_accuracy_record.result()
        ##Print
        #print('Epoch:{}, train loss:{}, eval_accuracy:{}'.format(epoch, train_loss, eval_accuracy))
        #print(eval_accuracy, eval_old_accuracy)
        if eval_accuracy<=eval_old_accuracy:
            flag+=1
        else:
            flag=0
            eval_old_accuracy=eval_accuracy
        if flag>=20:
            return eval_old_accuracy
    return eval_old_accuracy

def classifier_experiment(out_dim, z_input_train, y_input_train, z_input_test, y_input_test, regress, data_size_list=[50, 100, 500, 1000, 2000, 5000]):
    Classifier=classifier(out_dim)
    classifier_optimizer=tf.keras.optimizers.Adam()
    classifier_loss_record=tf.keras.metrics.Mean(name='classifier_loss')
    classifier_accuracy_record=tf.keras.metrics.Mean(name='classifier_accuracy')

    Classifier.save_weights('models/classifier/model.ckpt')
    
    z_input_train=z_input_train.astype(np.float32)
    z_input_test=z_input_test.astype(np.float32)
    if regress:
        y_input_train=y_input_train.astype(np.float32)
        y_input_test=y_input_test.astype(np.float32)
    else:
        y_input_train=y_input_train.astype(np.int32)
        y_input_test=y_input_test.astype(np.int32)
    
    eval_accuracy_list=[]
    for data_size in data_size_list:
        Classifier.load_weights('models/classifier/model.ckpt')
        eval_accuracy=classifier_train(z_input_train[:data_size], y_input_train[:data_size], z_input_test, y_input_test, Classifier, classifier_optimizer, regress, classifier_loss_record, classifier_accuracy_record)
        eval_accuracy_list.append(eval_accuracy.numpy())
        #print('Data size: {}, eval accuracy: {}'.format(data_size, eval_accuracy))
    return eval_accuracy_list

if __name__=='__main__':
    z_input=np.random.random([10000, 10]).astype(np.float32)
    label=np.random.random([10000]).astype(np.float32)
    classifier_experiment(1, z_input[:8000],label[:8000], z_input[8000:],label[8000:], True)
