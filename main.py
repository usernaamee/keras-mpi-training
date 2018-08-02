import numpy as np
from mpi4py import MPI
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[:2000]
y_train = y_train[:2000]
x_test = x_test[:2000]
y_test = y_test[:2000]
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

model = Sequential()
model.add(Reshape((28, 28, 1), input_shape=(28, 28)))
model.add(Conv2D(32, 3, activation='relu'))
model.add(Conv2D(32, 3, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(128, 3, activation='relu'))
model.add(Conv2D(128, 3, activation='relu'))
model.add(Conv2D(128, 3, activation='relu'))
model.add(Conv2D(128, 3, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
sgd = SGD(lr=1e-4)
model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
wts = model.get_weights()
comm.bcast(wts, root=0)
model.set_weights(wts)

n_procs = size
n_training_samples = x_train.shape[0]
n_testing_samples = x_test.shape[0]
training_samples_per_proc = n_training_samples // n_procs 
testing_samples_per_proc = n_testing_samples // n_procs 
minibatch_size = 100
n_minibatches = training_samples_per_proc // minibatch_size

for epoch in range(2):
    for mb_idx in range(n_minibatches):
        X = x_train[training_samples_per_proc*rank:training_samples_per_proc*(rank+1)]
        Y = y_train[training_samples_per_proc*rank:training_samples_per_proc*(rank+1)]
        wts_before = model.get_weights()
        model.fit(X, Y, epochs=1, verbose=0)
        wts_after = model.get_weights()
        local_grads = [(wb-wa) for wb, wa in zip(wts_before, wts_after)]
        global_grads = []
        new_wts = []
        for lg, wb in zip(local_grads, wts_before):
            gg = np.empty_like(lg)
            comm.Allreduce(lg, gg, op=MPI.SUM)
            gg /= n_procs
            global_grads.append(gg)
            wt = wb - gg
            new_wts.append(wt)
        model.set_weights(new_wts)
        #if rank==0:
        #    print('l1 norm wts (sanity check): before={:.3f}, after={:.3f}'.format(np.sum(np.abs(wts_before[0])), np.sum(np.abs(new_wts[0]))))
        # eval performance
        Xtest = x_test[testing_samples_per_proc*rank:testing_samples_per_proc*(rank+1)]
        Ytest = y_test[testing_samples_per_proc*rank:testing_samples_per_proc*(rank+1)]
        local_loss, local_acc = model.evaluate(Xtest, Ytest, verbose=0)
        global_loss = 0
        global_acc = 0
        locals_ = np.array([local_loss, local_acc])
        globals_ = np.zeros(2)
        comm.Allreduce(locals_, globals_, op=MPI.SUM)
        global_loss = globals_[0]/n_procs
        global_acc = globals_[1]/n_procs
        if rank==0:
            print('Epoch: {}  MinibatchIndex: {}/{}    loss:{:.3f}   acc:{:.2f}'.format(epoch, mb_idx, n_minibatches, global_loss, global_acc))
