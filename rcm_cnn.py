# Lab 10 MNIST and Dropout
import tensorflow as tf
import random
import scipy.io as scio
import numpy as np
from pylab import *



PAPR_DB = arange(1,8,0.2)
PAPR = 10**(np.divide(PAPR_DB,10))

L = 4
learning_rate = 0.001
batch_size = 100
Nsubc = 32
test_batch_size = 300000
modulation_level = 4

mapping =np.array([1+1j,1-1j,-1+1j,-1-1j]/sqrt(2))

train_x_rand = np.random.randint(modulation_level, size=(300000, Nsubc))
train_x = mapping[train_x_rand]
train_x_r = real(train_x)
train_x_i = imag(train_x)
train_x_f = np.hstack((train_x_r,train_x_i))

X = tf.placeholder(tf.float32, [None, Nsubc*2])

layers = Nsubc*16

W1 = tf.get_variable("W1", shape=[2*Nsubc, layers], initializer=tf.contrib.layers.xavier_initializer())  
b1 = tf.Variable(tf.random_normal([layers]))

L1 = tf.nn.tanh(tf.matmul(X, W1) + b1)
L1 = tf.layers.batch_normalization(L1)
# L1 = tf.layers.batch_normalization(tf.matmul(X, W1) + b1)
# L1 = tf.nn.tanh(L1)


W2 = tf.get_variable("W2", shape=[layers, layers], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([layers]))
L2 = tf.nn.tanh(tf.matmul(L1, W2) + b2)
L2 = tf.layers.batch_normalization(L2)
# L2 = tf.layers.batch_normalization(tf.matmul(L1, W2) + b2)
# L2 = tf.nn.tanh(L2)

# W3 = tf.get_variable("W3", shape=[layers, layers], initializer=tf.contrib.layers.xavier_initializer())
# b3 = tf.Variable(tf.random_normal([layers]))
# L3 = tf.nn.tanh(tf.matmul(L2, W3) + b3)
# L3 = tf.layers.batch_normalization(L3)

# W4 = tf.get_variable("W4", shape=[layers, layers], initializer=tf.contrib.layers.xavier_initializer())
# b4 = tf.Variable(tf.random_normal([layers]))
# L4 = tf.nn.tanh(tf.matmul(L3, W4) + b4)
# L4 = tf.layers.batch_normalization(L4)

# W6 = tf.get_variable("W6", shape=[layers, layers], initializer=tf.contrib.layers.xavier_initializer())  
# b6 = tf.Variable(tf.random_normal([layers]))
# L6= tf.nn.tanh(tf.matmul(L4, W6) + b6)
# L6 = tf.layers.batch_normalization(L6)

# W7 = tf.get_variable("W7", shape=[layers, layers], initializer=tf.contrib.layers.xavier_initializer())
# b7 = tf.Variable(tf.random_normal([layers]))
# L7 = tf.nn.tanh(tf.matmul(L6, W7) + b7)
# L7 = tf.layers.batch_normalization(L7)

# W8 = tf.get_variable("W8", shape=[layers, layers], initializer=tf.contrib.layers.xavier_initializer())
# b8 = tf.Variable(tf.random_normal([layers]))
# L8 = tf.nn.tanh(tf.matmul(L7, W8) + b8)
# L8 = tf.layers.batch_normalization(L8)

# W9 = tf.get_variable("W9", shape=[layers, layers], initializer=tf.contrib.layers.xavier_initializer())
# b9 = tf.Variable(tf.random_normal([layers]))
# L9 = tf.nn.tanh(tf.matmul(L8, W9) + b9)
# L9 = tf.layers.batch_normalization(L9)

W5 = tf.get_variable("W5", shape=[layers, Nsubc*2], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([Nsubc*2]))

hypothesis = tf.matmul(L2, W5) + b5

X_real = hypothesis[:,0:Nsubc]
X_imag = hypothesis[:,Nsubc:2*Nsubc]
X_symbol = tf.complex(X_real,X_imag)
# X_symbol = tf.transpose(X_symbol)

# X_symbol = np.vstack((X_symbol,np.zeros((batch_size,Nsubc*(L-1)))))

X_symbol_ifft = tf.ifft(X_symbol)

encoded_symbol_mean = tf.reduce_mean(tf.abs(X_symbol_ifft)**2,axis=1)
RCM = tf.sqrt(tf.reduce_mean((tf.abs(X_symbol_ifft)**6),axis=1)/encoded_symbol_mean**3)
# cost = 0.01*tf.reduce_mean(tf.abs(peak_power_symbol)) + tf.reduce_sum(tf.square(tf.abs(hypothesis-X)))/tf.reduce_sum(tf.square(tf.abs(X))) + tf.contrib.layers.l2_regularizer(.5)(W1)
cost = 0.1*tf.reduce_mean(tf.abs(RCM)) + tf.reduce_sum(tf.square(tf.abs(hypothesis-X)))/tf.reduce_sum(tf.square(tf.abs(X))) 
# cost = 0.01*tf.reduce_mean(tf.abs(peak_power_symbol)) + tf.reduce_sum(tf.square(tf.abs(hypothesis-X))) 
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


cost_plot =[]
for epoch in range(int(test_batch_size/batch_size)):
    avg_cost = 0
    batch_x = train_x_f[epoch*batch_size:(epoch+1)*batch_size,:]
         
    feed_dict = {X: batch_x}
    c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
    avg_cost += c
    if epoch % 100 ==0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    cost_plot.append(avg_cost)


print('Learning Finished!')



te_x = np.random.randint(modulation_level, size=(10000, Nsubc))
te_x_f = mapping[te_x]
te_x_r = real(te_x_f)
te_x_i = imag(te_x_f)
te_x_te = np.hstack((te_x_r,te_x_i))

te_x_ifft = ifft(te_x_f,Nsubc)*sqrt(Nsubc)
# te_x_ifft = ifft(te_x_f,Nsubc)*sqrt(Nsubc)

BER = []
BER1 = []

SNR = np.arange(0,14,1)
for SNR_range in SNR:
    sigma = sqrt(mean(abs(te_x_ifft)**2)/(10**(SNR_range/10)))
    noise = sigma/sqrt(2)*(np.random.normal(0,size=(10000,Nsubc))+1j*(np.random.normal(0,size=(10000,Nsubc))))
    recive = te_x_ifft + noise
    Y = fft(recive,Nsubc)/sqrt(Nsubc)
    errbit = 0
    for i in range(10000):
        for j in range(Nsubc):
            index = np.where(abs(Y[i,j]-mapping) == min(abs(Y[i,j]-mapping)))[0][0]
            if index == te_x[i,j]:
                errbit = errbit
            # elif abs(index-te_x[i,j]) == 3 or (index ==1 and te_x[i,j]==2) or (index ==2 and te_x[i,j]==1):
            #     errbit = errbit + 2
            elif abs(index-te_x[i,j]) == 2: 
                errbit = errbit + 2
            else:
                errbit = errbit + 1
    BER.append(errbit / (10000*Nsubc*log2(modulation_level)))


# X_symbol = sess.run(X_symbol,feed_dict={X: te_x_te})
X_symbol_out = sess.run(X_symbol_ifft,feed_dict={X: te_x_te}) * sqrt(Nsubc)
# X_symbol_ifft = ifft(X_symbol,Nsubc*L)*sqrt(Nsubc)*L
# X_symbol_out = X_symbol_ifft

for SNR_range in SNR:
    sigma = sqrt(mean(abs(X_symbol_out)**2)/(10**(SNR_range/10)))
    noise = sigma/sqrt(2)*(np.random.normal(0,size=(10000,Nsubc))+1j*(np.random.normal(0,size=(10000,Nsubc))))
    recive = X_symbol_out + noise
    Y = fft(recive,Nsubc)/sqrt(Nsubc)
    errbit = 0
    for i in range(10000):
        for j in range(Nsubc):
            index = np.where(abs(Y[i,j]-mapping) == min(abs(Y[i,j]-mapping)))[0][0]
            if index == te_x[i,j]:
                errbit = errbit
            # elif abs(index-te_x[i,j]) == 3 or (index ==1 and te_x[i,j]==2) or (index ==2 and te_x[i,j]==1):
            #     errbit = errbit + 2
            elif abs(index-te_x[i,j]) == 2:
                errbit = errbit + 2
            else:
                errbit = errbit + 1
    BER1.append(errbit / (10000*Nsubc*log2(modulation_level)))


PAPR_sample = []

ccdf1 = []
for j in range(len(PAPR_DB)):
    ccdf1.append(np.sum(PAPR[j]<np.sqrt(np.mean(np.abs(te_x_ifft)**6,axis=1)/(np.mean(np.abs(te_x_ifft)**2,axis=1)**3)))/10000)




# for j in range(len(PAPR_DB)):
#     ccdf1.append(np.sum(PAPR[j]<(np.max(np.abs(train_x_ifft)**2,axis=1) /(np.mean(np.abs(train_x_ifft)**2,axis=1))))/(test_batch_size))

# print(sess.run(peak_power_symbol, feed_dict={X: te_x_te}))
RCM = sess.run(RCM, feed_dict={X: te_x_te})
# encoded_symbol_mean = tf.reduce_mean(tf.abs(X_symbol_ifft)**2,axis=1)
# encoded_symbol_original_max = tf.reduce_max(tf.abs(X_symbol_ifft)**2,axis=1)
# peak_power_symbol = tf.div(encoded_symbol_original_max, encoded_symbol_mean)
# PAPR_sample = peak_power_symbol

CCDF=[]
for j in range(len(PAPR_DB)):
    CCDF.append(np.sum(PAPR[j]<(RCM))/(10000))
print(BER)
print(BER1)

print(ccdf1)
print(CCDF)


figure(1)

semilogy(PAPR_DB, ccdf1 ,color='black', ls='dashed', markersize=7,linewidth=1,label="Original OFDM")
semilogy(PAPR_DB, CCDF ,color='r', marker="o", markersize=7,linewidth=1,label="dnn")
legend()
xlabel('PAPR0 [dB]', fontsize=14)
ylabel('CCDF (PAPR<PAPR0)', fontsize=14)
ylim(ymin=10**-4)
tick_params(labelsize=10)
grid(which='minor')
grid(which='major')

figure(2)
semilogy(SNR, BER ,color='black', ls='dashed', markersize=7,linewidth=1,label="Original")
semilogy(SNR, BER1 ,color='r', marker="o", markersize=7,linewidth=1,label="dnn")
xlabel('SNR[dB]', fontsize=14)
ylabel('BER', fontsize=14)
ylim(ymin=10**-4)
show()
