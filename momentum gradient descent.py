
#_冲量梯度下降原理展示

import numpy as np
import os
import matplotlib.pyplot as plt

curdir = os.path.split(os.path.realpath(__file__))[0]

def costfun(x):
    return x**2+1;

def costfun_derivative(x):
    return 2*x;

def sgd_momentum(m,eta,epoch):
    x = -3  #np.random.randn(1)*5;x=x[0];
    v = 0;  #起始速度为0
    xx = [x];
    yy = [costfun(x)];
    for i in range(epoch):
        v = - eta*costfun_derivative(x) + m*v;  #_把前一步的速度的一部分加进来，m一般为(0,0.5)比较好，m>=1，速度累积量越来越大，发散，m=0，等同于普通的梯度下降。
        x = x + v;                              #_冲量好处是当速度方向和步行方向一致时，会加快步伐。在驻点附近时，也会逐渐减速至停止。（m如果大了，由梯度觉得的趋势会被上一次的速度覆盖，两者反向时，就有问题了。）
        xx.append(x);                           #_当步行方向和梯度方向不一致时，冲量会用上一次的速度减缓本次下降，使驻点震荡幅度减小
        yy.append(costfun(x))
    return (xx,yy)


#print(training_data[0][0].reshape(28,28))
#plt.imshow(training_data[0][2].reshape(28,28),cmap='binary')
plt.figure();
ax = plt.axes()
line_x = np.linspace(-5,5,100);
line_y = costfun(line_x);
plt.plot(line_x,line_y)
(xx,yy) = sgd_momentum(0.5,0.05,20)
print(xx,yy)
for i in range(len(xx)-1):
    plt.plot(xx[i],0,'*r')
    ax.arrow(xx[i],yy[i],xx[i+1]-xx[i],yy[i+1]-yy[i], head_width=0.2, head_length=0.2)
plt.show()

