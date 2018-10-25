#### This is a simple GAN Zoo for studying

---

## Mostly using the Comics dataset

### 1. WGAN-GP
- After resizing from 96 to 64 then the images are not belong to 0-255 any more, then doing the "img/127.5 + 1" process not exactly obey the thick transform img to -1~1, don't know weather it matters
- The 'x + eps * (G(z) - x)' step, the dimention of eps is '[batch_size, h, w, c]' or 
'[batchsize, 1, 1, 1] and then broadcast to [batch_size, h, w, c]' ?

### 2. ACGAN in mnist with WGAN-GP loss
#### What is learned
- BN ���� RuLU�����׹���
- ����ṹ��һ�����Գ����ģ��� G ���� z �� label �� embedding �������Ҳ����ʵ������ÿ��model��Ӧ��lr�ȳ�����û���á�
- ע��loss����������Ĺ�ϵ��loss��ע��label��logits��λ�ñ�Ū����...֮ǰ�и�bug����sigmoid�������飬���loss����е����...
- WGAN-GP�������Ŀ�ľ���Ϊ�˸�����train GAN������WGAN-GPӦ�ñ�DCGAN�ȸ�����train�����κ�ģ�͸����׵�����Ч���������õ���һ�������ɼ�WGAN-GP���ȶ�һ�㡣
- ��һ�ֽṹ��̫��������trainһ��classifier�Ļ�û�й���D�о�����feature���ڶ��ֽṹ�Ӿ�����feature��ʼ�ֱ�trainfc��ɵ�classifier���ܸ�����


#### TODO
- Add noise to true label. 
- ��ôд��ͨ�õ� fc_relu_bn ��...(slim��ֻ��fc_bn_relu)

#### Q
- WGAN-GP������loss�仯��ʲô���ģ�D��������Gһֱ�����Ƿ��Ѿ�optimal�ˣ�
- ��Ϊ�����һ��fc��ֿ�train class��logits������\����trian...��

#### Result:
![Alt text](https://github.com/1160300630/GANs-Playground/blob/master/images/AC-GAN(WGAN-GP%20loss).jpg)


### 3. WGAN-GP in mnist

#### Result:
![Alt text](https://github.com/1160300630/GANs-Playground/blob/master/images/WGAN-GP.jpg)

### 4. WGAN-GP in comic

#### Result:

![Alt text](https://github.com/1160300630/GANs-Playground/blob/master/images/WGAN-GP(vanilla, 100 epoch).jpg)

###  5. AC-GAN(WGAN-GP loss)

#### Result:

![Alt text](https://github.com/1160300630/GANs-Playground/blob/master/images/c1.jpg)

![Alt text](https://github.com/1160300630/GANs-Playground/blob/master/images/c2.jpg)

![Alt text](https://github.com/1160300630/GANs-Playground/blob/master/images/c3.jpg)

### 