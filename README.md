#### This is a simple GAN Zoo for studying

---

## Mostly using the Comics dataset

### 1. WGAN-GP
- After resizing from 96 to 64 then the images are not belong to 0-255 any more, then doing the "img/127.5 + 1" process not exactly obey the thick transform img to -1~1, don't know weather it matters
- The 'x + eps * (G(z) - x)' step, the dimention of eps is '[batch_size, h, w, c]' or 
'[batchsize, 1, 1, 1] and then broadcast to [batch_size, h, w, c]' ?

### 2. ACGAN in mnist with WGAN-GP loss
#### What is learned
- TODO: add noise to label. ����ǩ���noise������Ӧ���Ƿ�ֹ `mode collapse` ,���� noise ʱACGAN����ͬ�� condition �Ѿ����ֳ��˽Ϻõ����� z �� noise ���Կ� ` mode collapse`��
- ����ṹ��һ������Գ����ģ��� G ���� z �� label �� embedding �������bn �� activation function ��˳�򲻶Ի������D �� ln �滻 bn �������Ҳ����ʵ������ÿ��model��Ӧ��lr�ȳ�����û���á�
- ע��loss����������Ĺ�ϵ��loss��ע��label��logits��λ�ñ�Ū����...֮ǰ�и�bug����sigmoid�������飬���loss����е����...
- WGAN-GP�������Ŀ�ľ���Ϊ�˸�����train GAN������WGAN-GPӦ�ñ�DCGAN�ȸ�����train�����κ�ģ�͸����׵�����Ч���������õ���һ������

#### Q
- WGAN-GP������loss�仯��ʲô���ģ�

#### Result:
![Alt text](https://github.com/1160300630/GANs-Playground/blob/master/images/AC-GAN(WGAN-GP%20loss).jpg)


### 3. WGAN-GP in mnist

#### Result:
![Alt text](https://github.com/1160300630/GANs-Playground/blob/master/images/WGAN-GP.jpg)
