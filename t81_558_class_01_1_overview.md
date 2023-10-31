# T81-558: ���������Ӧ��

#### ģ��1��Python ����֪ʶԤ��

- ��ʦ��[Jeff Heaton](https://sites.wustl.edu/jeffheaton/)��[ʥ·��˹��ʢ�ٴ�ѧ](https://engineering.wustl.edu/Programs/Pages/default.aspx)��ά����ѧԺ��
- ������Ϣ�������[�γ���վ](https://sites.wustl.edu/jeffheaton/t81-558/)��

---

## ģ��1Ŀ¼

1. **�γ̸��� [[Video]](https://www.youtube.com/watch?v=r7eExQWKzdc&list=PLjy4p-07OYzuy_lHcRW8lPTLPTTOmUpmi) [Notebook]
   **
2. Python ���� [[Video]](https://www.youtube.com/watch?v=ZAOOinw51no&list=PLjy4p-07OYzuy_lHcRW8lPTLPTTOmUpmi) [Notebook]
3. Python
   �б��ֵ䡢���ϡ�JSON [[Video]](https://www.youtube.com/watch?v=5jZWWLO71bE&list=PLjy4p-07OYzuy_lHcRW8lPTLPTTOmUpmi) [Notebook]
4. �ļ����� [[Video]](https://www.youtube.com/watch?v=CPrp1Sm-AhQ&list=PLjy4p-07OYzuy_lHcRW8lPTLPTTOmUpmi) [Notebook]
5. ������Lambda ��
   Map/Reduce [[Video]](https://www.youtube.com/watch?v=DEg8a22mtBs&list=PLjy4p-07OYzuy_lHcRW8lPTLPTTOmUpmi) [Notebook]

�ۿ�����һ��������� Python ����

- [M1 Mac ��װ PyTorch](https://www.youtube.com/watch?v=VEDy-c5Sk8Y)

---

## Google CoLab Instructions

���´����ȷ�� Google CoLab �������У�������Ҫʱӳ�� Google Drive��

``` python
try:
    from google.colab import drive
    COLAB = True
    print("Note: using Google CoLab")
except:
    print("Note: not using Google CoLab")
    COLAB = False
```

## 1.1 ����

���ѧϰ��һ���µ������缼�� [[Cite:lecun2015deep]](https://www.nature.com/articles/nature14539)��
ͨ��ʹ�ø߼�ѵ��������������ļܹ���������ڿ���ѵ�����и������Ե������硣
�Ȿ������˶������ʹ����������硢���򻯵�Ԫ��ReLU��������������ѭ�������硣
�����ܼ��㣨HPC��������ʾ���������ͼ�δ���Ԫ��GPUs��������������������ѧϰ�����ѧϰ����ģ��ѧϰ��Ϣ��Σ�������������ԵĹ��ܡ�
�ص���Ҫ���������ѧϰ��Ӧ�ã��Լ�һЩ�������ѧϰ��ѧ�����Ľ��ܡ�
���߽�ʹ��Python������Թ���һ�����ѧϰģ�ͣ����ڼ���ʵ�����ݼ��Ͻ�����Щ����Ľ�� [[Cite:lecun2015deep]](https://www.deeplearningbook.org/)��

### 1.1.1 ���ѧϰ����Դ

�������ǻ���ѧϰ��һ������ʵ��������������20����40����ͱ����룬�������������о�������˥��
��ǰһ�������ѧϰʼ��2006�꣬��������Ҫ���Ƕ�ѵ���㷨�ĸĽ����� Geoffrey
Hinton [[Cite:hinton2006fast]](https://www.mitpressjournals.org/doi/abs/10.1162/neco.2006.18.7.1527) �����
���ַ������������������������磨��������磩��Ч��ѵ����
��λ�о��߶��������о���������Ҫ���ס�����ʼ���ƶ��������о����������� ups �� downs �С���Щ��λΰ����о�����ͼ1.LUM����ʽչʾ��

**ͼ1.LUM: ������ܳ�����**

<p> <img src="images/class_01/01_LUM.jpg"></p>

��ǰ artificial neural network��ANN���о����������ѧϰ�������о��ߣ���ͼ�е�˳����֣�

- [Yann LeCun](http://yann.lecun.com/) - Facebook ŦԼ��ѧ: ��������磨CNN���ڹ�ѧ�ַ�ʶ��ͼ�����Ӿ��е�Ӧ�á��������Ĵ�ʼ�ˡ�
- [Geoffrey Hinton](http://www.cs.toronto.edu/~hinton/) - Google ���׶��ѧ: �����緽��Ĺ㷺���������ѧϰ�Ĵ����ߺ������練�򴫲�������������/�����ߡ�
- [Yoshua Bengio](http://www.iro.umontreal.ca/~bengioy/yoshua_en/index.html) - Botler AI ����������ѧ:
  �����ѧϰ��������ͻ���ѧϰ�����˹㷺���о���
- [Andrew Ng](http://www.andrewng.org/) - Baidu ˹̹����ѧ: �����ѧϰ�������缰���ڻ����˼����е�Ӧ�ý����˹㷺���о���

Geoffrey Hinton��Yann LeCun �� Yoshua Bengio
��������ѧϰ����߽���֮һ��[Turing Award](https://www.acm.org/media-center/2019/march/turing-award-2018)��

### 1.1.2 ʲô�����ѧϰ

����Ĺ�ע�������ѧϰ������һ�����еĻ���ѧϰ���ͣ�������20����80������е������硣���ѧϰ��ԭʼ������֮��Ĳ��켸��û�С�
����һֱ�ܹ������ͼ�����������硣���������ֻ�Ǿ�������������磬�����ܹ�����/������������硣Ȼ������ȱ��ѵ�����ǵ���Ч���������ѧϰ�ṩ��ѵ��������������Ч������

������ѧϰ��һ�ֻ���ѧϰ���ͣ���ô����Ӧ˼��������ѧϰ��ʲô������ͼ 2.ML-DEV ��ʾ��

**ͼ2.ML-DEV: ����ѧϰ�ʹ�ͳ�������֮�������**

<p> <img src="images/class_01/02_ML-DEV.png"></p>

- ��ͳ������� - ����Ա����������Ϊ�˽�����ת��Ϊ����������
- ����ѧϰ - ����Ա����ģ�Ϳ���ѧϰ���������Բ�������������

�о���Ա�ѽ�����ѧϰӦ������಻ͬ�����򡣱��γ�̽�����������Ӧ�õ������ض�������ͼ 3.ML-DOM ��ʾ��

**ͼ3.ML-DOV: ����ѧϰӦ��**

<p> <img src="images/class_01/03_ML-DOM.jpg"></p>

- ������Ӿ���ʹ�û���ѧϰ������Ӿ������е�ģʽ�����磬һ��ͼƬ��è���ǹ���
- ������ݣ�������������ֵ����������Ԥ����һ������ֵ����Ϊ��������磬����ʹ���β���ļ��ֲ�����Ԥ�����֡���������ͨ������Ϊ������ݡ�
- ��Ȼ���Դ���NLP�������ѧϰ Transformers ���׸ı��� NLP�������ı��������ɸ����ı���ͼ�����ࡣ
- ǿ��ѧϰ��ǿ��ѧϰѵ��������ѡ�����ڽ��еĶ������Ա��㷨��������������ѷ�ʽ�������
- ʱ�����У�ʹ�û���ѧϰ��ʱ���ģʽ�����͵�ʱ������Ӧ���ǽ���Ӧ�á�����ʶ��������Ȼ���Դ���NLP����
- ����ģ�ͣ����������ѧϰ�����������µ�ԭʼ�ϳ����ݡ����ǽ��о� StyleGAN����ѧϰ������ѵ���ڼ俴����ͼ�����Ƶ���ͼ��

### 1.1.3 �ع顢���༰����

����ѧϰ�о��ӹ���ļලѧϰ���޼ලѧϰ�ĽǶ����������⡣����֪��ѵ������ÿ����Ŀ����ȷ���ʱ���ͻᷢ���ලѧϰ��
��һ���棬�޼ලѧϰ���ò�֪����ȷ�����ѵ���������ѧϰ֧�ּලѧϰ���޼ලѧϰ��Ȼ��������������ǿ���ͶԿ���ѧϰ��
ǿ��ѧϰ�̵���������ݻ���ִ�в������Կ���ѧϰ�������������໥�Կ����������޷��ṩ��ȷ���ʱ����ѧϰ��
�Կ���ѧϰ�������������໥�Կ����������޷��ṩ��ȷ���ʱ����ѧϰ���о���Ա��������µ����ѧϰѵ��������

����ѧϰ��ҵ��ͨ�����ලѧϰ��Ϊ����ͻع�, ����������ܽ��ܲ������ݲ���Ͷ�ʷ��շ���Ϊ���ջ�ȫ��
ͬ�����ع����������һ�����֣������ܲ�����ͬ�����ݲ����ط������֡� ���⣬���������ͬʱ�������ع�ͷ��������

��������ǿ��ķ���֮һ�������������������������಻ͬ�����ͣ����磺

- ͼ��
- ���Ա�ʾ�ı�����Ƶ������ʱ�����е�һϵ������
- �ع���
- �������

### 1.1.4 Ϊʲôѡ�����ѧϰ��

���ڱ�����ݣ�������ı���ͨ������������ģ�����Ը��ã����磺

- ֧����������SVM��
- ���ɭ�֣�RF��
- �ݶ���������GBM��

������ģ��һ�������������ִ��**����**��**�ع�**�� ��Ӧ������Ե�ά�ı����������ʱ����������粻һ��������ģ�������������׼ȷ�ԡ�
Ȼ������������Ƚ��Ľ��������������ͼ����Ƶ���ı�����Ƶ���ݵ���������硣

## 1.2 Python ���ѧϰ

���齫ʹ�� Python 3.x ������ԡ� Python ��Ϊһ�ֱ�����ԣ������ѧϰӵ����㷺��֧�֡� Python �����������е����ѧϰ����ǣ�

- [TensorFlow/Keras](https://www.tensorflow.org/) (Google)
- [PyTorch](https://pytorch.org/) (Facebook)

������Ҫ��ע PyTorch�� ��������������ǽ�ֱ��ʹ�� PyTorch�� Ȼ�������ǽ����õ�����������ɸ��߼������������ǿ��ѧϰ�����ɶԿ�������ȡ�
��Щ��������������ڲ�ʹ�� PyTorch �� Keras�� ��ѡ����Щ���ǻ������жȺ�Ӧ�ó��򣬶����������Ƿ�ʹ�� PyTorch ���� Keras��

Ҫ�ɹ�ʹ�ñ��飬�������ܹ������ִ������ PyTorch �������ѧϰ�� Python ���롣��������ѡ�����ʵ�ִ�Ŀ�ģ�

- ��װ Python��PyTorch ��һЩ IDE��Jupyter��TensorFlow �ȣ���
- ������ʹ�� Google CoLab��������ѷ��� GPU��
-

���²��ֽ�����������ڱ��ؼ�����ϰ�װ Python �Ĺ��̡�
�˹����� Windows��Linux �� Mac �ϻ�����ͬ�� �й��ض�����ϵͳ˵��������ı��ĵ�ǰ��� YouTube �̳���Ƶ֮һ��

### 1.2.1 Tokens �� Keys

����ӵ�иÿγ̵Ķ�� Tokens �� Keys��

- ��ҵ�ύ API Keys���������Ͽ�ǰͨ�������ʼ��յ� API ��Կ������ʹ�ô˼����ύ��ҵ��
- Hugging Face Keys������Ҫ��¼ HuggingFace ����ȡ�����Կ��������ĳЩԤѵ��ģ�͡�
- OpenAI ��Կ��������ô���Կ������ ChatGPT �������ҵ 6������Կ���ṩ��ѧ����

### 1.2.2 ������Python��װ

��װ Python ��������ʹ�����´������������ Python �Ϳ�汾��������� GPU���������Լ�� Keras �Ƿ�ʶ������

```python
# What version of Python do you have?
import platform
import sys

import pandas as pd
import sklearn as sk
import torch

has_gpu = torch.cuda.is_available()
has_mps = getattr(torch, 'has_mps', False)
device = "mps" if getattr(torch, 'has_mps', False) else "gpu" if torch.cuda.is_available() else "cpu"

print(f"Python Platform: {platform.platform()}")
print(f"PyTorch Version: {torch.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
print("GPU is", "available" if has_gpu else "NOT AVAILABLE")
print("MPS (Apple Metal) is", "AVAILABLE" if has_mps else "NOT AVAILABLE")
print(f"Target device is {device}")

```

```
Python Platform: macOS-13.5.2-arm64-arm-64bit
PyTorch Version: 2.0.1

Python 3.9.18 | packaged by conda-forge | (main, Aug 30 2023, 03:53:08) 
[Clang 15.0.7 ]
Pandas 2.1.0
Scikit-Learn 1.3.0
GPU is NOT AVAILABLE
MPS (Apple Metal) is AVAILABLE
Target device is mps

```





