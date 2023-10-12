# fedml答辩



**课题1：边端场景下的分布式训练系统框架搭建**
**内容：联邦学习是边端场景下优选的分布式训练架构，它让多方数据分开训练成为可能，**
**可以使用高效的加密算法来保护用户隐私和数据安全：**
**目标：**
**1、完成不少于10个边缘节点与边服务器的分布式训练，使用K8S 进行多Pod 分层部**
**署。**
**2、提供联邦学习训练场景（图像识别、自然语言处理、推荐系统等）**
**3、集成不少于一个的加密算法，保护用户数据隐私安全**

## 运行配置fedml_config.yaml

```yaml
# common_args:
#   这些是用于训练的常见参数。
common_args:
  training_type: "cross_silo"  # 训练类型: 跨边缘节点
  scenario: "horizontal"  # 场景: 水平联邦学习
  using_mlops: false  # 是否使用 MLOps: 否
  random_seed: 0  # 随机种子: 0

# environment_args:
#   这些是与环境设置相关的参数。
environment_args:
  bootstrap: config/bootstrap.sh  # 引导脚本路径: config/bootstrap.sh

# data_args:
#   这些是与数据配置相关的参数。
data_args:
  dataset: "mnist"  # 数据集: mnist
  data_cache_dir: ~/fedml_data  # 数据缓存目录: ~/fedml_data
  partition_method: "hetero"  # 数据划分方法: 异质划分
  partition_alpha: 0.5  # 数据划分系数: 0.5

# model_args:
#   这些是与模型配置相关的参数。
model_args:
  model: "lr"  # 模型: 逻辑回归
  model_file_cache_folder: "./model_file_cache"  # 模型文件缓存目录: ./model_file_cache（将由服务器自动填充）
  global_model_file_path: "./model_file_cache/global_model.pt"  # 全局模型文件路径: ./model_file_cache/global_model.pt

# train_args:
#   这些是与训练配置相关的参数。
train_args:
  federated_optimizer: "FedAvg"  # 联邦优化器: FedAvg
  client_id_list: "[1]"  # 客户端 ID 列表: "[1]"
  client_num_in_total: 1  # 客户端总数: 1
  client_num_per_round: 1  # 每轮的客户端数: 1
  comm_round: 20  # 通信轮数: 20
  epochs: 5  # 训练轮数: 5
  batch_size: 32  # 批量大小: 32
  client_optimizer: sgd  # 客户端优化器: sgd
  learning_rate: 0.03  # 学习率: 0.03
  weight_decay: 0.0001  # 权重衰减: 0.0001

# validation_args:
#   这些是与验证配置相关的参数。
validation_args:
  frequency_of_the_test: 1  # 测试频率: 1

# device_args:
#   这些是与设备配置相关的参数。
device_args:
  worker_num: 1  # 工作节点数: 1
  using_gpu: false  # 是否使用 GPU: false
  gpu_mapping_file: config/gpu_mapping.yaml  # GPU 映射文件路径: config/gpu_mapping.yaml
  gpu_mapping_key: mapping_default  # GPU 映射键: mapping_default

# comm_args:
#   这些是与通信配置相关的参数。
comm_args:
  backend: "MQTT_S3"  # 后端: MQTT_S3
  mqtt_config_path: config/mqtt_config.yaml  # MQTT 配置文件路径: config/mqtt_config.yaml
  s3_config_path: config/s3_config.yaml  # S3 配置文件路径: config/s3_config.yaml
  # 如果要使用自定义的 MQTT 或 S3 服务器作为训练后端，请取消注释并设置以下行。
  #customized_training_mqtt_config: {'BROKER_HOST': 'your mqtt server address or domain name', 'MQTT_PWD': 'your mqtt password', 'BROKER_PORT': 1883, 'MQTT_KEEPALIVE': 180, 'MQTT_USER': 'your mqtt user'}
  #customized_training_s3_config: {'CN_S3_SAK': 'your s3 aws_secret_access_key', 'CN_REGION_NAME': 'your s3 region name', 'CN_S3_AKI': 'your s3 aws_access_key_id', 'BUCKET_NAME': 'your s3 bucket name'}

# tracking_args:
#   这些是与跟踪配置相关的参数。
tracking_args:
  # 当在 MLOps 平台(open.fedml.ai)上运行时，默认日志路径为 ~/fedml-client/fedml/logs/ 和 ~/fedml-server/fedml/logs/
  enable_wandb: false  # 是否启用 WandB: false
  wandb_key: ee0b5f53d949c84cee7decbe7a629e63fb2f8408  # WandB API 密钥
  wandb_project: fedml  # WandB 项目名称: fedml
  wandb_name: fedml_torch_fedavg_mnist_lr  # WandB 运行名称: fedml_torch_fedavg_mnist_lr

# lsa_args:
#   这些是与 LSA（局部敏感哈希）配置相关的参数。
#  prime_number: 2 ** 15 - 19
#  precision_parameter: 10

```



![image-20230628004047004](/Users/houjinchang/Library/Application Support/typora-user-images/image-20230628004047004.png)

![image-20230628004016525](/Users/houjinchang/Library/Application Support/typora-user-images/image-20230628004016525.png)

![image-20230628004116962](/Users/houjinchang/Library/Application Support/typora-user-images/image-20230628004116962.png)

![image-20230628004149896](/Users/houjinchang/Library/Application Support/typora-user-images/image-20230628004149896.png)![image-20230628004210250](/Users/houjinchang/Library/Application Support/typora-user-images/image-20230628004210250.png)

![image-20230628004244232](/Users/houjinchang/Library/Application Support/typora-user-images/image-20230628004244232.png)

![image-20230628020303864](/Users/houjinchang/Library/Application Support/typora-user-images/image-20230628020303864.png)

![image-20230628020622969](/Users/houjinchang/Library/Application Support/typora-user-images/image-20230628020622969.png)

### lightsecagg实现

/Users/houjinchang/miniconda3/python.app/Contents/lib/python3.10/site-packages/fedml/core/mpc/lightsecagg.py









## ### cifar10 10种物体分类

tensor(7381, device='mps:0') 10000 tensor(0.7381, device='mps:0') 准确率 80%

![img](https://img-blog.csdnimg.cn/20190730170546931.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly95dW55YW5pdS5ibG9nLmNzZG4ubmV0,size_16,color_FFFFFF,t_70)

![image-20230628220550911](/Users/houjinchang/Library/Application Support/typora-user-images/image-20230628220550911.png)

![image20230111004202113](http://localhost:63342/4f800f8a-bbed-4dd8-b03c-00449c9f6698/1440500441/fileSchemeResource/64f06ae85c3a8ee1a295436832a539c8-image-20230111004202113.png?_ijt=q9qisnh8o46o1998h3enbm1d77)



## 答辩ppt 要点

### 背景介绍

1. 联邦学习（Federated Learning）的背景：
   - 传统机器学习方法通常需要将数据集集中在一个中心化的服务器上进行训练，这可能涉及数据隐私和安全性问题。
   - 联邦学习是一种分布式机器学习方法，旨在让多个参与方共同训练一个共享的机器学习模型，而无需将数据集集中在一处。
   - 联邦学习的背景可以追溯到隐私保护、数据安全性和分散数据资源的需求。
2. 同态加密（Homomorphic Encryption）的背景：
   - 在传统的数据加密方法中，加密后的数据无法直接进行计算，必须解密后才能进行操作，这可能导致数据泄露的风险。
   - 同态加密是一种特殊的加密技术，允许对加密数据进行计算，而不需要解密操作，从而保护数据的隐私性。
   - 同态加密的背景可以追溯到隐私保护、安全计算和云计算的需求。
3. 联邦学习与同态加密的结合：
   - 联邦学习和同态加密可以结合使用，以解决联邦学习中的隐私和安全性问题。
   - 同态加密允许在加密状态下对数据进行计算，使得在联邦学习中，参与方可以在保护数据隐私的同时共享加密的模型参数或计算结果。
   - 结合同态加密的联邦学习方法可以实现隐私保护的模型训练和推断，从而促进数据合作与共享，同时确保数据的隐私性。
4. 应用领域和挑战：
   - 联邦学习和同态加密在多个领域都有潜在的应用，如医疗健康、金融、智能交通等，可以促进数据共享和模型协作。
   - 然而，联邦学习和同态加密也面临着一些挑战，如计算效率、通信开销、安全性保证等方面的问题，需要进一步研究和改进。

**请注意，这只是一个简要的背景介绍，你可以根据需要进行扩展和进一步的深入讲解。**

### 技术手段

- 图像识别

  1. 数据集概述：CIFAR-10数据集由60000个32x32像素的彩色图像组成，涵盖了10个不同的类别。每个类别包含6000个图像样本。这些类别包括：飞机、汽车、鸟类、猫、鹿、狗、青蛙、马、船和卡车。
  2. 数据集划分：CIFAR-10数据集通常被划分为训练集和测试集。训练集包含50000个样本，用于训练模型的参数。测试集包含10000个样本，用于评估模型的性能。
  3. 图像识别任务：CIFAR-10数据集用于图像识别任务，即将输入的图像分为10个不同的类别。对于给定的图像，模型需要学习从图像中提取特征，并将其分类到正确的类别。
  4. 模型构建：为了进行CIFAR-10图像识别，常用的模型是卷积神经网络（Convolutional Neural Network，CNN）。CNN具有多个卷积层和池化层，用于提取图像中的特征，并通过全连接层进行分类。模型的架构和参数设置对于获得良好的分类性能至关重要。
  5. 训练和优化：使用CIFAR-10数据集进行图像识别时，需要将训练集上的图像输入到CNN模型中，并使用损失函数进行模型训练。常用的优化算法如随机梯度下降（Stochastic Gradient Descent，SGD）用于调整模型参数，以最小化损失函数。
  6. 模型评估：在训练完成后，使用测试集上的图像对模型进行评估。通过计算分类准确率（Accuracy）来衡量模型的性能，即模型正确分类的图像数量与总图像数量之比。
  7. 结果分析和改进：通过分析模型在CIFAR-10数据集上的表现，可以识别出模型的优点和缺点。根据分析结果，可以采取一系列改进措施，包括调整模型架构、改进数据预处理、调整超参数等，以提高模型的性能。

  CIFAR-10数据集的广泛使用促进了图像识别算法的研究和发展。该数据集对于研究人员和开发者来说是一个标准的基准数据集，用于评估不同图像识别模型的性能和比较

- fedml 

  ```
  FedML（Federated Learning）是一个开源的联邦学习框架，旨在支持研究人员和开发者开展联邦学习的实验和应用。以下是FedML框架的技术介绍：
  
  联邦学习支持：FedML框架提供了一套完整的工具和算法，用于支持联邦学习的实施。它包括了各种经典和先进的联邦学习算法，如FedAvg、FedProx、FedAdapt等，以满足不同的应用需求。
  
  分布式架构：FedML框架采用分布式架构，可以有效地处理大规模的分布式数据和计算。它支持在不同设备和节点之间进行通信和协作，以实现模型的聚合和参数更新。
  
  可扩展性和灵活性：FedML框架具有良好的可扩展性和灵活性，可以适应不同规模和复杂度的任务。它支持在多个参与方之间进行联邦学习，并可以轻松地扩展到更多参与方和更大的数据集。
  
  隐私保护：FedML框架关注隐私保护，在联邦学习中采取了一系列隐私增强技术，如差分隐私、同态加密等。这些技术可以有效地保护参与方的隐私数据，并确保在联邦学习过程中的数据安全性。
  
  模块化设计：FedML框架采用模块化的设计，将联邦学习的各个组件分解为独立的模块，如数据加载、模型定义、参数更新等。这种设计使得用户可以根据需求进行定制和扩展，以适应不同的联邦学习场景。
  
  高性能和优化：FedML框架注重提高联邦学习的性能和效率。它利用并行计算和优化算法，减少通信开销和计算负载，提高模型训练和推理的速度。
  
  文档和示例：FedML框架提供了详细的文档和示例代码，以帮助用户了解和使用该框架。文档中包含了框架的安装说明、使用指南和算法原理的解释，示例代码则展示了如何使用框架进行联邦学习实验。
  
  总的来说，FedML框架为联邦学习提供了一个强大的工具和平台，使研究人员和开发者能够更轻松地进行联邦学习的实验和应用。它的设计目标是提供高效、可扩展和隐私安全的联邦学习解决方案。
  ```

   - 差分隐私

     ```
     FedML框架提供了差分隐私（Differential Privacy）的实现，用于保护参与方的隐私数据。差分隐私是一种在联邦学习中广泛使用的隐私增强技术，通过向模型更新过程中的梯度添加噪声，以保护个体数据的隐私。
     
     FedML框架中实现差分隐私的关键步骤如下：
     
     定义隐私预算：在差分隐私中，隐私预算是一个重要的概念，用于量化和控制隐私泄露的程度。用户需要定义隐私预算的大小，通常用ε来表示，数值越小表示隐私保护程度越高。
     
     计算梯度的敏感性：在差分隐私中，梯度敏感性用于衡量模型对个体数据的敏感程度。在FedML框架中，根据参与方的本地数据计算每个参与方的梯度，并计算梯度的敏感性。
     
     添加噪声：在模型更新的过程中，FedML框架通过向参与方的梯度添加噪声来实现差分隐私。噪声的大小受到隐私预算的控制，通常使用拉普拉斯噪声或高斯噪声。
     
     聚合模型：在参与方完成梯度更新后，FedML框架使用安全聚合算法将参与方的模型参数进行聚合，以生成全局模型。聚合过程需要确保隐私泄露的控制，因此也需要考虑隐私预算的分配。
     
     隐私保护的评估：FedML框架还提供了隐私保护的评估功能，可以对联邦学习过程中的隐私泄露进行量化和评估。这有助于用户了解差分隐私的效果，并调整隐私预算和噪声的参数。
     
     需要注意的是，差分隐私只是一种隐私增强技术，它不能完全消除隐私泄露的可能性。在实际应用中，用户还需要根据具体需求和安全要求来选择适当的隐私预算和差分隐私参数。
     
     总的来说，FedML框架通过定义隐私预算、计算梯度敏感性、添加噪声和聚合模型等步骤，实现了差分隐私的保护机制，为联邦学习提供了隐私安全的解决方案。
     ```

      adsf阿迪舒服啊但是啊的发a 啊阿萨德撒的饭撒的饭阿迪舒服secagg啊多少分adsf ads 

     

   - lightsecagg

   - kubernates

   - trainer

   - fedavg

   - aggrater （secagg 安全聚合方案）

     

- secagg------>lightsecagg

  

  

  

  - 安装 miniconda3
  - conda create -n fedmlenv python=3.8
  - conda activate fedmlenv
  - 再执行 pip install fedml

  ```python
  import random
  
  total_runs = 5
  for run in range(total_runs):
    wandb.init(
        project="basic-intro", 
        name=f"experiment_{run}", 
        config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "MINIST",
        "epochs": 10,
        })
    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset
        wandb.log({"acc": acc, "loss": loss})
    wandb.finish()
  ```

