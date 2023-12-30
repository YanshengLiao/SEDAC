import pandas as pd
import numpy as np
import math
from transformers import pipeline
import torch.utils.data as Data
from CVAE_model import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import warnings

# Text representation
project = pd.read_csv("data/Ambari.csv")

feature_extraction = pipeline('feature-extraction', model="distilbert-base-uncased",
                              tokenizer="distilbert-base-uncased", truncation = True, max_length = 512)
features = feature_extraction(list(project["sentences"]))

labels = pd.DataFrame(project["Security"])
index = torch.tensor(labels.values, dtype=torch.int64)

br = torch.mean(torch.tensor(features[0][0]),dim = 0).unsqueeze(0)
for i in range(len(project)-1):
    br = torch.cat([br,torch.mean(torch.tensor(features[i+1][0]),dim = 0).unsqueeze(0)],dim=0)

br_np = np.array(br)
br_df = pd.DataFrame(br_np)

# CVAE model training
equal_zero_index = (labels != 1).values
equal_one_index = ~equal_zero_index

pass_feature = np.array(br_df[equal_zero_index])
fail_feature = np.array(br_df[equal_one_index])

diff_num = len(pass_feature) - len(fail_feature)

min_batch = 40
batch_size = min_batch if len(labels) >= min_batch else len(labels)
torch_dataset = Data.TensorDataset(br,index)
loader = Data.DataLoader(dataset=torch_dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         )
input_dimension = len(br_df.values[0])
hidden_dimension = math.floor(math.sqrt(input_dimension))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cvae = CVAE(encoder_layer_sizes=[input_dimension, hidden_dimension],
            latent_size=5,
            decoder_layer_sizes=[hidden_dimension, input_dimension],
            conditional=True,
            num_labels=2).to(device)
optimizer = torch.optim.Adam(cvae.parameters(), lr=0.005)
EPOCH = 200
for epoch in range(EPOCH):
    cvae.train()
    train_loss = 0
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        recon_x, mu, logvar, z = cvae(x, y)
        loss = loss_fn(recon_x, x, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    if epoch % 100 == 0:
        print('====>CVAE training... Epoch: {} Average loss: {:.4f}'.format(epoch,
                                                                           train_loss / len(loader.dataset)))

# Synthesize bug report vectors
with torch.no_grad():
  c = torch.ones(diff_num).long().unsqueeze(1).to(device)
  z = torch.randn([c.size(0), 5]).to(device)
  x = cvae.inference(z, c=c).to("cpu").numpy()
br_np = np.array(br_df)
compose_feature = np.vstack((br_np, x))

label_np = np.array(labels)
gen_label = np.ones(diff_num).reshape((-1, 1))
compose_label = np.vstack((label_np.reshape(-1, 1), gen_label))

labels = pd.DataFrame(compose_label, columns=['Security'], dtype=float)
br_df = pd.DataFrame(compose_feature, columns=br_df.columns, dtype=float)

data = pd.concat([br_df, labels], axis=1)

# SBR identification
warnings.filterwarnings("ignore")

classifier = LogisticRegression()

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
pd_list, pf_list, g_measure_list = [], [], []

for train_index,test_index in skfold.split(br_df,labels):
    X_train_fold = br_df.iloc[train_index]
    y_train_fold = labels.iloc[train_index]
    X_test_fold = br_df.iloc[test_index]
    y_test_fold = labels.iloc[test_index]

    classifier.fit(X_train_fold, np.ravel(y_train_fold))
    y_pred_lr = classifier.predict(X_test_fold)

    cm_lr = confusion_matrix(y_test_fold, y_pred_lr).ravel()
    pd_lr = cm_lr[3]/(cm_lr[3]+cm_lr[2])
    pf_lr = cm_lr[1]/(cm_lr[1]+cm_lr[0])
    g_measure_lr = (2*pd_lr*(1-pf_lr))/(pd_lr+(1-pf_lr))
    pd_list.append(pd_lr)
    pf_list.append(pf_lr)
    g_measure_list.append(g_measure_lr)

print(round(np.average(g_measure_list),4)*100, round(np.average(pd_list),4)*100,
      round(np.average(pf_list),4)*100)
